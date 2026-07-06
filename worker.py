import base64
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("worker")

ACE_ENDPOINT       = "https://api.acemusic.ai/v1/chat/completions"
ACE_MODEL          = "acemusic/acestep-v1.5-turbo"
ACE_MAX_RETRIES    = 6
MIN_AUDIO_DURATION = 60
PLAYLIST_CAP       = 5000
TIME_BUDGET_S      = 35 * 60

GENRES = [
    {"name": "electro-swing",  "vocal_language": "en", "ace_duration": 190},
    {"name": "rock",           "vocal_language": "en", "ace_duration": 190},
    {"name": "pop",            "vocal_language": "en", "ace_duration": 190},
    {"name": "k-pop",          "vocal_language": "ko", "ace_duration": 120},
    {"name": "lofi-chillout",  "vocal_language": "en", "ace_duration": 190},
]

GEMINI_MODEL = "gemini-3.1-flash-lite"


def get_drive_service():
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build
    creds = Credentials(
        token=None,
        refresh_token=os.environ["DRIVE_REFRESH_TOKEN"],
        client_id=os.environ["DRIVE_CLIENT_ID"],
        client_secret=os.environ["DRIVE_CLIENT_SECRET"],
        token_uri="https://oauth2.googleapis.com/token",
        scopes=["https://www.googleapis.com/auth/drive.file"],
    )
    return build("drive", "v3", credentials=creds)


def get_or_create_folder(drive, name, parent_id=None):
    q = f"name='{name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
    if parent_id:
        q += f" and '{parent_id}' in parents"
    res = drive.files().list(q=q, fields="files(id)").execute()
    files = res.get("files", [])
    if files:
        return files[0]["id"]
    meta = {"name": name, "mimeType": "application/vnd.google-apps.folder"}
    if parent_id:
        meta["parents"] = [parent_id]
    return drive.files().create(body=meta, fields="id").execute()["id"]


def list_children(drive, parent_id, mime_type=None):
    q = f"'{parent_id}' in parents and trashed=false"
    if mime_type:
        q += f" and mimeType='{mime_type}'"
    items, token = [], None
    while True:
        res = drive.files().list(
            q=q, fields="nextPageToken, files(id, name)", pageToken=token, pageSize=200
        ).execute()
        items.extend(res.get("files", []))
        token = res.get("nextPageToken")
        if not token:
            break
    return items


def find_file(drive, filename, folder_id):
    res = drive.files().list(
        q=f"name='{filename}' and trashed=false and '{folder_id}' in parents", fields="files(id)"
    ).execute()
    files = res.get("files", [])
    return files[0]["id"] if files else None


def download_file(drive, file_id, dest_path):
    from googleapiclient.http import MediaIoBaseDownload
    req = drive.files().get_media(fileId=file_id)
    with open(dest_path, "wb") as f:
        downloader = MediaIoBaseDownload(f, req)
        done = False
        while not done:
            _, done = downloader.next_chunk()


def upload_file(drive, local_path, folder_id, mime_type, remote_name=None):
    from googleapiclient.http import MediaFileUpload
    path = Path(local_path)
    meta = {"name": remote_name or path.name, "parents": [folder_id]}
    media = MediaFileUpload(str(path), mimetype=mime_type, resumable=True)
    return drive.files().create(body=meta, media_body=media, fields="id").execute()["id"]


def move_file(drive, file_id, from_folder_id, to_folder_id):
    drive.files().update(
        fileId=file_id, addParents=to_folder_id, removeParents=from_folder_id, fields="id,parents"
    ).execute()


def delete_file(drive, file_id):
    drive.files().delete(fileId=file_id).execute()


def read_json_file(drive, file_id):
    raw = drive.files().get_media(fileId=file_id).execute()
    return json.loads(raw)


def write_json(drive, data, filename, folder_id):
    from googleapiclient.http import MediaInMemoryUpload
    media = MediaInMemoryUpload(json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8"), mimetype="application/json")
    fid = find_file(drive, filename, folder_id)
    if fid:
        drive.files().update(fileId=fid, media_body=media).execute()
    else:
        meta = {"name": filename, "parents": [folder_id]}
        drive.files().create(body=meta, media_body=media, fields="id").execute()


def generate_audio(caption, lyrics, bpm, key_scale, vocal_language, duration=190):
    api_key = os.environ["ACE_API_KEY"]
    content = f"<prompt>{caption}</prompt>\n<lyrics>{lyrics}</lyrics>"
    payload = {
        "model": ACE_MODEL,
        "messages": [{"role": "user", "content": content}],
        "bpm": bpm, "duration": duration, "key_scale": key_scale,
        "time_signature": "4", "vocal_language": vocal_language,
        "temperature": 0.85, "top_p": 0.9, "instrumental": False,
    }
    for attempt in range(ACE_MAX_RETRIES):
        try:
            result = _ace_call(payload, api_key)
            if result:
                dur = _ffprobe_duration(result)
                if dur < MIN_AUDIO_DURATION:
                    logger.warning(f"ACE tentativo {attempt + 1}: durata {dur:.0f}s troppo corta")
                    if attempt < ACE_MAX_RETRIES - 1:
                        time.sleep(3)
                    continue
                return result
        except Exception as e:
            logger.warning(f"ACE tentativo {attempt + 1}: {e}")
            if attempt < ACE_MAX_RETRIES - 1:
                time.sleep(2 ** attempt * 5)
    return None


def _ace_call(payload, api_key):
    data = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "User-Agent": "curl/8.18.0",
        "Accept": "*/*",
    }
    req = urllib.request.Request(ACE_ENDPOINT, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=600) as resp:
        body = json.loads(resp.read())
    try:
        audio_url = body["choices"][0]["message"]["audio"][0]["audio_url"]["url"]
    except (KeyError, IndexError) as e:
        logger.error(f"ACE: risposta inattesa: {e}")
        return None
    b64 = audio_url.split(",", 1)[1] if "," in audio_url else audio_url
    return base64.b64decode(b64)


def _ffprobe_duration(mp3_bytes):
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(mp3_bytes)
            tmp = f.name
        r = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", tmp],
            capture_output=True, text=True, timeout=15,
        )
        Path(tmp).unlink(missing_ok=True)
        if r.returncode == 0 and r.stdout.strip():
            return float(r.stdout.strip())
    except Exception:
        pass
    return 0.0


def get_youtube_service():
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build
    creds = Credentials(
        token=None,
        refresh_token=os.environ["YT_REFRESH_TOKEN"],
        client_id=os.environ["YT_CLIENT_ID"],
        client_secret=os.environ["YT_CLIENT_SECRET"],
        token_uri="https://oauth2.googleapis.com/token",
        scopes=["https://www.googleapis.com/auth/youtube.upload", "https://www.googleapis.com/auth/youtube"],
    )
    return build("youtube", "v3", credentials=creds)


def get_or_create_playlist(yt, title, description=""):
    res = yt.playlists().list(part="snippet", mine=True, maxResults=50).execute()
    for item in res.get("items", []):
        if item["snippet"]["title"] == title:
            return item["id"]
    pl = yt.playlists().insert(
        part="snippet,status",
        body={"snippet": {"title": title, "description": description}, "status": {"privacyStatus": "public"}},
    ).execute()
    return pl["id"]


def count_playlist_items(yt, playlist_id):
    res = yt.playlists().list(part="contentDetails", id=playlist_id).execute()
    items = res.get("items", [])
    return items[0]["contentDetails"]["itemCount"] if items else 0


def upload_video(yt, video_path, title, description, tags, playlist_id=None, privacy="public"):
    from googleapiclient.http import MediaFileUpload
    body = {
        "snippet": {"title": title[:100], "description": description, "tags": tags[:500], "categoryId": "10"},
        "status": {"privacyStatus": privacy, "selfDeclaredMadeForKids": False},
    }
    media = MediaFileUpload(video_path, mimetype="video/mp4", resumable=True, chunksize=10 * 1024 * 1024)
    request = yt.videos().insert(part=",".join(body.keys()), body=body, media_body=media)
    response = None
    while response is None:
        _, response = request.next_chunk()
    video_id = response["id"]
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    if playlist_id:
        try:
            yt.playlistItems().insert(
                part="snippet",
                body={"snippet": {"playlistId": playlist_id, "resourceId": {"kind": "youtube#video", "videoId": video_id}}},
            ).execute()
        except Exception as e:
            logger.warning(f"aggiunta playlist fallita: {e}")
    return {"youtube_id": video_id, "youtube_url": video_url}


def build_tags(genre, mood, bpm, key_scale, title, existing_tags):
    base = [genre, mood, f"{genre} music", "original music", "Majesty Music", f"{bpm} bpm", key_scale]
    if genre == "k-pop":
        base += ["케이팝", "K-POP", "Korean pop"]
    combined = base + (existing_tags or [])
    seen, result = set(), []
    for t in combined:
        norm = t.lower().strip()
        if norm not in seen and t.strip():
            seen.add(norm)
            result.append(t.strip())
    return result[:500]


def build_title(genre, title, suffix=None):
    base = f"{title} - {genre.title()}"
    base += f" | {suffix}" if suffix else " | Majesty Music"
    return base[:100]


def build_description(title, genre, mood, bpm, key_scale, base_description, tags):
    import re

    def clean_tag(tag):
        return re.sub(r"[^a-zA-Z0-9가-힣]", "", tag.replace(" ", ""))

    hashtags = " ".join(f"#{clean_tag(t)}" for t in tags[:15])
    return (
        f"{base_description}\n\n"
        f"Genre: {genre.title()} | Mood: {mood.title()} | BPM: {bpm} | Key: {key_scale}\n\n"
        f"{hashtags}"
    )[:5000]


def archive_song(drive, root_folder_id, meta, audio_path, cover_path, video_path):
    genre_folder = get_or_create_folder(drive, meta["genre"], root_folder_id)
    date_folder = get_or_create_folder(drive, str(date.today()), genre_folder)
    results = {}
    for key, path, mime in [
        ("audio_id", audio_path, "audio/mpeg"),
        ("cover_id", cover_path, "image/jpeg"),
        ("video_id", video_path, "video/mp4"),
    ]:
        if path and Path(path).exists():
            results[key] = upload_file(drive, path, date_folder, mime)
    meta_json = {
        "title": meta.get("title"), "genre": meta.get("genre"), "mood": meta.get("mood"),
        "bpm": meta.get("bpm"), "key_scale": meta.get("key_scale"), "vocal_language": meta.get("vocal_language"),
        "description": meta.get("description"), "tags": meta.get("tags"), "lyrics": meta.get("lyrics"),
        "youtube_url": meta.get("youtube_url"),
    }
    tmp = Path("_meta_tmp.json")
    tmp.write_text(json.dumps(meta_json, ensure_ascii=False, indent=2), encoding="utf-8")
    results["meta_id"] = upload_file(drive, str(tmp), date_folder, "application/json", remote_name="metadata.json")
    tmp.unlink(missing_ok=True)
    return results


def yt_quota_exhausted(drive, queue_folder_id):
    fid = find_file(drive, "yt_quota.json", queue_folder_id)
    if not fid:
        return False
    data = read_json_file(drive, fid)
    reset_after = datetime.fromisoformat(data["reset_after"])
    return datetime.now(timezone.utc) < reset_after


def mark_yt_quota_exhausted(drive, queue_folder_id):
    reset_after = datetime.now(timezone.utc) + timedelta(hours=24)
    write_json(drive, {"reset_after": reset_after.isoformat()}, "yt_quota.json", queue_folder_id)



# ─────────────────────────────────────────────────────────────────────────────
# GENERAZIONE TESTO (replicato da agents/text_agent.py)
# ─────────────────────────────────────────────────────────────────────────────

TEXT_MOODS = {
    "electro-swing":  ["upbeat", "playful", "jazzy", "nostalgic", "energetic"],
    "rock":           ["powerful", "intense", "rebellious", "melancholic", "anthemic"],
    "pop":            ["happy", "romantic", "danceable", "emotional", "catchy"],
    "k-pop":          ["cute", "fierce", "dreamy", "powerful", "trendy"],
    "lofi-chillout":  ["relaxed", "nostalgic", "melancholic", "cozy", "peaceful"],
}

TEXT_IMAGE_BASE = {
    "electro-swing": (
        "warm amber and sepia tones, art deco aesthetic, cinematic grain, "
        "soft vignette, vintage film photography, high contrast chiaroscuro"
    ),
    "rock": (
        "high contrast, dramatic harsh shadows, desaturated palette with strong accent colors, "
        "gritty raw texture, dark cinematic atmosphere, moody tonal range"
    ),
    "pop": (
        "vivid saturated colors, bright dynamic lighting, clean commercial aesthetic, "
        "sharp hyperrealistic photography, modern glossy finish"
    ),
    "k-pop": (
        "ethereal soft lighting, pastel and neon color palette, ultra-clean finish, "
        "dreamy cinematic atmosphere, hyperrealistic, elegant Korean commercial aesthetic"
    ),
    "lofi-chillout": (
        "muted warm tones, soft analog film grain, low-key intimate lighting, "
        "vintage texture, hazy atmospheric depth, cozy lo-fi aesthetic"
    ),
}

ELECTRO_SWING_VARIANTS = [
    (
        "Genre: Sensual Electro Swing / Club Neo-Swing. "
        "Instrumentation: Deep pulsing synth bass fused with jazz double bass, "
        "four-on-the-floor electronic kick drum, crisp sampled finger snaps on the beat, "
        "sensual gritty brass section with syncopated riffs, club piano chords, modern synth accents. "
        "Rhythm: Super danceable 130-135 BPM, seductive glamorous luxurious atmosphere, "
        "exclusive club lit by warm lights where vintage meets house music. "
        "Structure: Theatrical elegant spoken intro, whispered verses building tension, "
        "explosive high-energy drop choruses with vocal choirs and brass riffs, "
        "brief rhythmic pauses with only finger snaps and kick before the final drop."
    ),
    (
        "Genre: Cinematic Electro Swing / Spy-Jazz / Neo-Swing. "
        "Instrumentation: Bold cinematic brass section (trumpets and trombones) with James Bond-style riffs, "
        "walking syncopated double bass, modern drum machine with sharp claps and open hi-hats, "
        "dramatic orchestral string inserts for tension. "
        "Rhythm: Energetic bouncy 132 BPM, ironic mysterious adventurous atmosphere, "
        "1960s spy soundtrack meets modern wild club track. "
        "Structure: Theatrical spoken intro with radio noise and comic quotes, tight groove verses, "
        "explosive brass-led choruses, alternating fake-suspense moments and overwhelming dance drops, "
        "chaotic exuberant finale."
    ),
    (
        "Genre: Modern Electro Swing / High-Energy Neo-Swing. "
        "Instrumentation: Blaring aggressive brass led by solo trumpets and trombones, "
        "powerful four-on-the-floor electronic kick drum, rhythmic bouncy slap bass, "
        "syncopated jazz piano accents, swing guitar hints. "
        "Rhythm: Fast overwhelming danceable 132 BPM, "
        "chaotic smoky vintage night party with modern energetic production. "
        "Structure: Alternating brass-led instrumental sections and tight groove verses, "
        "powerful drops where kick meets trumpet riffs, "
        "energetic raspy charismatic vocals in modern crooner style."
    ),
    (
        "Genre: Fast Electro Swing / Quirky Swing-House. "
        "Instrumentation: Frantic rhythm section with driving electronic kick and extremely bouncy slap double bass, "
        "brass (trumpets and saxophones) with staccato punchy rhythmic riffs "
        "alternating with vocal samples and playful sound effects, "
        "fast honky-tonk piano and tight hi-hat. "
        "Rhythm: Very fast syncopated 130-135 BPM, ironic sarcastic theatrical quirky atmosphere, "
        "dynamic dance challenge full of charisma and geometric moves. "
        "Structure: Theatrical spoken voice intro, tight groove verses with one-two step rhythm, "
        "explosive instrumental drops with sharp brass and ultra-high-energy club atmosphere."
    ),
    (
        "Genre: Dark Electro Swing / Vintage Gangster Jazz / Speakeasy Swing. "
        "Instrumentation: Dark rich brass section with muted trumpets, smoky saxophones and deep trombones "
        "performing mysterious Prohibition-era 1930s melodies, "
        "deep groovy bass line (synth or double bass), vintage jazz piano samples, "
        "modern mid-tempo electronic beat with straight kick and clap. "
        "Rhythm: Fast driving tempo, smoky shady cinematic suspenseful atmosphere yet strongly danceable, "
        "underground illegal speakeasy with gangster stories, mystery and retro elegance. "
        "Structure: Atmospheric intro with vintage brass riffs, narrative verses building tension, "
        "massive explosive drop choruses driven by a wall of tight brass."
    ),
    (
        "Genre: High-Octane Electro Swing / Vintage Club Dance. "
        "Instrumentation: Overwhelming unhinged brass section with extremely high trumpets and roaring trombones, "
        "very deep hammering synth bass line, rhythmic syncopated jazz piano, "
        "modern energetic dance drum kit with heavily marked open hi-hats pushing the groove. "
        "Rhythm: Fast wild tempo, shameless ironic chaotic euphoric atmosphere, "
        "sweaty dance night in a retro venue where classic elegance collides "
        "with raw energy and rebellious attitude of contemporary electronic music. "
        "Structure: Theatrical ironic spoken opening, driving verses with tight groove, "
        "explosive choruses with brass solos and riffs pushed to the limit."
    ),
]

KPOP_VARIANTS = [
    (
        "Viral K-Pop Dance Challenge track, 125 BPM, high-energy, infectious, extremely catchy. "
        "Modern slap bassline, aggressive electronic claps, punchy kick drum, and a quirky, memorable "
        "synth whistle riff. Minimalist but driving structure. Sassy female vocals with an easy, "
        "repetitive hook (\"chant\") designed for a viral dance routine. Sudden explosive beat drop."
    ),
    (
        "2014 Summer K-Pop, Sistar style, 120 BPM, extremely catchy, bright, sunny, flirtatious "
        "feel-good vibe. Driving hip-hop inspired drum beat, funky sub-bassline, bouncy synth plucks, "
        "and a signature, highly addictive brass saxophone hook that repeats throughout the song. "
        "Upbeat and infectious pool party atmosphere. Sassy and confident female vocals, combining a "
        "rhythmic and playful \"talking\" flow in the verses with powerful, soulful belting and bubbly "
        "group chants in a massive, melodic chorus."
    ),
    (
        "K-Pop EDM Anthem, PSY style, 132 BPM, explosive, high-energy, comedic, highly addictive, "
        "viral party vibe. Massive pulsating Electro-House synthesizer lead riff, heavy four-on-the-floor "
        "kick drum, rolling sub-bass, and sharp electronic claps. Rowdy, anthemic festival atmosphere. "
        "Energetic, charismatic male vocals, combining a fast rhythmic rap-talking flow in the verses "
        "with loud, shouted vocal chants (\"hey!\", \"go!\") leading into a massive, booming "
        "stadium-status instrumental drop."
    ),
    (
        "Powerful K-Pop Girl Crush, Jennie style, 105 BPM, dark, heavy, hypnotic, absolute swagger, "
        "fashion runway attitude. Deep pulsing sub-bass 808, crisp modern hip-hop drum kit with heavy "
        "claps, minimalist but sharp synth stabs, and subtle futuristic glitch effects. Sassy, confident "
        "female vocals switching between a fast, sharp rhythmic rap flow in the verses and an ultra-catchy, "
        "repetitive, and chant-like melodic chorus. High-energy, bold, and luxurious underground club atmosphere."
    ),
    (
        "Bright K-Pop Boyhood Pop, TWS style, 130 BPM, up-tempo, youthful, fresh, highly energetic, "
        "innocent feel-good vibe. Driving synth-pop bassline, crisp acoustic pop drums, bright piano chords, "
        "and shimmering synthesizer plucks. Uplifting and nostalgic high school anime opening atmosphere. "
        "Clean, youthful, and sweet male vocals with clear melodic verses, group chants "
        "(\"one, two, three, go!\"), and highly layered, airy harmonies in a triumphant, soaring chorus."
    ),
]

POP_VARIANTS = [
    (
        "Dramatic Synth-Pop, Taylor Swift style, 112 BPM, mid-tempo, cinematic, elegant, melancholic "
        "yet powerful storytelling vibe. Pulsing electronic synth bassline, crisp modern pop percussion, "
        "combined with rich, sweeping orchestral string arrangements and subtle piano chords. Sophisticated "
        "and theatrical atmosphere. Expressive, smooth female vocals, starting with intimate, clear verses "
        "that build tension, exploding into a grand, layered, and deeply emotional melodic chorus with "
        "soaring vocal harmonies."
    ),
    (
        "Upbeat Indie-Pop, modern Bedroom Pop, 105 BPM, bright, breezy, sunny, highly infectious "
        "feel-good vibe. Groovy acoustic guitar strumming, plucky electric guitar riffs, warm melodic "
        "bassline, and a crisp, punchy pop drum kit with handclaps. Playful and romantic late-summer "
        "atmosphere. Casual, charismatic male vocals with rhythmic, fast-paced verses that transition "
        "smoothly into a highly repetitive, sweet, and ultra-catchy melodic chorus with layered vocal harmonies."
    ),
    (
        "Melodic EDM-Pop, 103 BPM, mid-tempo, emotional, uplifting, anthemic, bittersweet festival vibe. "
        "Gentle organic piano chords and soft ambient synth pads in the verses, building tension with "
        "automated filter sweeps and sharp electronic claps. The track explodes into a bright, pulsing "
        "electronic synthesizer lead drop with a warm, driving bassline and a heavy four-on-the-floor kick "
        "drum. Nostalgic and cinematic atmosphere. Clean, expressive, and melancholic male vocals with clear "
        "storytelling lyrics, rising into a soaring, layered vocal harmony just before a massive, "
        "melodic instrumental drop."
    ),
    (
        "Modern Tropical House, Synth-Pop, 116 BPM, mid-tempo, empowering, moody yet danceable, cool "
        "confident vibe. Minimalist and deep sub-bassline, steady electronic pop drum beat, layered with a "
        "prominent plucked marimba-style synth riff and sharp digital claps that build tension. Sleek, "
        "commanding, and smooth female vocals, delivering rhythmic storytelling in the verses, rising through "
        "a layered pre-chorus, and dropping into a highly repetitive, hypnotic, and infectious vocal-chopped "
        "synth chorus. Sophisticated, tropical-pop club atmosphere."
    ),
    (
        "Nu-Disco, Retro Synth-Pop, 120 BPM, mid-tempo, bright, breezy, flirtatious, effortless feel-good "
        "vibe. Driving and infectious slap funk bassline, rhythmic muted electric guitar strums, crisp "
        "electro-pop drum kit with a steady four-on-the-floor kick, and soft shimmering synthesizer chords "
        "in the background. Sassy, confident, and smooth female vocals, combining playful conversational "
        "rhythmic verses with a highly repetitive, hypnotic, and catchy melodic chorus. Sunny, chic, and "
        "breezy summer pool party atmosphere."
    ),
    (
        "High-octane EDM Festival Anthem, Electropop, 128 BPM, aggressive, commanding, powerful, "
        "high-intensity workout vibe. Loud heavy four-on-the-floor electronic kick drum, sharp grinding "
        "sawtooth synthesizer riffs, and massive digital filter sweeps that build intense cinematic tension "
        "before exploding. Fierce, confident female vocals delivering spoken-word rhythmic commands, list-like "
        "lyrics, and sharp repetitive punchlines. Pounding club music production with frantic percussion "
        "build-ups, lasers, and a giant, earth-shaking electronic bass drop that demands movement."
    ),
]

TEXT_BPM_RANGE = {
    "electro-swing":  (130, 140),
    "rock":           (100, 160),
    "pop":            (90,  130),
    "k-pop":          (90,  135),
    "lofi-chillout":  (65,   90),
}


def generate_text(genre: str, vocal_language: str) -> dict | None:
    """Genera metadati canzone via Gemini. Replicato da agents/text_agent.py."""
    import random, re, json as _json
    try:
        from google import genai as _genai
    except ImportError:
        logger.error("google-genai non installato")
        return None

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        logger.error("GEMINI_API_KEY mancante")
        return None
    client = _genai.Client(api_key=api_key)

    avoid_ctx = (
        '\n\nDo NOT use the word "velvet" (or "velvety"/"velveteen") anywhere in the '
        "title, lyrics, caption, or description.\n"
    )
    moods_list  = ", ".join(TEXT_MOODS.get(genre, ["upbeat"]))
    image_base  = TEXT_IMAGE_BASE.get(genre, "")
    is_new      = not image_base

    if is_new:
        image_style_field = (
            '  "image_style": "Visual aesthetic for this genre: color palette, lighting style, '
            'rendering technique, atmosphere. 60-90 chars. NO subjects, NO scenes.",\n'
        )
        image_prompt_instr = (
            "For image_prompt: cinematic album cover scene inspired by title, mood, and lyrics. "
            "Use the image_style you generated as visual guide. 100-150 chars. NO text, NO logos."
        )
    else:
        image_style_field = ""
        image_prompt_instr = (
            f"For image_prompt: cinematic album cover scene inspired by title, mood, and lyrics. "
            f"Visual style: {image_base}. 100-150 chars. NO text, NO logos."
        )

    _preset = None
    if genre == "electro-swing":
        _preset = random.choice(ELECTRO_SWING_VARIANTS)
        genre_hint = (
            f"\nProduction style (use it for title/lyrics/mood — do NOT put it in caption, "
            f"just write 'see preset'):\n{_preset}\n"
        )
    elif genre == "k-pop":
        _preset = random.choice(KPOP_VARIANTS)
        genre_hint = (
            f"\nProduction style (use it for title/lyrics/mood — do NOT put it in caption, "
            f"just write 'see preset'):\n{_preset}\n"
            f"\nIMPORTANT for lyrics: dense syllable-packed lines. "
            f"Mandatory structure: [Verse] 4 lines, [Pre-Chorus] 3 lines, [Chorus] 4 lines, "
            f"[Verse 2] 3 lines, [Bridge] 2 lines, [Outro] 2 lines.\n"
            f"\nIMPORTANT for title: bilingual — Korean script first, English in parentheses "
            f"(e.g. \"한여름의 꿈 (Midsummer Dream)\"). Never English-only.\n"
        )
    elif genre == "pop":
        _preset = random.choice(POP_VARIANTS)
        genre_hint = (
            f"\nProduction style (use it for title/lyrics/mood — do NOT put it in caption, "
            f"just write 'see preset'):\n{_preset}\n"
        )
    else:
        genre_hint = ""

    bpm_min, bpm_max = TEXT_BPM_RANGE.get(genre, (80, 160))

    prompt = (
        f"Create an original {genre} song. Respond ONLY with valid JSON, no markdown.\n\n"
        f"Requirements:\n"
        f"- Vocal language: {vocal_language}\n"
        f"- Genre: {genre}\n"
        f"- Available moods: {moods_list}\n"
        f"- BPM: choose between {bpm_min} and {bpm_max}\n"
        f"{genre_hint}{avoid_ctx}\n"
        f"{image_prompt_instr}\n\n"
        f'JSON structure:\n{{\n'
        f'  "title": "Song title (max 80 chars)",\n'
        f'  "caption": "Music production style for AI generation (300-500 chars)",\n'
        f'  "lyrics": "[Verse]\\nLine 1\\nLine 2\\n\\n[Chorus]\\nLine 1\\nLine 2\\n\\n'
        f'[Verse 2]\\nLine 1\\nLine 2\\n\\n[Bridge]\\nLine 1\\n\\n[Outro]\\nLine 1",\n'
        f'  "bpm": 120,\n'
        f'  "key_scale": "C Major",\n'
        f'  "vocal_language": "{vocal_language}",\n'
        f'  "mood": "one mood from the list",\n'
        f'  "description": "YouTube SEO description 150-300 chars, no hashtags",\n'
        f'  "tags": ["tag1","tag2","tag3","tag4","tag5","tag6","tag7","tag8"],\n'
        f'{image_style_field}'
        f'  "image_prompt": "cinematic album cover scene"\n'
        f'}}'
    )

    for attempt in range(5):
        try:
            resp  = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
            raw   = resp.text.strip()
            clean = re.sub(r"^```(?:json)?\n?", "", raw)
            clean = re.sub(r"\n?```$", "", clean).strip()
            data  = _json.loads(clean)
            if _preset:
                data["caption"] = _preset
            return data
        except _json.JSONDecodeError as e:
            logger.warning(f"generate_text [{genre}] JSON non valido (tentativo {attempt+1}): {e}")
        except Exception as e:
            err = str(e)
            if "429" in err or "quota" in err.lower():
                wait = 2 ** attempt * 5
                logger.warning(f"generate_text [{genre}] quota, attendo {wait}s")
                time.sleep(wait)
            elif "503" in err or "unavailable" in err.lower():
                logger.warning(f"generate_text [{genre}] 503, retry in 10s")
                time.sleep(10)
            else:
                logger.error(f"generate_text [{genre}]: {e}")
                break
    logger.error(f"generate_text [{genre}]: fallito dopo 5 tentativi")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# GENERAZIONE IMMAGINE (replicato da agents/image_agent.py)
# ─────────────────────────────────────────────────────────────────────────────

IMG_GENRE_STYLES = {
    "electro-swing": (
        "1930s jazz club interior, art deco architecture, warm amber stage lighting, "
        "brass instruments on stage, elegant dancers in vintage attire, atmospheric smoke, "
        "cinematic film photography, hyperrealistic, no visible faces"
    ),
    "rock": (
        "dark arena concert stage, massive crowd silhouettes, dramatic colored spotlights "
        "cutting through fog, electric guitar silhouette backlit, raw powerful energy, "
        "cinematic rock concert photography, moody atmosphere"
    ),
    "pop": (
        "glamorous rooftop party at golden hour, city skyline glowing at dusk, "
        "colorful confetti falling, vibrant neon signs reflecting on glass, "
        "cinematic concert photography, euphoric atmosphere, ultra hyperrealistic"
    ),
    "k-pop": (
        "Seoul cityscape at night, neon reflections on wet pavement, cherry blossom trees "
        "with pink LED lights, futuristic Korean street aesthetic, ethereal purple and pink "
        "atmosphere, cinematic hyperrealistic photography"
    ),
    "lofi-chillout": (
        "cozy bedroom desk at rainy window, warm lamp glow, vinyl record player, "
        "stack of books and coffee mug, soft hazy bokeh, muted warm tones, "
        "analog film grain, intimate lo-fi atmosphere, hyperrealistic"
    ),
}

IMG_CF_ENDPOINT = (
    "https://api.cloudflare.com/client/v4/accounts/{account_id}"
    "/ai/run/@cf/black-forest-labs/flux-1-schnell"
)
IMG_MS_ENDPOINT      = "https://api-inference.modelscope.cn/v1/images/generations"
IMG_MS_TASK_ENDPOINT = "https://api-inference.modelscope.cn/v1/tasks/{task_id}"
IMG_MS_MODELS        = ["Tongyi-MAI/Z-Image-Turbo", "Qwen/Qwen-Image-2512"]
IMG_HF_SPACES = [
    {
        "space_id": "AP123/SDXL-Lightning",
        "api_name": "/generate_image",
        "args": lambda p: [p, "4-Step"],
    },
    {
        "space_id": "stabilityai/stable-diffusion-3-medium",
        "api_name": "/infer",
        "args": lambda p: [p, "text watermark logo letters", 0, 7.0, 1344, 768, True],
    },
]


def generate_image(genre: str, mood: str, title: str, image_prompt: str = None) -> bytes | None:
    """Genera copertina album. Replicato da agents/image_agent.py."""
    prompt_text = _img_build_prompt(genre, mood, image_prompt)
    ms_token = os.environ.get("MODELSCOPE_TOKEN", "")
    if ms_token:
        for model_id in IMG_MS_MODELS:
            img = _img_from_modelscope(model_id, prompt_text, ms_token)
            if img:
                return img
        logger.warning("ModelScope fallito — provo Cloudflare FLUX")
    else:
        logger.warning("MODELSCOPE_TOKEN mancante — salto ModelScope")

    img = _img_from_cloudflare(prompt_text)
    if img:
        return img

    logger.warning("Cloudflare fallito — provo HF Spaces")
    for space_cfg in IMG_HF_SPACES:
        img = _img_from_hf_space(
            space_cfg["space_id"], space_cfg["api_name"], space_cfg["args"](prompt_text)
        )
        if img:
            return img

    logger.error("Tutti i provider immagine falliti")
    return None


def _img_build_prompt(genre: str, mood: str, image_prompt: str = None) -> str:
    base = image_prompt or IMG_GENRE_STYLES.get(genre, "cinematic abstract music artwork, vibrant colors")
    return (
        f"{base}, {mood} mood, "
        "ultra high quality, cinematic widescreen 16:9, "
        "NO text, NO letters, NO watermarks, NO logos"
    )


def _img_from_modelscope(model_id: str, prompt_text: str, token: str) -> bytes | None:
    import json as _json, urllib.request as _ur, urllib.error as _ue
    short   = model_id.split("/")[-1]
    payload = _json.dumps({"model": model_id, "prompt": prompt_text, "n": 1, "size": "1344x768"}).encode()
    req = _ur.Request(IMG_MS_ENDPOINT, data=payload,
                      headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                      method="POST")
    try:
        with _ur.urlopen(req, timeout=120) as resp:
            body = _json.loads(resp.read())
        items = body.get("data") or body.get("images") or []
        if items:
            return _img_extract_ms(items[0], short)
        task_id = body.get("task_id") or body.get("id")
        if task_id:
            return _img_poll_ms(task_id, token, short)
        logger.warning(f"ModelScope {short}: risposta inattesa")
        return None
    except _ue.HTTPError as e:
        logger.warning(f"ModelScope {short} HTTP {e.code}")
        return None
    except Exception as e:
        logger.warning(f"ModelScope {short}: {e}")
        return None


def _img_poll_ms(task_id: str, token: str, label: str) -> bytes | None:
    import json as _json, urllib.request as _ur
    url = IMG_MS_TASK_ENDPOINT.format(task_id=task_id)
    deadline, interval = time.time() + 120, 3
    while time.time() < deadline:
        time.sleep(interval)
        interval = min(interval * 1.5, 15)
        try:
            req = _ur.Request(url, headers={"Authorization": f"Bearer {token}"})
            with _ur.urlopen(req, timeout=30) as resp:
                body = _json.loads(resp.read())
        except Exception as e:
            logger.warning(f"ModelScope poll {label}: {e}")
            continue
        status = body.get("status", "")
        if status in ("succeeded", "completed"):
            items = body.get("output", {}).get("data") or body.get("data") or []
            return _img_extract_ms(items[0], label) if items else None
        if status in ("failed", "error"):
            logger.warning(f"ModelScope {label}: task fallito")
            return None
    logger.warning(f"ModelScope {label}: timeout")
    return None


def _img_extract_ms(item: dict, label: str) -> bytes | None:
    import base64 as _b64
    if "b64_json" in item:
        try:
            data = _b64.b64decode(item["b64_json"])
            if len(data) > 10_000:
                logger.info(f"ModelScope {label}: OK {len(data)//1024}KB")
                return data
        except Exception as e:
            logger.warning(f"ModelScope {label} b64: {e}")
        return None
    if "url" in item:
        return _img_download_url(item["url"], label)
    return None


def _img_from_cloudflare(prompt_text: str) -> bytes | None:
    import json as _json, base64 as _b64, urllib.request as _ur, urllib.error as _ue
    token      = os.environ.get("CLOUDFLARE_TOKEN", "")
    account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "")
    if not token or not account_id:
        logger.warning("Cloudflare: credenziali mancanti")
        return None
    url     = IMG_CF_ENDPOINT.format(account_id=account_id)
    payload = _json.dumps({"prompt": prompt_text, "width": 1344, "height": 768, "num_steps": 8}).encode()
    req = _ur.Request(url, data=payload,
                      headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                      method="POST")
    try:
        with _ur.urlopen(req, timeout=120) as resp:
            ct   = resp.headers.get("Content-Type", "")
            data = resp.read()
        if "image/" in ct and len(data) > 10_000:
            logger.info(f"Cloudflare FLUX: OK {len(data)//1024}KB")
            return data
        try:
            body = _json.loads(data)
            img  = _b64.b64decode(body["result"]["image"])
            if len(img) > 10_000:
                return img
        except Exception:
            pass
        logger.warning("Cloudflare: risposta non valida")
        return None
    except _ue.HTTPError as e:
        logger.warning(f"Cloudflare HTTP {e.code}")
        return None
    except Exception as e:
        logger.warning(f"Cloudflare: {e}")
        return None


def _img_from_hf_space(space_id: str, api_name: str, args: list) -> bytes | None:
    try:
        from gradio_client import Client
    except ImportError:
        logger.warning("gradio_client non installato")
        return None
    try:
        c      = Client(space_id, verbose=False)
        result = c.predict(*args, api_name=api_name)
    except Exception as e:
        logger.warning(f"HF Space {space_id}: {e}")
        return None
    if isinstance(result, (list, tuple)):
        result = result[0]
    if isinstance(result, str) and os.path.isfile(result):
        try:
            data = open(result, "rb").read()
            os.remove(result)
            if len(data) > 10_000:
                logger.info(f"HF Space {space_id}: OK {len(data)//1024}KB")
                return data
        except Exception as e:
            logger.warning(f"HF Space {space_id}: {e}")
    return None


def _img_download_url(url: str, label: str) -> bytes | None:
    import urllib.request as _ur
    try:
        with _ur.urlopen(url, timeout=60) as r:
            data = r.read()
        if len(data) > 10_000:
            return data
    except Exception as e:
        logger.warning(f"{label} download URL: {e}")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# VALIDAZIONE (replicato da core/validator.py)
# ─────────────────────────────────────────────────────────────────────────────

def validate_text(data: dict) -> tuple:
    if not isinstance(data, dict):
        return False, "non un dict"
    for f in ("title", "caption", "lyrics", "bpm", "key_scale", "vocal_language",
              "description", "tags", "mood"):
        if not data.get(f):
            return False, f"campo mancante: {f}"
    bpm = data.get("bpm")
    if not isinstance(bpm, int) or not (60 <= bpm <= 200):
        return False, f"bpm non valido: {bpm}"
    if not any(s in data.get("lyrics", "") for s in ("[Verse]", "[Chorus]")):
        return False, "lyrics senza [Verse]/[Chorus]"
    if not isinstance(data.get("tags"), list) or len(data["tags"]) < 3:
        return False, "tags insufficienti"
    return True, ""


def validate_image(img_bytes: bytes) -> tuple:
    if not img_bytes or len(img_bytes) < 1024:
        return False, "immagine vuota o troppo piccola"
    if not (img_bytes[:2] == b"\xff\xd8" or img_bytes[:4] == b"\x89PNG"):
        return False, "header immagine non valido"
    return True, ""


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE FULL (testo + immagine + audio + video + upload, tutto cloud)
# ─────────────────────────────────────────────────────────────────────────────

def full_pipeline_run(drive, yt, root_folder_id: str, count: int, genre_names: list | None = None, dry_run: bool = False):
    genres = [g for g in GENRES if not genre_names or g["name"] in genre_names]
    start  = time.time()
    done, failed = 0, 0
    for genre_cfg in genres:
        for _ in range(count):
            if time.time() - start > TIME_BUDGET_S:
                logger.info("budget tempo esaurito")
                return done, failed
            genre = genre_cfg["name"]
            try:
                logger.info(f"{genre}: genera testo")
                meta = generate_text(genre, genre_cfg["vocal_language"])
                if not meta:
                    failed += 1; continue
                ok, reason = validate_text(meta)
                if not ok:
                    logger.error(f"{genre}: testo non valido ({reason})")
                    failed += 1; continue

                logger.info(f"{genre}: genera immagine — {meta['title']}")
                img_bytes = generate_image(genre, meta.get("mood", ""), meta["title"],
                                           meta.get("image_prompt"))
                if not img_bytes:
                    failed += 1; continue
                ok, reason = validate_image(img_bytes)
                if not ok:
                    logger.error(f"{genre}: immagine non valida ({reason})")
                    failed += 1; continue

                meta["genre"]          = genre
                meta["vocal_language"] = genre_cfg["vocal_language"]
                item_id = f"cloud_{int(time.time())}_{genre}"
                process_item(drive, yt, root_folder_id, None, item_id, meta,
                             cover_bytes=img_bytes, dry_run=dry_run)
                done += 1
            except Exception as e:
                logger.error(f"{genre}: {e}")
                failed += 1
    return done, failed

def process_item(drive, yt, root_folder_id, processing_folder_id, item_id, meta, cover_bytes=None, dry_run=False):
    genre = meta["genre"]
    workdir = Path(f"work_{item_id}")
    workdir.mkdir(exist_ok=True)
    try:
        logger.info(f"{item_id}: audio")
        audio_bytes = generate_audio(
            caption=meta["caption"], lyrics=meta["lyrics"], bpm=meta["bpm"],
            key_scale=meta["key_scale"], vocal_language=meta["vocal_language"],
        )
        if not audio_bytes:
            raise RuntimeError("ACE Music fallito")
        audio_path = workdir / "audio.mp3"
        audio_path.write_bytes(audio_bytes)

        cover_path = workdir / "cover.jpg"
        if cover_bytes:
            cover_path.write_bytes(cover_bytes)
        else:
            cover_file_id = find_file(drive, f"{item_id}.jpg", processing_folder_id)
            if not cover_file_id:
                raise RuntimeError("copertina mancante in coda")
            download_file(drive, cover_file_id, str(cover_path))

        logger.info(f"{item_id}: render")
        video_path = workdir / "output.mp4"
        r = subprocess.run(
            [sys.executable, "render.py", "--audio", str(audio_path), "--cover", str(cover_path),
             "--title", meta["title"], "--genre", genre, "--output", str(video_path)],
            capture_output=True, text=True,
        )
        if r.returncode != 0 or not video_path.exists():
            raise RuntimeError(f"render.py fallito: {r.stderr[-2000:]}")

        logger.info(f"{item_id}: upload")
        tags = build_tags(genre, meta.get("mood", ""), meta.get("bpm", 0), meta.get("key_scale", ""), meta["title"], meta.get("tags", []))
        yt_title = build_title(genre, meta["title"])
        yt_desc = build_description(meta["title"], genre, meta.get("mood", ""), meta.get("bpm", 0), meta.get("key_scale", ""), meta.get("description", ""), tags)

        playlist_name = f"Majesty Music — {genre.title()}"
        playlist_id = get_or_create_playlist(yt, playlist_name)
        if playlist_id:
            count = count_playlist_items(yt, playlist_id)
            if count >= PLAYLIST_CAP:
                vol = count // PLAYLIST_CAP + 1
                playlist_name = f"Majesty Music — {genre.title()} Vol.{vol}"
                playlist_id = get_or_create_playlist(yt, playlist_name)

        if dry_run:
            logger.info(f"{item_id}: [dry_run] skip upload YouTube")
            return
        result = upload_video(yt, str(video_path), yt_title, yt_desc, tags, playlist_id=playlist_id)
        meta["youtube_url"] = result["youtube_url"]

        logger.info(f"{item_id}: archivio")
        archive_song(drive, root_folder_id, meta, str(audio_path), str(cover_path), str(video_path))
        logger.info(f"{item_id}: ok {result['youtube_url']}")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)



def _redispatch_if_no_stop_flag(root_folder_id: str) -> None:
    """Re-dispatcha pipeline.yml se stop flag assente su Drive."""
    try:
        from google.oauth2.credentials import Credentials
        from googleapiclient.discovery import build
        creds = Credentials(
            token=None,
            refresh_token=os.environ["DRIVE_REFRESH_TOKEN"],
            client_id=os.environ["DRIVE_CLIENT_ID"],
            client_secret=os.environ["DRIVE_CLIENT_SECRET"],
            token_uri="https://oauth2.googleapis.com/token",
            scopes=["https://www.googleapis.com/auth/drive.file"],
        )
        svc = build("drive", "v3", credentials=creds)
        res = svc.files().list(
            q=f"name='_pipeline_stop.json' and '{root_folder_id}' in parents and trashed=false",
            fields="files(id)"
        ).execute()
        if res.get("files"):
            logger.info("Stop flag trovato — fine modalita continua")
            return
        import subprocess as _sp
        _sp.run(
            ["gh", "workflow", "run", "pipeline.yml",
             "--repo", "pilgrimdelamare/task-runner",
             "--field", "mode=full",
             "--field", "continuous=true"],
            check=True, capture_output=True
        )
        logger.info("Re-dispatch pipeline.yml (modalita continua)")
    except Exception as e:
        logger.error(f"_redispatch_if_no_stop_flag: {e}")

def main():
    root_folder_id = os.environ["DRIVE_ROOT_FOLDER_ID"]
    mode       = os.environ.get("PIPELINE_MODE", "worker")
    count      = int(os.environ.get("PIPELINE_COUNT", "1"))
    continuous = os.environ.get("PIPELINE_CONTINUOUS", "false").lower() == "true"
    dry_run    = os.environ.get("PIPELINE_DRY_RUN",    "false").lower() == "true"

    drive = get_drive_service()
    yt    = get_youtube_service()

    if mode == "full":
        logger.info(f"Modalità full: {count} canzoni per genere")
        done, failed = full_pipeline_run(drive, yt, root_folder_id, count, dry_run=dry_run)
        logger.info(f"Pipeline full: {done} ok, {failed} fallite")
        if continuous and not dry_run:
            _redispatch_if_no_stop_flag(root_folder_id)
        return

    # ── Modalità worker (default): processa coda Drive ────────────────────────
    queue_folder = get_or_create_folder(drive, "_queue", root_folder_id)
    pending_folder = get_or_create_folder(drive, "pending", queue_folder)
    processing_folder = get_or_create_folder(drive, "processing", queue_folder)
    failed_folder = get_or_create_folder(drive, "failed", queue_folder)

    if yt_quota_exhausted(drive, queue_folder):
        logger.info("quota YouTube esaurita, non reclamo nulla in questo giro")
        return

    start = time.time()
    while time.time() - start < TIME_BUDGET_S:
        pending_jsons = [f for f in list_children(drive, pending_folder) if f["name"].endswith(".json")]
        if not pending_jsons:
            logger.info("coda vuota")
            break
        pending_jsons.sort(key=lambda f: f["name"])
        item = pending_jsons[0]
        item_id = item["name"][:-5]

        jpg_id = find_file(drive, f"{item_id}.jpg", pending_folder)
        move_file(drive, item["id"], pending_folder, processing_folder)
        if jpg_id:
            move_file(drive, jpg_id, pending_folder, processing_folder)

        meta = read_json_file(drive, item["id"])

        try:
            process_item(drive, yt, root_folder_id, processing_folder, item_id, meta)
            delete_file(drive, item["id"])
            jpg2 = find_file(drive, f"{item_id}.jpg", processing_folder)
            if jpg2:
                delete_file(drive, jpg2)
        except Exception as e:
            err = str(e)
            logger.error(f"{item_id}: err {err}")
            if "quotaExceeded" in err or "uploadLimitExceeded" in err:
                mark_yt_quota_exhausted(drive, queue_folder)
            meta["_error"] = err[:1000]
            write_json(drive, meta, f"{item_id}.json", failed_folder)
            jpg2 = find_file(drive, f"{item_id}.jpg", processing_folder)
            if jpg2:
                move_file(drive, jpg2, processing_folder, failed_folder)
            delete_file(drive, item["id"])
            break


if __name__ == "__main__":
    main()
