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


def process_item(drive, yt, root_folder_id, processing_folder_id, item_id, meta):
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

        result = upload_video(yt, str(video_path), yt_title, yt_desc, tags, playlist_id=playlist_id)
        meta["youtube_url"] = result["youtube_url"]

        logger.info(f"{item_id}: archivio")
        archive_song(drive, root_folder_id, meta, str(audio_path), str(cover_path), str(video_path))
        logger.info(f"{item_id}: ok {result['youtube_url']}")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def main():
    root_folder_id = os.environ["DRIVE_ROOT_FOLDER_ID"]
    drive = get_drive_service()
    yt = get_youtube_service()

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
