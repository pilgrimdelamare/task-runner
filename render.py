#!/usr/bin/env python3
"""
render.py - Standalone video renderer for GitHub Actions.
No external Python dependencies — only stdlib + FFmpeg.
Usage: python render.py --audio audio.mp3 --cover cover.jpg --title "Title" --genre rock --output output.mp4
"""
import argparse
import math
import struct
import subprocess
import sys
from pathlib import Path

LOGO_PATH  = Path(__file__).parent / "assets" / "logo.png"
INTRO_PATH = Path(__file__).parent / "assets" / "intro.mp4"

CHUNK_SIZE      = 30
SAMPLE_RATE     = 8000
HOP_MS          = 20
FLASH_DUR       = 0.20
MIN_GAP_S       = 0.70
BASS_LOWPASS    = 200
LOGO_AREA_RATIO = 0.55
FADEOUT_DUR     = 5.0
XFADE_DUR       = 1.5


# ── Beat detection ───────────────────────────────────────────────────────────

def detect_onsets(audio_path):
    r = subprocess.run(
        ["ffmpeg", "-i", audio_path,
         "-af", f"lowpass=f={BASS_LOWPASS}",
         "-f", "s16le", "-acodec", "pcm_s16le",
         "-ar", str(SAMPLE_RATE), "-ac", "1", "-"],
        capture_output=True, timeout=120,
    )
    if not r.stdout:
        return []
    n = len(r.stdout) // 2
    samples = list(struct.unpack(f"<{n}h", r.stdout))

    hop      = max(1, int(SAMPLE_RATE * HOP_MS / 1000))
    min_gap  = int(MIN_GAP_S * 1000 / HOP_MS)
    rms = []
    for i in range(0, len(samples) - hop, hop):
        chunk = samples[i:i + hop]
        rms.append(math.sqrt(sum(s * s for s in chunk) / hop) / 32768.0)
    if len(rms) < 4:
        return []

    flux     = [max(0.0, rms[i] - rms[i - 1]) for i in range(1, len(rms))]
    half_win = max(1, int(250 / HOP_MS))
    onsets   = []
    last_idx = -min_gap
    for i in range(half_win, len(flux) - half_win):
        local_mean = sum(flux[i - half_win:i + half_win]) / (2 * half_win)
        if (flux[i] > local_mean * 2.5
                and flux[i] > 0.001
                and (i - last_idx) > min_gap
                and flux[i] == max(flux[i - 2:i + 3])):
            onsets.append(round((i + 1) * hop / SAMPLE_RATE, 3))
            last_idx = i
    return onsets


def build_enable_chains(onsets):
    if not onsets:
        return ["0"]
    chains = []
    for i in range(0, len(onsets), CHUNK_SIZE):
        chunk = onsets[i:i + CHUNK_SIZE]
        chains.append("+".join(
            f"between(t,{t:.3f},{t + FLASH_DUR:.3f})" for t in chunk
        ))
    return chains


# ── FFmpeg filter complex ────────────────────────────────────────────────────

def _build_logo_chain(logo_h, chains):
    geq_args = (
        "r='min(r(X,Y)*1.35,255)':"
        "g='min(g(X,Y)*1.35,255)':"
        "b='min(b(X,Y)*1.35,255)':"
        "a='alpha(X,Y)'"
    )
    parts = [
        f"[1:v]scale=-1:{logo_h}:force_original_aspect_ratio=decrease,"
        f"format=rgba[logo_s]"
    ]
    for i, expr in enumerate(chains):
        inp = "[logo_s]" if i == 0 else f"[lc{i-1}]"
        out = f"[lc{i}]" if i < len(chains) - 1 else "[logo]"
        parts.append(f"{inp}geq={geq_args}:enable='{expr}'{out}")
    return ";\n".join(parts)


def build_main_filter(audio_duration, enable_chains):
    fade_start = max(0.0, audio_duration - FADEOUT_DUR)
    logo_h     = int(880 * LOGO_AREA_RATIO)
    logo_chain = _build_logo_chain(logo_h, enable_chains)
    return (
        "[0:v]"
        "scale=3000:1800:force_original_aspect_ratio=increase,"
        "crop=1920:1080:"
        "x='500*(0.5+0.5*sin(t/5))':"
        "y='300*(0.5+0.5*sin(t/3+0.8))',"
        "gblur=sigma=1,"
        "eq=brightness=-0.12:saturation=0.80,"
        "noise=alls=6:allf=t+u,"
        "format=yuv420p"
        "[bg];\n"
        f"{logo_chain};\n"
        "[2:a]"
        "showfreqs=s=1920x200:mode=bar:ascale=cbrt:fscale=log:"
        "win_size=4096:win_func=hann:averaging=3:"
        "colors=0x8844FF"
        "[bars];\n"
        "[bg][bars]overlay=0:880:format=auto[bg_bars];\n"
        "[bg_bars][logo]overlay=(W-w)/2:(880-h)/2:format=auto[composed];\n"
        f"[composed]fade=t=out:st={fade_start:.2f}:d={FADEOUT_DUR:.1f},"
        "format=yuv420p[out]"
    )


# ── Helpers ──────────────────────────────────────────────────────────────────

def get_duration(path):
    r = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, timeout=30,
    )
    try:
        return float(r.stdout.strip())
    except Exception:
        return None


def run_cmd(cmd):
    result = subprocess.run(cmd, capture_output=True, text=True,
                            encoding="utf-8", errors="replace", timeout=1800)
    if result.returncode != 0:
        print(result.stderr[-3000:], file=sys.stderr)
        return False
    return True


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio",  required=True)
    ap.add_argument("--cover",  required=True)
    ap.add_argument("--title",  default="")
    ap.add_argument("--genre",  default="")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    audio_dur = get_duration(args.audio) or 180.0
    use_intro = INTRO_PATH.exists()
    out       = Path(args.output)
    tmp_out   = out.with_name("_tmp_main.mp4")

    print(f"Audio duration: {audio_dur:.1f}s | intro: {use_intro}")

    print("Beat detection...")
    onsets = detect_onsets(args.audio)
    chains = build_enable_chains(onsets)
    print(f"{len(onsets)} beats -> {len(chains)} geq chains")

    meta = [
        "-map_metadata", "-1",
        "-metadata", f"title={args.title}",
        "-metadata", "artist=Majesty Music",
        "-metadata", f"album=Majesty Music — {args.genre.title()}",
        "-metadata", f"genre={args.genre.title()}",
    ]

    cmd1 = [
        "ffmpeg", "-y",
        "-loop", "1", "-i", args.cover,
        "-loop", "1", "-i", str(LOGO_PATH),
        "-i", args.audio,
        "-filter_complex", build_main_filter(audio_dur, chains),
        "-map", "[out]", "-map", "2:a",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-pix_fmt", "yuv420p", "-profile:v", "high", "-level:v", "4.0",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart", "-shortest",
        *meta,
        str(tmp_out) if use_intro else str(out),
    ]

    print("FFmpeg pass 1...")
    if not run_cmd(cmd1):
        sys.exit(1)

    if not use_intro:
        size = out.stat().st_size / 1048576
        print(f"Done: {out} ({size:.1f} MB)")
        return

    intro_dur    = get_duration(str(INTRO_PATH)) or 0.0
    xfade_offset = max(0.0, intro_dur - XFADE_DUR)
    intro_filter = (
        "[0:v]scale=1920:1080:force_original_aspect_ratio=increase,"
        "crop=1920:1080,fps=25,format=yuv420p[vi];"
        "[1:v]fps=25,format=yuv420p[vm];"
        f"[vi][vm]xfade=transition=fade:duration={XFADE_DUR}:offset={xfade_offset:.2f}[vout]"
    )

    cmd2 = [
        "ffmpeg", "-y",
        "-i", str(INTRO_PATH),
        "-i", str(tmp_out),
        "-i", args.audio,
        "-filter_complex", intro_filter,
        "-map", "[vout]", "-map", "2:a",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-pix_fmt", "yuv420p", "-profile:v", "high", "-level:v", "4.0",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart", "-shortest",
        *meta,
        str(out),
    ]

    print("FFmpeg pass 2 (intro xfade)...")
    if not run_cmd(cmd2):
        tmp_out.unlink(missing_ok=True)
        sys.exit(1)

    tmp_out.unlink(missing_ok=True)
    size = out.stat().st_size / 1048576
    print(f"Done: {out} ({size:.1f} MB)")


if __name__ == "__main__":
    main()
