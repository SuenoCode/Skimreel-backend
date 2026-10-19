"""Download a YouTube video locally and send it to the SkimReel API.

This keeps Firefox cookies and YouTube access on the user's computer. Render
only receives the resulting video upload and performs the analysis.
"""

from __future__ import annotations

import argparse
import configparser
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import requests
import yt_dlp


DEFAULT_API = "https://skimreel-backend.onrender.com"


def firefox_profile() -> str | None:
    """Return the actual Firefox profile directory name for the default profile."""
    appdata = os.environ.get("APPDATA")
    if not appdata:
        return None
    ini_path = Path(appdata) / "Mozilla" / "Firefox" / "profiles.ini"
    if not ini_path.exists():
        return None

    parser = configparser.RawConfigParser()
    parser.read(ini_path, encoding="utf-8")

    # Modern Firefox records the active profile under an [Install...] section.
    # Prefer that path over ProfileN's Default=1 flag, which can identify an
    # older profile kept for compatibility.
    for section in parser.sections():
        if not section.lower().startswith("install"):
            continue
        path = parser.get(section, "Default", fallback="").strip()
        if not path:
            continue
        profile_path = (
            (ini_path.parent / path) if not os.path.isabs(path) else Path(path)
        ).resolve()
        if profile_path.is_dir():
            return profile_path.name

    candidates: list[tuple[bool, str]] = []
    for section in parser.sections():
        if not section.lower().startswith("profile"):
            continue
        path = parser.get(section, "Path", fallback="").strip()
        if not path:
            continue
        is_relative = parser.getint(section, "IsRelative", fallback=1) == 1
        profile_path = (
            (ini_path.parent / path) if is_relative else Path(path)
        ).resolve()
        if profile_path.is_dir():
            candidates.append((parser.getint(section, "Default", fallback=0) == 1, profile_path.name))

    if not candidates:
        return None
    defaults = [name for is_default, name in candidates if is_default]
    return defaults[0] if defaults else candidates[0][1]


def download_source(url: str, workdir: Path, profile: str | None) -> Path:
    deno = shutil.which("deno")
    if not deno:
        raise RuntimeError(
            "Deno was not found. Install it with: winget install --id DenoLand.Deno -e"
        )

    browser = ("firefox", profile) if profile else ("firefox",)
    last_reported = [-10]

    def progress_hook(status: dict) -> None:
        if status.get("status") != "downloading":
            return
        total = status.get("total_bytes") or status.get("total_bytes_estimate") or 0
        downloaded = status.get("downloaded_bytes") or 0
        if total:
            pct = int(downloaded * 100 / total)
            if pct >= last_reported[0] + 10:
                last_reported[0] = pct
                print(f"Downloading locally: {pct}%")

    opts = {
        "cookiesfrombrowser": browser,
        "js_runtimes": {"deno": {"path": deno}},
        "format": "bestvideo[height<=720]+bestaudio/best[height<=720]/best",
        "merge_output_format": "mp4",
        "outtmpl": str(workdir / "%(id)s.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": False,
        "noprogress": True,
        "progress_hooks": [progress_hook],
        "retries": 3,
    }

    print(f"Using Firefox profile: {profile or 'auto'}")
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        video_id = info.get("id", "")

    files = [
        p for p in workdir.iterdir()
        if p.is_file()
        and video_id in p.name
        and not p.name.endswith((".part", ".ytdl"))
    ]
    if not files:
        raise RuntimeError("yt-dlp completed but no downloaded video file was found")
    return max(files, key=lambda p: p.stat().st_size)


def upload_and_wait(api: str, video: Path, args: argparse.Namespace) -> None:
    api = api.rstrip("/")
    print(f"Uploading {video.name} ({video.stat().st_size / 1024 / 1024:.1f} MB) to {api}")
    with video.open("rb") as fh:
        response = requests.post(
            f"{api}/api/analyze",
            files={"file": (video.name, fh, "video/mp4")},
            data={
                "length": str(args.length),
                "count": str(args.count),
                "quality": args.quality,
            },
            timeout=1800,
        )
    response.raise_for_status()
    job = response.json()["job"]
    print(f"Analysis job: {job}")

    while True:
        status_response = requests.get(f"{api}/api/job/{job}", timeout=30)
        status_response.raise_for_status()
        status = status_response.json()
        detail = status.get("detail") or ""
        print(
            f"{status.get('pct', 0):>3}% {status.get('status', 'Working')}"
            + (f" — {detail}" if detail else "")
        )
        if status.get("done"):
            if status.get("error"):
                raise RuntimeError(status["error"])
            print("Analysis complete. Open SkimReel in your browser to view the result.")
            return
        time.sleep(3)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url", help="YouTube video URL")
    parser.add_argument("--api", default=DEFAULT_API, help="SkimReel backend URL")
    parser.add_argument("--length", type=float, default=30.0)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--quality", choices=("360", "480", "720", "1080"), default="720")
    parser.add_argument("--profile", help="Firefox profile folder; auto-detected by default")
    parser.add_argument("--keep", action="store_true", help="Keep the downloaded local video")
    args = parser.parse_args()

    workdir = Path(tempfile.mkdtemp(prefix="skimreel-helper-"))
    try:
        profile = args.profile or firefox_profile()
        video = download_source(args.url, workdir, profile)
        upload_and_wait(args.api, video, args)
        if args.keep:
            target = Path.cwd() / video.name
            shutil.copy2(video, target)
            print(f"Saved local copy: {target}")
        return 0
    except KeyboardInterrupt:
        print("Cancelled", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    finally:
        if not args.keep:
            shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
