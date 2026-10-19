import os
import re
import json
import uuid
import hashlib
import threading
import time
import subprocess
import tempfile
import cv2
import numpy as np
import shutil
import sys
from fastapi import FastAPI, Form, BackgroundTasks, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
import yt_dlp

app = FastAPI()

# The SPA is served by Vercel while this API runs on a separate host, so every
# browser request is cross-origin. CORS_ALLOW_ORIGINS is a comma-separated list
# of exact origins; it defaults to localhost only, never to "*", because the API
# accepts uploads and spends the operator's bandwidth.
_allowed = os.environ.get("CORS_ALLOW_ORIGINS", "http://localhost:5173,http://localhost:8443")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _allowed.split(",") if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"],
)

JOBS = {}
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Where downloaded sources and generated clips live. A container host mounts a
# volume here so artifacts survive restarts; Vercel has no writable disk outside
# /tmp, hence the tempfile fallback for that (unsupported) target.
_dl = os.environ.get("SKIMREEL_DOWNLOAD_DIR", "").strip()
if _dl:
    DOWNLOAD_DIR = _dl
elif os.environ.get("VERCEL"):
    DOWNLOAD_DIR = os.path.join(tempfile.gettempdir(), "skimreel-downloads")
else:
    DOWNLOAD_DIR = os.path.join(BASE_DIR, "downloads")

# Cloud sync clients hold freshly written files open while they upload, which makes
# yt-dlp's final rename fail with WinError 32 on Windows. The download then dies
# before merging, leaving per-format fragments behind, and a clip cut from one of
# those comes out silent - while the request still returns 200. Detected at boot so
# the cause is named before a user waits on it.
SYNC_DIR_MARKERS = (
    ("OneDrive", "OneDrive"),
    ("Dropbox", "Dropbox"),
    ("Google Drive", "Google Drive"),
    ("GoogleDrive", "Google Drive"),
    ("iCloud", "iCloud"),
)


def _sync_provider(path: str):
    """Name the sync service owning this path, or None if it is not synced.

    OneDrive exposes its root as an environment variable, so check that first.
    The rest fall back to a substring test, which is less precise but catches the
    common 'Dropbox' or 'Google Drive' folder layouts.
    """
    try:
        resolved = os.path.abspath(path)
    except Exception:
        return None

    root = os.environ.get("OneDrive", "").strip()
    if root and os.path.abspath(root) in os.path.normcase(resolved):
        return "OneDrive"

    lowered = os.path.normcase(resolved)
    for marker, name in SYNC_DIR_MARKERS:
        if os.path.normcase(marker) in lowered:
            return name
    return None


SYNC_PROVIDER = _sync_provider(DOWNLOAD_DIR)
if SYNC_PROVIDER:
    print(
        f"[skimreel] WARNING: the downloads directory is inside a {SYNC_PROVIDER}"
        f"-synced folder ({DOWNLOAD_DIR}). The sync client locks files while it"
        " uploads them, so yt-dlp's final rename fails with WinError 32, the"
        " download never merges, and clips come out with no audio track. Set"
        " SKIMREEL_DOWNLOAD_DIR to a local path outside the sync root.",
        file=sys.stderr,
    )

# Vite builds the SPA into ../frontend/dist. On Vercel the CDN serves it from
# the frontend service, so this only matters for local `uvicorn app.py` runs.
WEB_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "frontend", "dist"))
os.makedirs(DOWNLOAD_DIR, exist_ok=True)
# The frontend is a sibling project in the combined checkout. In the standalone
# backend repository it is intentionally absent, so do not create or mount an
# empty frontend directory there.
if not os.environ.get("VERCEL") and os.path.isdir(os.path.dirname(WEB_DIR)):
    os.makedirs(WEB_DIR, exist_ok=True)

# Per-clip work (clip cut / preview render / transcription) must not run twice
# for the same moment when the user clicks fast, so guard each one with a lock.
ARTIFACT_LOCKS = {}
ARTIFACT_LOCKS_GUARD = threading.Lock()

WHISPER_MODELS = {}


def artifact_lock(key):
    with ARTIFACT_LOCKS_GUARD:
        if key not in ARTIFACT_LOCKS:
            ARTIFACT_LOCKS[key] = threading.Lock()
        return ARTIFACT_LOCKS[key]


def sweep_stale_artifacts() -> None:
    """Delete artifacts left behind by a previous process, at boot.

    DELETE /api/job/{id} only knows about jobs this process started: JOBS lives
    in memory, so a deploy, crash or restart orphans every source and preview
    already on the volume, and nothing would ever release them. This is the
    backstop for that case.

    Age, not job id, is the test, because an orphan cannot be distinguished from
    a live job by anything on disk. The window is long by default so a restart
    during a long download does not pull the source out from under it, and 0
    turns the sweep off.
    """
    try:
        ttl_hours = float(os.environ.get("SKIMREEL_ARTIFACT_TTL_HOURS", "24"))
    except ValueError:
        ttl_hours = 24.0
    if ttl_hours <= 0:
        return

    cutoff = time.time() - ttl_hours * 3600
    try:
        names = os.listdir(DOWNLOAD_DIR)
    except OSError:
        return

    removed = 0
    freed = 0
    for name in names:
        path = os.path.join(DOWNLOAD_DIR, name)
        try:
            if not os.path.isfile(path) or os.path.getmtime(path) > cutoff:
                continue
            freed += os.path.getsize(path)
            os.remove(path)
            removed += 1
        except OSError:
            continue

    if removed:
        print(
            f"[skimreel] startup sweep removed {removed} stale artifact(s), "
            f"{freed / (1024 * 1024):.0f} MB, older than {ttl_hours:g}h",
            file=sys.stderr,
        )


sweep_stale_artifacts()


def norm_quality(quality: str) -> str:
    """UI sends '1080p'; the format selectors below expect '1080'."""
    q = str(quality or "").strip().lower()
    if q.endswith("p"):
        q = q[:-1]
    return q if q in ("1080", "720", "480", "360") else "720"


def format_str_for(quality: str) -> str:
    """Build a yt-dlp selector that will not silently hand back 360p.

    The old string ended in a bare `/best`, and yt-dlp's "best" means the best
    *pre-merged* file, which on YouTube is format 18 - 640x360, unchanged since
    2010. Every clause here is height filtered, so a bare `/best` can never be
    reached.

    The first clause asks for H.264 video with AAC audio, which is what merges
    into an MP4 without re-encoding a byte: the download is then already the MP4
    everything downstream assumes, and costs nothing to get. The clauses after it
    are the previous selector unchanged, so a source that publishes no H.264 at
    all (a 4K-only feed, a site with one codec) still resolves to the same
    streams it always did - preferring MP4-native streams never trades away
    resolution. Whatever those clauses return, ensure_mp4() converts afterwards.

    Within the ceiling the best available stream is chosen deliberately: if a
    source only publishes 720p, asking for 1080p yields 720p rather than an error,
    because that is the source's ceiling and not a fault in the request. The
    resolution actually obtained is probed from the downloaded file and reported
    in the job status, so a caller is never left guessing.
    """
    height = norm_quality(quality)
    return (
        f"bestvideo[height<={height}][vcodec~='^(avc|h264)']"
        f"+bestaudio[acodec~='^(mp4a|aac)']/"
        f"bestvideo[height<={height}]+bestaudio/"
        f"best[height<={height}]/"
        f"bestvideo[height<={height}]"
    )


# --- yt-dlp options ---------------------------------------------------------
# yt-dlp rejects a cookie file that does not start with this line, so health
# checks for the same magic rather than its own idea of a valid cookie file.
NETSCAPE_MAGIC = "# Netscape HTTP Cookie File"

# YouTube answers anonymous requests from datacenter IPs with
# "Sign in to confirm you're not a bot". Cookies are the reliable fix, so both a
# Netscape cookies.txt path and a browser name are supported via the environment.

# player_client is deliberately NOT pinned.
#
# An earlier version hardcoded ["tv", "web_safari", "mweb", "web", "android"] and
# that list was the cause of the 640x360 output: every client in it was either
# dead or capped, so yt-dlp quietly fell through to format 18 (the 360p
# pre-merged stream). Pinning is worse than useless here because the set of
# clients that work changes with YouTube and with the yt-dlp version, and a stale
# pin silently costs resolution instead of erroring.
#
# Measured on yt-dlp 2026.08.19 with Node 24 available for nsig extraction:
#   default clients                 1920x1080 available  <- what we now use
#   player_client=web               no formats (PO token gated)
#   player_client=mweb              no formats (PO token gated)
# So the defaults are both the simplest and the highest quality; the only reason
# to pin would be a format check that showed defaults failing, which it does not.


def _curl_cffi_available():
    try:
        import curl_cffi  # noqa: F401
        return True
    except ImportError:
        return False


def _write_inline_cookies(text: str) -> str:
    """Materialise YTDLP_COOKIES into a file yt-dlp can read.

    yt-dlp only accepts a cookie *path*. The filename embeds a hash of the
    contents so a rotated secret never serves a stale file, and it is rewritten
    whenever the hash changes. Permissions are owner-only because this is a live
    session for a real account.
    """
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    path = os.path.join(tempfile.gettempdir(), f"skimreel-cookies-{digest}.txt")
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text if text.endswith("\n") else text + "\n")
    except OSError:
        # A read-only /tmp would make transcription unavailable, but analysis and
        # clip cutting should still work, so fall back to no cookies.
        return ""
    return path


def _materialize_cookie_file(source: str) -> str:
    """Copy a read-only cookie secret to a writable temp file for yt-dlp."""
    try:
        with open(source, "rb") as fh:
            payload = fh.read()
    except OSError:
        return ""

    digest = hashlib.sha256(payload).hexdigest()[:16]
    path = os.path.join(tempfile.gettempdir(), f"skimreel-cookies-file-{digest}.txt")
    if not os.path.exists(path):
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "wb") as fh:
                fh.write(payload)
        except FileExistsError:
            pass
        except OSError:
            return ""
    return path


def _usable_cookie_text(text: str) -> bool:
    """True if this is a cookie payload yt-dlp will accept.

    yt-dlp requires the Netscape magic on the first line and refuses anything
    else with "Cookies file must be Netscape formatted", which is the single most
    common way to point a config at the wrong file. Both the download options
    and /api/health ask this, so health cannot report a green light for a file
    that every download would then reject.
    """
    return text.startswith(NETSCAPE_MAGIC)


def _usable_cookie_file(path: str) -> tuple[bool, str]:
    """(accepted, why not) for a cookie file path."""
    if not os.path.exists(path):
        return False, "missing"
    if not os.access(path, os.R_OK):
        return False, "unreadable"
    try:
        with open(path, "r", encoding="utf-8-sig", errors="replace") as fh:
            head = fh.read(4096)
    except OSError as e:
        return False, str(e)
    if not head.strip():
        return False, "empty"
    if not _usable_cookie_text(head):
        return False, "not Netscape format"
    return True, ""


def base_ydl_opts() -> dict:
    opts = {
        'geo_bypass': True,
        'nocheckcertificate': True,
        'quiet': True,
        'no_warnings': True,
        # Download progress reaches the UI through a progress_hook, not through
        # the console. Left on, yt-dlp prints a line per fragment, which floods
        # the uvicorn log with the 600-line tail of a single download and buries
        # the request log next to it.
        'noprogress': True,
        'retries': 3,
        # No player_client / extractor_args here on purpose: pinning the client was
        # what forced the 640x360 output. Letting yt-dlp pick its own defaults is
        # both simpler and higher quality, and it means a yt-dlp upgrade can fix
        # client selection without touching this file.
    }

    # Render runs yt-dlp in a container where Deno is installed by Dockerfile.
    # Select it explicitly so YouTube's JavaScript challenge solver is used even
    # when yt-dlp cannot infer the runtime from the process environment.
    deno_path = shutil.which("deno")
    if deno_path:
        opts["js_runtimes"] = {"deno": {"path": deno_path}}

    # Cloud hosts often use IP ranges YouTube challenges or blocks. Allow the
    # operator to route yt-dlp's requests through an egress IP YouTube accepts.
    # Keep this environment setting secret; the URL may contain credentials.
    proxy = os.environ.get("YTDLP_PROXY", "").strip()
    if proxy:
        opts["proxy"] = proxy

    cookies_file = os.environ.get("YTDLP_COOKIES_FILE", "").strip()
    if cookies_file:
        # Resolve a bare filename against the repo so the uvicorn process cwd
        # does not decide whether it resolves.
        if not os.path.isabs(cookies_file):
            cookies_file = os.path.join(BASE_DIR, cookies_file)
        cookies_file = os.path.abspath(cookies_file)
        # Same acceptance test as health, so a file that is not usable is not
        # handed to yt-dlp: it would fail every download with a message about
        # formatting rather than about the bot check the user is actually
        # hitting. A rejected file falls through to the next form.
        ok, _why = _usable_cookie_file(cookies_file)
        if ok:
            # Render secret files are read-only. yt-dlp may refresh its cookie
            # jar when closing, so give it a writable temp copy.
            writable_copy = _materialize_cookie_file(cookies_file)
            if writable_copy:
                opts['cookiefile'] = writable_copy

    cookies_from_browser = os.environ.get("YTDLP_COOKIES_FROM_BROWSER", "").strip()
    # "false" is how yt-dlp spells "no browser"; passed through as a name it
    # becomes a request for a browser literally called false, and every
    # download fails on it.
    if cookies_from_browser.lower() in ("false", "none", "0"):
        cookies_from_browser = ""
    if cookies_from_browser and not opts.get("cookiefile"):
        opts['cookiesfrombrowser'] = tuple(
            part.strip() for part in cookies_from_browser.split(":")
        )

    # Container platforms pass secrets as env vars, and shipping a cookie file
    # in the image would bake a live account session into it. YTDLP_COOKIES wins
    # only when no readable cookie file was found, so a file on the volume stays
    # authoritative.
    cookies_inline = os.environ.get("YTDLP_COOKIES", "").strip()
    if cookies_inline and _usable_cookie_text(cookies_inline) and not opts.get("cookiefile"):
        opts['cookiefile'] = _write_inline_cookies(cookies_inline)

    impersonate = os.environ.get("YTDLP_IMPERSONATE", "").strip()
    if impersonate and _curl_cffi_available():
        opts['impersonate'] = impersonate

    return opts


SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]")
VIDEO_EXTS = (".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".mpg", ".mpeg", ".wmv", ".flv")

# One bar covers the three stages a job goes through, and each one ends at a
# published percentage so the UI can cut the bar into three bars and show the
# running stage's own progress - not the job's overall total, which is the
# number that made a two-thirds-full bar read as "87%". The transfer owns the
# most because yt-dlp reports exact byte counts for it and because it is by far
# the longest wait: it used to sit at a flat 5% for the ten minutes a 600 MB
# source took, which read as a hang rather than a download.
PCT_START = 0
PCT_TRANSFER_END = 60
PCT_MOTION_END = 90
PCT_PICK_END = 99


def human_bytes(n) -> str:
    """Byte count for a human, at the precision the number deserves."""
    try:
        n = float(n)
    except (TypeError, ValueError):
        return "?"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


class DownloadProgress:
    """Publishes yt-dlp's byte counters as job progress the UI can render.

    Two things make the raw hook unusable on its own. It fires once per output
    file, and a `bestvideo+bestaudio` job writes two, so reading it directly
    restarts the figure from zero when the audio stream begins; the bytes of the
    files already written are summed into the one in flight to keep a single
    figure for the whole job. And the percentage yt-dlp reports is not
    monotonic - a progressive stream revises its length estimate as it goes, so
    its own log shows 43.5% then 42.6% - so the high water mark is kept and the
    bar never runs backwards.
    """

    def __init__(self, job_id: str):
        self.job_id = job_id
        self.written = 0
        self.written_total = 0
        self.peak_pct = 0.0
        self.last_line = ""

    def __call__(self, d: dict):
        j = JOBS.get(self.job_id)
        if j is None:
            # Released while the download was still running.
            return

        info = d.get("info_dict") or {}
        downloaded = int(d.get("downloaded_bytes") or 0)
        total = int(
            d.get("total_bytes")
            or d.get("total_bytes_estimate")
            # Both are absent while a stream is being sized from scratch; the
            # format dict still carries what the extractor advertised, which is
            # what makes a percentage possible at all here.
            or info.get("filesize")
            or info.get("filesize_approx")
            or 0
        )

        status = d.get("status")
        if status == "finished":
            self.written += downloaded
            self.written_total += total or downloaded
        elif status == "error":
            return

        have = self.written + downloaded
        of = self.written_total + total
        pct = (have / of * 100.0) if of > 0 else 0.0
        pct = max(self.peak_pct, min(pct, 100.0))
        self.peak_pct = pct

        j["pct"] = PCT_START + int((PCT_TRANSFER_END - PCT_START) * pct / 100.0)
        j["status"] = "Downloading source..."

        parts = []
        if status == "finished":
            # One stream is written. Its bytes are already folded into `written`
            # for the total, so counting them again here would read as more than
            # the whole file.
            parts.append(f"Downloaded {human_bytes(downloaded)}")
        elif total:
            parts.append(f"{human_bytes(have)} of {human_bytes(of)}")
        elif downloaded > 0:
            # No length was ever advertised, so bytes are all there is to say.
            parts.append(f"{human_bytes(have)} so far")
        if d.get("speed"):
            parts.append(f"{human_bytes(d['speed'])}/s")
        eta = d.get("eta")
        if eta and status != "finished":
            parts.append(f"{eta // 60} min left" if eta >= 60 else f"{eta}s left")

        # The hook fires per fragment, so only publish when the human-readable
        # line actually changes instead of writing to the job dict on every tick.
        line = " · ".join(parts)
        if line and line != self.last_line:
            self.last_line = line
            j["detail"] = line


# ffprobe reports the QuickTime family under any of these names, and all of them
# are containers ffmpeg writes as .mp4 - so an MP4 check has to accept them all
# or it will transcode a file that was already fine.
MP4_CONTAINERS = ("mov", "mp4", "m4a", "3gp", "3g2", "mj2")


def probe_codecs(path: str):
    """(container, video codec, audio codec) for a file, or None if unreadable.

    Audio is None for a genuinely silent source, which is valid and must not be
    confused with "no audio stream found" - the caller treats a silent video as
    already-correct rather than re-encoding it.
    """
    try:
        res = subprocess.run(
            ["ffprobe", "-v", "error",
             "-show_entries", "format=format_name:stream=codec_type,codec_name",
             "-of", "json", path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=180,
        )
        info = json.loads(res.stdout.decode("utf-8", "ignore") or "{}")
    except Exception:
        return None

    container = ((info.get("format") or {}).get("format_name") or "").split(",")[0].lower()
    video = audio = None
    for stream in info.get("streams") or []:
        kind = stream.get("codec_type")
        if kind == "video" and video is None:
            video = (stream.get("codec_name") or "").lower()
        elif kind == "audio" and audio is None:
            audio = (stream.get("codec_name") or "").lower()
    return container, video, audio


def ensure_mp4(path: str) -> str:
    """Guarantee an H.264/AAC MP4 exists for this source, and return its path.

    Everything downstream assumes MP4: the clips are named .mp4, cut with
    `-c copy`, and handed to a browser, which will not decode VP9 or Opus in an
    MP4 container even though ffmpeg will happily write it there. yt-dlp can and
    does deliver exactly that - a bestvideo+bestaudio pair is usually VP9 with
    Opus - so the file is normalised here rather than trusted.

    Conversion is a lossless remux whenever the codecs are already compatible
    (H.264 video kept as-is, AAC kept, container repacked), and only re-encodes
    when the codecs genuinely are not: an unplayable clip is a much worse failure
    than a few minutes of CPU, and the alternative - a silent 200 with a video
    nobody can play - is the failure mode this whole file is written around.

    Best effort by design. If ffmpeg fails, the original file is left in place
    and the job carries on: the source may still be readable, and failing a whole
    analysis over a container fix would be the worse trade.
    """
    probed = probe_codecs(path)
    if probed is None:
        return path
    container, video, audio = probed

    if (
        os.path.splitext(path)[1].lower() == ".mp4"
        and container in MP4_CONTAINERS
        and video == "h264"
        and audio in (None, "", "aac")
    ):
        return path

    target = os.path.splitext(path)[0] + ".mp4"
    args = ["-y", "-i", path]
    if video == "h264":
        # The codec is already right; only the container needs rebuilding, and a
        # copy costs a second rather than a re-encode.
        args += ["-c:v", "copy"]
    else:
        args += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                 "-pix_fmt", "yuv420p"]
    if audio in (None, ""):
        args += ["-an"]
    elif audio == "aac":
        args += ["-c:a", "copy"]
    else:
        # Opus and Vorbis in an MP4 play in VLC and in nothing a browser ships.
        args += ["-c:a", "aac", "-b:a", "192k"]
    args += ["-movflags", "+faststart", "-f", "mp4"]

    # Written to a temp name and renamed, for the same reason encode_to() does
    # it: an interrupted run must not leave a truncated file that every later
    # request would treat as a finished download.
    tmp = target + ".part.mp4"
    try:
        ffmpeg_run(["ffmpeg", *args, tmp])
        os.replace(tmp, target)
    except RuntimeError as e:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
        print(
            f"[skimreel] could not convert {os.path.basename(path)} to mp4: {e}",
            file=sys.stderr,
        )
        return path

    if os.path.abspath(path) != os.path.abspath(target):
        # A .mkv or .webm that has just been converted is dead weight; leaving it
        # would double the space a single source takes.
        try:
            os.remove(path)
        except OSError:
            pass
    return target


def safe_upload_name(filename: str) -> str:
    """Reduce a client-supplied filename to a safe, unique-on-disk basename.

    Keeps the extension (ffmpeg and OpenCV both sniff by it) while stripping
    directory components and anything the host filesystem may reject.
    """
    base = os.path.basename(str(filename or "").replace("\\", "/"))
    base = SAFE_NAME_RE.sub("_", base).strip("._") or "upload"
    stem, ext = os.path.splitext(base)
    if ext.lower() not in VIDEO_EXTS:
        ext = ".mp4"
    return f"{stem[:80]}{ext}"


def probe_duration(path: str) -> float:
    """Container duration in seconds.

    OpenCV reports 0 (or a wildly wrong value) for some uploads, which silently
    skews every clip offset, so ask ffprobe first and fall back to OpenCV.
    """
    try:
        res = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60,
        )
        val = float(res.stdout.decode("utf-8", "ignore").strip())
        if val > 0:
            return val
    except Exception:
        pass

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return 0.0
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    cap.release()
    if fps > 0 and frames > 0:
        return float(frames) / float(fps)
    return 0.0


def _has_playable_video(path: str) -> bool:
    """True if the file has a video stream that is complete enough to cut from.

    Used to reject a source recovered by the filename glob in ensure_source,
    which by definition has not been through this code path and may be a
    fragment or a truncated download.

    Three checks, because neither one alone is sufficient. Duration is useless on
    its own: an interrupted faststart MP4 still carries a full moov atom in its
    first few megabytes, so ffprobe cheerfully reports the complete runtime of a
    file that is 1.7% downloaded. The size check is what actually catches that.
    """
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return False

    try:
        res = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=codec_type,duration,bit_rate",
             "-of", "csv=p=0", path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60,
        )
    except Exception:
        return False

    fields = res.stdout.decode("utf-8", "ignore").strip().split(",")
    if not fields or fields[0] != "video":
        return False

    try:
        duration = float(fields[1]) if len(fields) > 1 and fields[1] else 0.0
        bit_rate = int(fields[2]) if len(fields) > 2 and fields[2] else 0
    except ValueError:
        return False

    if duration <= 0:
        return False

    # A truncated file is far smaller than duration x bitrate. The tolerance is
    # generous because container overhead and VBR make the estimate rough in both
    # directions; a genuinely complete file measures well above 1.0.
    if bit_rate > 0:
        expected_bytes = duration * bit_rate / 8
        if expected_bytes > 0 and os.path.getsize(path) < expected_bytes * 0.9:
            return False

    return True


def open_capture(path: str):
    """Open a video for frame analysis, transcoding as a last resort.

    Returns (capture, cleanup). `cleanup` removes any temporary proxy file and
    must be called by the caller. OpenCV ships without every codec, so an upload
    it cannot decode is re-encoded through ffmpeg before we give up.
    """
    cap = cv2.VideoCapture(path)
    if cap.isOpened():
        return cap, (lambda: None)

    cap.release()
    tmp_dir = tempfile.mkdtemp(prefix="skimreel-proxy-")
    proxy = os.path.join(tmp_dir, "proxy.mp4")

    def cleanup():
        shutil.rmtree(tmp_dir, ignore_errors=True)

    ffmpeg_run([
        "ffmpeg", "-y", "-v", "error", "-i", path,
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
        "-pix_fmt", "yuv420p", "-c:a", "aac", proxy,
    ])

    if not os.path.exists(proxy) or os.path.getsize(proxy) == 0:
        cleanup()
        raise Exception(
            "Could not read this video. It may be corrupt or use a codec "
            "neither OpenCV nor ffmpeg can decode."
        )

    cap = cv2.VideoCapture(proxy)
    if not cap.isOpened():
        cap.release()
        cleanup()
        raise Exception("Could not read this video. Try re-exporting it as H.264 MP4.")
    return cap, cleanup


def cookie_status() -> dict:
    """What yt-dlp will actually be given for cookies, and why not if it won't.

    base_ydl_opts() accepts a Netscape cookie file, inline cookie text, or a
    local browser, in that order of precedence, so health has to consider all
    three - checking only the file form reported "cookies": false on installs
    that were working fine.

    It checks what will be used, not what is set, because an env var pointing
    at a path that is not there hands yt-dlp nothing at all while still looking
    configured. The format is checked too, using yt-dlp's own rule: it refuses a
    file that does not start with the Netscape magic line, so health should not
    call a saved HTML cookie export "configured".
    """
    path = os.environ.get("YTDLP_COOKIES_FILE", "").strip()
    if path:
        # base_ydl_opts resolves a bare filename against the repo, so health has
        # to as well or the two disagree about the same setting.
        if not os.path.isabs(path):
            path = os.path.join(BASE_DIR, path)
        path = os.path.abspath(path)
        ok, why = _usable_cookie_file(path)
        if ok:
            return {"configured": True, "source": f"YTDLP_COOKIES_FILE={path}"}
        return {"configured": False, "source": f"YTDLP_COOKIES_FILE={path} ({why})"}

    inline = os.environ.get("YTDLP_COOKIES", "").strip()
    if inline:
        if _usable_cookie_text(inline):
            return {"configured": True, "source": "YTDLP_COOKIES"}
        return {"configured": False, "source": "YTDLP_COOKIES (not Netscape format)"}

    browser = os.environ.get("YTDLP_COOKIES_FROM_BROWSER", "").strip()
    if browser:
        # "false" is yt-dlp's documented way of saying no browser at all, and it
        # would otherwise be passed through as a browser literally named false.
        if browser.lower() in ("false", "none", "0"):
            return {"configured": False, "source": "YTDLP_COOKIES_FROM_BROWSER=false"}
        return {"configured": True, "source": f"browser {browser}"}

    return {
        "configured": False,
        "source": (
            "no YTDLP_COOKIES_FILE, YTDLP_COOKIES or YTDLP_COOKIES_FROM_BROWSER set"
        ),
    }


@app.get("/api/health")
def health():
    """Liveness probe for the container host. Also reports whether yt-dlp is
    likely to pass YouTube's bot check, which is the usual reason for a
    "deployed but every link fails" report."""
    cookies = cookie_status()
    return {
        "ok": True,
        "ffmpeg": bool(shutil.which("ffmpeg")),
        "deno": bool(shutil.which("deno")),
        "yt_dlp": getattr(yt_dlp, "__version__", "unknown"),
        "youtube_proxy": bool(os.environ.get("YTDLP_PROXY", "").strip()),
        "cookies": cookies["configured"],
        # Which of the three cookie forms was found, or why none was: without it
        # a false here is indistinguishable from a typo in a secret name.
        "cookie_source": cookies["source"],
        "download_dir_writable": os.access(DOWNLOAD_DIR, os.W_OK),
        "download_dir_synced": SYNC_PROVIDER,
    }


@app.post("/api/analyze")
async def analyze(
    url: str = Form(""),
    length: float = Form(30.0),
    count: int = Form(8),
    quality: str = Form("720"),
    file: UploadFile = File(None)
):
    job_id = str(uuid.uuid4())
    
    # Handle file upload
    video_path = None
    original_url = url
    display_title = "Untitled"
    if file and file.filename:
        # The browser-supplied name is untrusted: it may carry separators or
        # characters the filesystem rejects, and it must never escape DOWNLOAD_DIR.
        safe_name = safe_upload_name(file.filename)
        video_path = os.path.join(DOWNLOAD_DIR, f"{job_id}_{safe_name}")
        try:
            with open(video_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer, length=1024 * 1024)
        except Exception as e:
            # The copy is what actually receives the body, and it fails the same
            # way any write does when the client hangs up mid-upload. No job
            # exists yet, so nothing would ever clean the partial file up - it
            # would just take up room until the volume filled.
            received = 0
            try:
                received = os.path.getsize(video_path)
            except OSError:
                pass
            try:
                os.remove(video_path)
            except OSError:
                pass
            raise HTTPException(
                status_code=500,
                detail=(
                    f"Upload failed after {human_bytes(received)}: {e}. "
                    "The partial file was removed, so it is safe to try again."
                ),
            ) from e
        original_url = f"file://{video_path}"
        # Show the name the user recognises, not the prefixed one on disk.
        display_title = os.path.basename(str(file.filename).replace("\\", "/")) or safe_name
    
    JOBS[job_id] = {
        "done": False,
        # An upload already has all its bytes by the time the job exists - the
        # browser posted the whole file to reach this point - so it starts where
        # a URL job's download ends. A URL job starts at the bottom instead, for
        # DownloadProgress to climb from.
        "status": "Queued",
        "pct": PCT_TRANSFER_END if video_path else PCT_START,
        "detail": "",
        "title": display_title,
        "moments": [],
        "error": None,
        "video_path": video_path,
        "video_url": None,
        "audio_url": None,
        "original_url": original_url,
        "quality": norm_quality(quality),
        "duration": 0,
        "transcripts": {}
    }
    threading.Thread(target=process_job, args=(job_id, original_url, length, count, quality), daemon=True).start()
    return {"job": job_id}

@app.get("/api/job/{job_id}")
def get_job(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    j = JOBS[job_id]
    return {
        "done": j["done"],
        "status": j["status"],
        "pct": j["pct"],
        "title": j["title"],
        "moments": j["moments"],
        "error": j["error"],
        "duration": j.get("duration", 0),
        "source_height": j.get("source_height", 0),
        "requested_quality": j.get("quality", ""),
        "transcripts": j.get("transcripts", {}),
        "detail": j.get("detail", ""),
    }

def purge_job_files(job_id: str) -> int:
    """Delete every on-disk artifact named with this job id. Returns the count.

    Artifacts are matched on the job-id prefix every one of them is named with,
    which also catches yt-dlp's per-format fragments and .part files that an
    interrupted download leaves behind - those are never served but they are just
    as large. Job ids are uuids, so no other file's name can be one id's prefix
    and this cannot delete an unrelated file.
    """
    removed = 0
    try:
        names = os.listdir(DOWNLOAD_DIR)
    except OSError:
        return 0
    for name in names:
        if not name.startswith(job_id):
            continue
        try:
            os.remove(os.path.join(DOWNLOAD_DIR, name))
            removed += 1
        except OSError:
            # A file still locked by a cloud-sync client is left for the next
            # sweep rather than failing the release.
            pass
    return removed


@app.delete("/api/job/{job_id}")
def delete_job(job_id: str):
    """Release a finished job's disk artifacts.

    The client calls this when the user starts a new analysis, which is the point
    at which the previous results are no longer being used. A 1080p source alone
    runs to hundreds of megabytes and every preview and clip adds to it, so a
    handful of analyses fills the volume the container mounts.

    Only a finished job can be released: a job still running holds a download or
    an encode open, and deleting its source under a live ffmpeg fails the request
    instead of reclaiming anything. 409 keeps that visible rather than silently
    leaking the disk.
    """
    j = JOBS.get(job_id)
    if j is None:
        # Already released, or the process restarted and lost it. Report success
        # so the client never blocks a new analysis on cleanup it cannot undo.
        return {"released": True, "removed": 0}
    if not j.get("done"):
        raise HTTPException(status_code=409, detail="Job is still running")

    removed = purge_job_files(job_id)
    JOBS.pop(job_id, None)
    return {"released": True, "removed": removed}


def moment_of(job_id: str, idx: int):
    if job_id not in JOBS or not JOBS[job_id]["done"]:
        raise HTTPException(status_code=404, detail="Job not finished or not found")
    moments = JOBS[job_id]["moments"]
    if idx < 0 or idx >= len(moments):
        raise HTTPException(status_code=404, detail="Moment index out of range")
    return moments[idx]


def _fetch_source(job_id: str, original_url: str, video_path: str):
    """Download the source video with yt-dlp's own downloader.

    This has to be a real download to disk, not a stream URL handed to ffmpeg.
    The googlevideo URLs yt-dlp resolves are signed and expect the headers yt-dlp
    itself sends, so a bare `ffmpeg -i <url>` is answered with
    "Server returned 403 Forbidden". Downloading first also means the analysis pass
    and every later clip, preview and transcript read one local file instead of
    re-resolving the source.
    """
    quality = JOBS[job_id].get("quality", "720")
    ydl_opts = base_ydl_opts()
    ydl_opts.update({
        'format': format_str_for(quality),
        'outtmpl': video_path,
        # Merge straight to MP4 where the chosen codecs allow it, which is the
        # normal case now that the selector prefers H.264 + AAC. When they do
        # not, ffmpeg still writes an MP4 container with the wrong codecs, and
        # ensure_mp4() rewrites it properly afterwards.
        'merge_output_format': 'mp4',
        # The hook is what the user sees; with it, yt-dlp's own progress output
        # is silenced in base_ydl_opts so it is not printed twice.
        'progress_hooks': [DownloadProgress(job_id)],
    })
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            return ydl.extract_info(original_url, download=True)
    except yt_dlp.utils.DownloadError as e:
        for leftover in (video_path + ".part", video_path):
            if os.path.exists(leftover):
                try:
                    os.remove(leftover)
                except OSError:
                    pass
        raise HTTPException(
            status_code=502,
            detail=(
                f"Could not download the source video: {e}. "
                "If this is a YouTube link, the usual cause is the bot check - "
                "close the browser holding the cookie database, then set "
                "YTDLP_COOKIES_FROM_BROWSER (e.g. chrome) or YTDLP_COOKIES. "
                "If valid cookies still fail from a cloud host, configure "
                "YTDLP_PROXY with an egress IP YouTube accepts. Cookies as configured: "
                f"{cookie_status()['source']}."
            ),
        ) from e


def ensure_source(job_id: str) -> str:
    """Return a local path to the full source video, downloading it if needed."""
    j = JOBS[job_id]
    video_path = j.get("video_path")
    if video_path and os.path.exists(video_path):
        return video_path

    original_url = j.get("original_url") or ""
    if original_url.startswith("file://"):
        raise HTTPException(status_code=404, detail="Source video file not found")

    video_path = os.path.join(DOWNLOAD_DIR, f"{job_id}_source.mp4")
    if not os.path.exists(video_path):
        # Serialised, and re-checked inside the lock: two requests for the same
        # job would otherwise run yt-dlp concurrently against one output path,
        # collide on the .part file, and fail the loser's rename with WinError 32
        # on Windows while the winner's download sits complete but unrenamed.
        with artifact_lock(f"source:{job_id}"):
            if not os.path.exists(video_path):
                info = _fetch_source(job_id, original_url, video_path)
                # Record the real title and duration while the metadata is in hand.
                # Downloading the source here means process_job no longer has a
                # separate extract to call, so without this the UI would show the
                # on-disk filename, which is just the job id.
                if isinstance(info, dict):
                    if info.get("title") and j.get("title") in (None, "", "Untitled"):
                        j["title"] = info["title"]
                    if info.get("duration"):
                        j["duration"] = float(info["duration"])

    if not os.path.exists(video_path):
        found = None
        for f in os.listdir(DOWNLOAD_DIR):
            # Generated artifacts share the job-id prefix, so a preview or clip
            # could otherwise be picked up here and handed to the cutter as if it
            # were the source.
            if "_preview" in f or "_clip_" in f:
                continue
            # yt-dlp downloads each format as its own fragment and merges
            # afterwards, so a download that dies before the merge leaves these
            # behind: {job}_source.f137.mp4 is the video-only stream and
            # {job}_source.f251.webm is the audio-only one. They are named with a
            # real video extension, so the suffix test below does not catch them.
            # Handing f137 to the cutter yields a 200 and a completely silent clip,
            # which is the worst possible outcome. yt-dlp removes them on success,
            # so their presence means the download did not complete.
            if re.search(r"\.f\d{3}\.", f) or f.endswith(".part"):
                continue
            if f.startswith(job_id) and f.endswith(('.mp4', '.mkv', '.webm', '.mov', '.ts')):
                found = os.path.join(DOWNLOAD_DIR, f)
                break
        if found:
            video_path = found

    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Failed to download source video")

    # Anything recovered by the glob is unverified, so probe it before the clip
    # cutter trusts it. A truncated or wrong-stream file that decodes to something
    # still produces a 200 with broken output; failing here names the real problem.
    # Audio is deliberately not required - a genuinely silent source is valid - but
    # the audio-only fragment is already excluded by the video-stream check.
    if not _has_playable_video(video_path):
        raise HTTPException(
            status_code=502,
            detail=(
                "The downloaded source file is not a usable video (no video "
                "stream, or it is truncated). This usually means the download was "
                "interrupted - commonly because the downloads folder is inside a "
                "cloud-synced location like OneDrive, which locks freshly written "
                "files and makes yt-dlp's rename fail. Set SKIMREEL_DOWNLOAD_DIR "
                "to a local path outside OneDrive and retry."
            ),
        )

    # Normalise the container and codecs once, the first time the source is
    # needed. A download can legitimately arrive as WebM, or as MP4 holding VP9
    # and Opus, and every clip, preview and transcript below assumes H.264/AAC
    # MP4: cut with `-c copy` from anything else, the result is a file browsers
    # refuse to play. Done after the playable check so a truncated download is
    # rejected before any CPU is spent on it.
    if not j.get("mp4_ready"):
        video_path = ensure_mp4(video_path)
        j["mp4_ready"] = True

    # Report the resolution actually obtained. Format selection is a ceiling, so a
    # 1080p request against a 360p source legitimately yields 360p; surfacing the
    # real height keeps that from looking like a silent downgrade.
    if not j.get("source_height"):
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            cap.release()
            if h:
                j["source_height"] = h

    j["video_path"] = video_path
    return video_path


def ffmpeg_run(cmd):
    """Run ffmpeg and fail loudly.

    No call site inspects the result, so without a check here a missing binary
    surfaces as a raw FileNotFoundError traceback and a failed encode surfaces as
    a misleading "Failed to generate clip" with ffmpeg's stderr discarded.
    """
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError:
        raise RuntimeError(
            "ffmpeg was not found on PATH. It is required for preview, "
            "Download MP4 and transcription - install it and restart the API."
        )
    if proc.returncode != 0:
        tail = proc.stderr.decode("utf-8", "replace").strip().splitlines()[-4:]
        raise RuntimeError("ffmpeg failed: " + " ".join(tail))
    return proc


def encode_to(args, out_path):
    """Publish an ffmpeg output atomically, failing with an actionable 503.

    ffmpeg writes incrementally, so an interrupted run leaves a non-empty partial
    file. Both callers treat "exists and non-empty" as a cache hit, so that
    truncation would be served permanently - encode to a temp name and rename only
    after ffmpeg reports success.

    The temp name keeps the real extension: ffmpeg picks its muxer from it, so
    `foo.mp4.part` fails to initialise with "Error opening output file".
    """
    root, ext = os.path.splitext(out_path)
    tmp = f"{root}.part{ext}"
    try:
        ffmpeg_run(["ffmpeg", *args, tmp])
        os.replace(tmp, out_path)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


@app.get("/api/clip/{job_id}/{idx}")
def get_clip(job_id: str, idx: int):
    m = moment_of(job_id, idx)
    video_path = ensure_source(job_id)

    out_clip = os.path.join(DOWNLOAD_DIR, f"{job_id}_clip_{idx}.mp4")
    if not os.path.exists(out_clip):
        with artifact_lock(f"clip:{job_id}:{idx}"):
            if not os.path.exists(out_clip):
                encode_to([
                    "-y", "-ss", str(m["start"]), "-i", video_path,
                    "-t", str(m["end"] - m["start"]), "-c", "copy",
                    "-movflags", "+faststart",
                ], out_clip)

    if os.path.exists(out_clip) and os.path.getsize(out_clip) > 0:
        return FileResponse(out_clip, media_type="video/mp4", filename=f"clip_{idx + 1}.mp4")
    raise HTTPException(status_code=500, detail="Failed to generate clip")


# Bumped whenever the preview encode settings change, so every machine with a warm
# cache picks up the new encode instead of serving stale files forever.
PREVIEW_CACHE_VERSION = 2


@app.get("/api/preview/{job_id}/{idx}")
def get_preview(job_id: str, idx: int):
    """Web-optimised re-encode of a moment so the browser can scrub it instantly.

    Re-encoded rather than stream-copied because the source can be 4K/HEVC, and a
    browser should not have to decode that to play a 30-second moment. Size and
    CRF are set for visual transparency; the encode is version-stamped so changing
    them later invalidates the cache.
    """
    m = moment_of(job_id, idx)
    video_path = ensure_source(job_id)

    out_preview = os.path.join(
        DOWNLOAD_DIR, f"{job_id}_preview{PREVIEW_CACHE_VERSION}_{idx}.mp4"
    )
    if not os.path.exists(out_preview) or os.path.getsize(out_preview) == 0:
        with artifact_lock(f"preview:{job_id}:{idx}"):
            if not os.path.exists(out_preview) or os.path.getsize(out_preview) == 0:
                encode_to([
                    "-y", "-ss", str(m["start"]), "-i", video_path,
                    "-t", str(m["end"] - m["start"]),
                    "-vf", "scale='min(1280,iw)':-2",
                    "-c:v", "libx264", "-preset", "fast", "-crf", "20",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                    "-c:a", "aac", "-b:a", "128k",
                ], out_preview)

    if os.path.exists(out_preview) and os.path.getsize(out_preview) > 0:
        return FileResponse(out_preview, media_type="video/mp4")
    raise HTTPException(status_code=500, detail="Failed to generate preview")


def _parse_ts(value: str) -> float:
    parts = value.strip().replace(",", ".").split(":")
    try:
        parts = [float(p) for p in parts]
    except ValueError:
        return 0.0
    seconds = 0.0
    for p in parts:
        seconds = seconds * 60 + p
    return seconds


def _parse_vtt(raw: str):
    cues = []
    for block in re.split(r"\n\s*\n", raw.replace("\r\n", "\n").replace("\r", "\n")):
        lines = [l.strip() for l in block.split("\n") if l.strip()]
        if not lines or lines[0].startswith(("WEBVTT", "NOTE", "STYLE", "Kind:", "Language:")):
            continue
        ti = next((i for i, l in enumerate(lines) if "-->" in l), None)
        if ti is None:
            continue
        match = re.match(r"([\d:.,]+)\s*-->\s*([\d:.,]+)", lines[ti])
        if not match:
            continue
        text = " ".join(lines[ti + 1:])
        text = re.sub(r"<[^>]+>", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            cues.append((_parse_ts(match.group(1)), _parse_ts(match.group(2)), text))
    # YouTube auto-captions repeat the previous line; collapse consecutive dupes.
    deduped = []
    for start, end, text in cues:
        if deduped and deduped[-1][2] == text:
            deduped[-1] = (deduped[-1][0], end, text)
        else:
            deduped.append((start, end, text))
    return deduped


def _parse_json3(raw: str):
    data = json.loads(raw)
    cues = []
    for event in data.get("events", []):
        segs = event.get("segs")
        if not segs:
            continue
        text = re.sub(r"\s+", " ", "".join(s.get("utf8", "") for s in segs)).strip()
        if not text:
            continue
        start = event.get("tStartMs", 0) / 1000.0
        cues.append((start, start + event.get("dDurationMs", 0) / 1000.0, text))
    deduped = []
    for start, end, text in cues:
        if deduped and deduped[-1][2] == text:
            deduped[-1] = (deduped[-1][0], end, text)
        else:
            deduped.append((start, end, text))
    return deduped


def _captions_from_source(job_id: str, start: float, end: float):
    """Free fallback: reuse captions the source already ships with (YouTube)."""
    import requests

    original_url = JOBS[job_id].get("original_url") or ""
    if original_url.startswith("file://"):
        return None

    ydl_opts = base_ydl_opts()
    ydl_opts['skip_download'] = True
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(original_url, download=False)
    except Exception:
        return None

    tracks = info.get("subtitles") or {}
    preferred = [k for k in ("en", "en-US", "en-GB") if k in tracks]
    pool = {k: v for k, v in preferred} or (dict(list(tracks.items())[:1]) if tracks else {})
    if not pool:
        auto = info.get("automatic_captions") or {}
        for key in ("en", "en-orig", "en-US", "a.en"):
            if key in auto:
                pool = {key: auto[key]}
                break
        else:
            if auto:
                pool = {list(auto.keys())[0]: auto[list(auto.keys())[0]]}
    if not pool:
        return None

    formats = list(pool.values())[0]
    pick = next((f for f in formats if f.get("ext") == "json3"), None) \
        or next((f for f in formats if f.get("ext") == "vtt"), None)
    if not pick:
        return None

    try:
        res = requests.get(pick["url"], timeout=20, verify=False)
        res.raise_for_status()
    except Exception:
        return None

    try:
        cues = _parse_json3(res.text) if pick.get("ext") == "json3" else _parse_vtt(res.text)
    except Exception:
        return None

    windowed = []
    for c_start, c_end, text in cues:
        if c_end <= start or c_start >= end:
            continue
        windowed.append({"start": max(0.0, c_start - start),
                         "end": min(end - start, c_end - start),
                         "text": text})
    if not windowed:
        return None
    return {"text": " ".join(c["text"] for c in windowed), "segments": windowed, "engine": "captions"}


def _whisper_segments(job_id: str, wav_path: str, offset: float, length: float):
    """Local speech-to-text. Prefers faster-whisper, falls back to openai-whisper."""
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        try:
            import whisper
        except ImportError:
            return None
        if "whisper" not in WHISPER_MODELS:
            WHISPER_MODELS["whisper"] = whisper.load_model("base")
        result = WHISPER_MODELS["whisper"].transcribe(wav_path, fp16=False)
        segments = [{"start": s["start"], "end": s["end"], "text": s["text"].strip()}
                    for s in result.get("segments", [])]
        return {"text": result.get("text", "").strip(), "segments": segments, "engine": "whisper"}

    if "faster-whisper" not in WHISPER_MODELS:
        WHISPER_MODELS["faster-whisper"] = WhisperModel("base", device="cpu", compute_type="int8")
    segments_iter, _info = WHISPER_MODELS["faster-whisper"].transcribe(
        wav_path, beam_size=1, vad_filter=True
    )
    segments = [{"start": s.start, "end": s.end, "text": s.text.strip()}
                for s in segments_iter if s.text and s.text.strip()]
    return {"text": " ".join(s["text"] for s in segments), "segments": segments, "engine": "faster-whisper"}


def _format_srt_time(seconds: float) -> str:
    ms = max(0, int(round(seconds * 1000)))
    h = ms // 3600000
    ms %= 3600000
    m = ms // 60000
    ms %= 60000
    s = ms // 1000
    ms %= 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _segments_to_srt(segments) -> str:
    blocks = []
    for i, seg in enumerate(segments, start=1):
        s_ts = _format_srt_time(seg.get("start", 0.0))
        e_ts = _format_srt_time(seg.get("end", 0.0))
        txt = str(seg.get("text", "")).strip()
        blocks.append(f"{i}\n{s_ts} --> {e_ts}\n{txt}")
    return "\n\n".join(blocks) + "\n"


@app.post("/api/transcribe/{job_id}/{idx}")
def post_transcribe(job_id: str, idx: int):
    m = moment_of(job_id, idx)
    cached = JOBS[job_id].get("transcripts", {}).get(str(idx))
    if cached:
        return cached

    with artifact_lock(f"transcribe:{job_id}:{idx}"):
        cached = JOBS[job_id].get("transcripts", {}).get(str(idx))
        if cached:
            return cached

        start, end = float(m["start"]), float(m["end"])
        length = max(0.1, end - start)
        video_path = ensure_source(job_id)

        with tempfile.TemporaryDirectory() as tmp:
            wav_path = os.path.join(tmp, "segment.wav")
            try:
                ffmpeg_run([
                    "ffmpeg", "-y", "-ss", str(start), "-i", video_path, "-t", str(length),
                    "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", wav_path
                ])
            except RuntimeError as e:
                raise HTTPException(status_code=503, detail=str(e)) from e
            if not os.path.exists(wav_path):
                raise HTTPException(status_code=422, detail="This clip has no audio track to transcribe.")

            try:
                result = _whisper_segments(job_id, wav_path, start, length)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")

        if not result or not result["segments"]:
            try:
                result = _captions_from_source(job_id, start, end)
            except Exception:
                result = None

        if not result:
            # Only reachable on a local checkout that skipped the optional
            # requirements file, or when the source has no speech at all.
            raise HTTPException(
                status_code=503,
                detail=(
                    "No speech could be transcribed from this clip. Local installs "
                    "need the optional engine: pip install -r requirements-transcribe.txt"
                ),
            )

        for seg in result["segments"]:
            seg["start"] = max(0.0, seg["start"])
            seg["end"] = min(length, seg["end"])
        if not result["text"]:
            result["text"] = " ".join(s["text"] for s in result["segments"])

        payload = {
            "idx": idx,
            "text": result["text"],
            "segments": result["segments"],
            "srt": _segments_to_srt(result["segments"]),
            "engine": result["engine"],
        }
        JOBS[job_id].setdefault("transcripts", {})[str(idx)] = payload
        return payload


@app.get("/api/transcript/{job_id}/{idx}")
def get_transcript_srt(job_id: str, idx: int):
    payload = JOBS[job_id].get("transcripts", {}).get(str(idx)) if job_id in JOBS else None
    if not payload:
        raise HTTPException(status_code=404, detail="No transcript yet. Run Transcribe first.")
    srt_name = f"skimreel_{idx + 1}.srt"
    return Response(
        content=payload["srt"],
        media_type="application/x-subrip",
        headers={"Content-Disposition": f'attachment; filename="{srt_name}"'},
    )


def fmt_time(seconds):
    s = max(0, int(round(seconds)))
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    if h:
        return f"{h}:{m:02d}:{sec:02d}"
    return f"{m}:{sec:02d}"

# get_stream_urls() used to hand a signed googlevideo URL to OpenCV for frame
# sampling. Those URLs expect the headers yt-dlp sends, so a bare request got 403
# and every URL job failed at "Analyzing video". process_job now downloads the
# source through yt-dlp instead, and ensure_source records the title and duration
# from the same extract, so this has no callers left.

def analyze_stream(video_url: str, duration: float, clip_length: float, count: int, job_id: str):
    j = JOBS[job_id]
    
    if duration <= 0:
        duration = 10.0
        
    N = max(10, min(200, int(duration / 5)))
    step = duration / N
    times = [(i + 0.5) * step for i in range(N)]
    
    cap, cleanup = open_capture(video_url)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30

        motion_scores = []
        for i, t in enumerate(times):
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ret1, frame1 = cap.read()
            if not ret1:
                motion_scores.append(0.0)
                continue

            cap.set(cv2.CAP_PROP_POS_MSEC, min(duration * 1000, (t + 0.7) * 1000))
            ret2, frame2 = cap.read()
            if not ret2:
                frame2 = frame1

            f1 = cv2.resize(cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY), (64, 36))
            f2 = cv2.resize(cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY), (64, 36))
            diff = np.mean(np.abs(f1.astype(float) - f2.astype(float)) / 255.0)
            motion_scores.append(float(diff))

            pct = PCT_TRANSFER_END + int((PCT_MOTION_END - PCT_TRANSFER_END) * (i + 1) / N)
            j["pct"] = pct
            j["status"] = f"Analyzing... {i+1}/{N}"
    finally:
        cap.release()
        cleanup()

    # Sampling is done; what is left is the ranking. It takes milliseconds, but
    # it is a distinct phase, so the bar's third segment has an honest end to
    # fill to rather than a label that never appears.
    j["status"] = "Picking moments..."
    j["pct"] = PCT_MOTION_END

    if motion_scores:
        max_m = max(motion_scores) or 1.0
        norm_m = [m / max_m for m in motion_scores]
    else:
        norm_m = [0.0] * len(times)
        
    smoothed = []
    for i in range(len(norm_m)):
        prev_val = norm_m[i-1] if i > 0 else norm_m[i]
        next_val = norm_m[i+1] if i < len(norm_m)-1 else norm_m[i]
        smoothed.append(prev_val * 0.25 + norm_m[i] * 0.5 + next_val * 0.25)
    
    indices = sorted(range(len(smoothed)), key=lambda i: smoothed[i], reverse=True)
    pick = []
    for i in indices:
        if len(pick) >= count:
            break
        if all(abs(times[i] - times[j_idx]) >= clip_length for j_idx in pick):
            pick.append(i)
            
    moments = []
    for i in pick:
        t = times[i]
        s = max(0.0, min(t - clip_length / 2, duration - clip_length))
        e = min(duration, s + clip_length)
        score = int(round(smoothed[i] * 100))
        why = "fast action or high motion" if smoothed[i] > 0.4 else "standout energy"
        moments.append({
            "start": float(s),
            "end": float(e),
            "range": f"{fmt_time(s)} – {fmt_time(e)}",
            "score": score,
            "why": why,
            "peak": float(t),
            "transcript": ""
        })
    
    moments.sort(key=lambda m: m["score"], reverse=True)
    j["pct"] = PCT_PICK_END
    return moments

def process_job(job_id: str, url: str, clip_length: float, count: int, quality: str = "720"):
    j = JOBS[job_id]
    try:
        is_local_file = url.startswith("file://")
        
        if is_local_file:
            j["status"] = "Analyzing local file..."
            j["pct"] = PCT_TRANSFER_END
            video_path = url.replace("file://", "")
            if not os.path.exists(video_path):
                raise Exception("Local video file not found")
            
            duration = probe_duration(video_path)
            if duration <= 0:
                raise Exception(
                    "Could not read this video's duration. It may be corrupt "
                    "or in a format this build cannot decode."
                )

            j["video_path"] = video_path
            if not j.get("title") or j["title"] == "Untitled":
                j["title"] = os.path.basename(video_path)
            j["duration"] = duration
        else:
            j["status"] = "Downloading source..."
            j["pct"] = PCT_START

            # Download through yt-dlp rather than streaming the signed googlevideo
            # URL into OpenCV. Those URLs are bound to the requesting IP and expect
            # the headers yt-dlp sends; a bare ffmpeg/OpenCV request gets 403, which
            # surfaced as "Error opening input: Server returned 403 Forbidden" and
            # made every URL job fail. yt-dlp already does this download for
            # ensure_source, so doing it here means one download serves both the
            # analysis and every later clip/preview/transcribe.
            video_path = ensure_source(job_id)

            if not j.get("title") or j["title"] == "Untitled":
                j["title"] = os.path.basename(video_path)
            duration = j.get("duration") or 0
            if duration <= 0:
                duration = probe_duration(video_path)
                j["duration"] = duration

        if duration <= 0:
            duration = 10.0

        j["status"] = "Analyzing video..."
        # The download phase owned everything up to here; take the bar to where it
        # ended so the first sampled frame does not drop it back.
        j["pct"] = PCT_TRANSFER_END

        video_url = j["video_path"]

        moments = analyze_stream(video_url, duration, clip_length, count, job_id)
        
        j["moments"] = moments
        j["done"] = True
        j["status"] = "Analysis complete. Clips will download on demand."
        j["pct"] = 100
        
    except Exception as e:
        j["done"] = True
        j["error"] = str(e)
        j["status"] = "Error"
        # A failed job is never resumed: the UI shows the message and sends the
        # user back to the input, so whatever it managed to download or upload
        # would sit on the volume until the next analysis or the next restart.
        # Nothing is served from a failed job, so it is safe to take back now.
        # The job entry stays, because the client still has to read the error.
        purge_job_files(job_id)

# Serve the React build for local runs (must be after all API routes).
# Skipped on Vercel: the frontend service owns static assets there.
if not os.environ.get("VERCEL") and os.path.isdir(WEB_DIR):
    app.mount("/assets", StaticFiles(directory=os.path.join(WEB_DIR, "assets")), name="assets")
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
