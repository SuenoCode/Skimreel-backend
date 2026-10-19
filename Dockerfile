# SkimReel API
#
# ffmpeg and a writable volume are the reason this runs on a container host
# rather than a serverless platform: clip cutting shells out to ffmpeg, and
# downloads need disk that outlives a single request.

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    # yt-dlp needs a CA bundle; certifi ships one so it is not tied to the OS store.
    REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

# ffmpeg for clip/preview/transcode, libGL + libglib for OpenCV's wheel.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv

COPY requirements.txt requirements-transcribe.txt ./
# Transcription ships in the image: the API no longer runs on a serverless
# platform, so there is no 500MB bundle ceiling to design around.
RUN pip install --no-cache-dir -r requirements.txt -r requirements-transcribe.txt

COPY app.py ./

# Where downloads and generated clips go, and where the Whisper model is cached
# so a cold start does not re-download ~140MB. Both are set by the platform:
# fly.toml points SKIMREEL_DOWNLOAD_DIR at the mounted volume and these paths
# assume that same volume. Nothing is baked into the image.
ENV HF_HOME=/data/huggingface

EXPOSE 8080

# One worker on purpose: JOBS is in-process state, so a second worker would not
# see jobs started by the first. Scale with more machines, not more workers.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]