# SkimReel

Find the good parts of a video. Point it at a YouTube or Twitch link or drop in a
local file, and it scores the footage for motion, cuts the strongest moments, and
generates clips, previews, and subtitles on demand.

## Why the backend is not on Vercel

The API was originally deployed to Vercel as a serverless function. That cannot
work, for reasons that are structural rather than fixable with configuration:

| Need | Vercel function | Why |
| --- | --- | --- |
| `ffmpeg` for clip/preview/transcode | not installed | binary must be in the image |
| Disk for downloaded sources | ephemeral, read-only outside `/tmp` | artifacts vanish on cold start |
| Long-running analysis | request-timeout bound | a Whisper pass can exceed it |
| Dependency bundle | 500 MB cap | OpenCV + `faster-whisper` exceeded it |

So the split is: **frontend on Vercel** (static, CDN-served) and **API on
Fly.io** (a container with ffmpeg and a volume). Same domain experience, no
serverless ceiling.

```
Browser ──► Vercel (SPA, static assets)
   │
   └─ CORS ─► Fly.io API  (FastAPI + ffmpeg + /data volume)
```

## Layout

```
backend/                  FastAPI API, Python dependencies, and Fly.io deployment
frontend/                 Vite + React SPA and Vercel build output
vercel.json               Frontend-only Vercel service configuration
README.md                 Project documentation
```

## Local development

Two processes, the usual split:

```bash
# 1. API on :8000
cd backend
pip install -r requirements.txt
uvicorn app:app --reload --port 8000

# 2. SPA on :8443, proxying /api to :8000
cd ..
cd frontend
npm install
npm run dev
```

Optional local transcription:

```bash
cd backend
pip install -r requirements-transcribe.txt
```

## YouTube bot check

YouTube refuses anonymous requests from datacenter IPs with *"Sign in to confirm
you're not a bot"*. There is no code change that avoids it; you need to
authenticate. Export cookies in Netscape format:

```bash
yt-dlp --cookies-from-browser chrome --cookies cookies.txt "<any url>"
```

Then point the API at the file:

```bash
export YTDLP_COOKIES_FILE=/data/cookies.txt   # path on the volume
```

Keep `cookies.txt` out of git (already ignored). It is a live account session:
anyone with it can act as that account. On Fly, prefer
`fly secrets set` over baking it into the image.

Also keep yt-dlp current — YouTube breaks older extractors often:

```bash
pip install -U yt-dlp
```

## Deploying the API to Fly.io

```bash
cd backend
fly auth login
fly launch --no-deploy --copy-config
fly volumes create skimreel_data --size 10 --region ord
fly secrets set CORS_ALLOW_ORIGINS=https://<your-vercel-domain>
fly deploy
curl https://<your-app>.fly.dev/api/health
```

`/api/health` reports whether ffmpeg is present, cookies are configured, and the
download directory is writable — the three things that break silently otherwise.
`cookies` is true only when yt-dlp will really be given a cookie file that it
accepts: an env var pointing at a missing, empty, or non-Netscape file reports
`false`, and `cookie_source` names which of the three forms was found or why
none was.

```bash
fly secrets set YTDLP_COOKIES="$(cat cookies.txt)"   # overrides YTDLP_COOKIES_FILE
```

Scale by adding machines, not workers: `JOBS` is in-process state, so a second
uvicorn worker would not see jobs started by the first.

## Pointing the frontend at the API

Vercel serves only the SPA — there is no API there. The browser has to be told
where the backend lives, and forgetting this is the single most common reason a
deployment fails while local works: the request goes to Vercel, gets a `404` with
the SPA's HTML, and reports a failure.

Two ways to set it, in precedence order:

### 1. `frontend/public/config.js` (usual path)

```js
window.__SKIMREEL_CONFIG__ = { apiBase: "https://skimreel-api.fly.dev" }
```

Committed to the repo and read at page load, so changing the backend is one
line and a push — no rebuild, no dashboard. It holds a public URL, not a secret.

### 2. `VITE_API_BASE` (Vercel project env var)

Set in Vercel → Settings → Environment Variables, for **Production and Preview**
(Production only leaves preview deploys broken). This wins over `config.js` when
present.

Vite inlines it at **build** time, so saving it does nothing until you redeploy.
Skip a redeploy and you get the same failure as having no API host at all.

### The value's format

Origin only: no trailing slash, no `/api` suffix. `https://skimreel-api.fly.dev`.

### Verify

```bash
curl -i -X POST https://<your-vercel-domain>/api/analyze -F length=30
# A JSON error (400/422) means the API answered. An HTML 404 means no API host.
```

Or in the browser: DevTools → Network → Analyze. `/api/analyze` must show the
Fly domain as its Request URL.

### CORS runs the other way

The browser is on Vercel and calls out to Fly, so **Fly** must accept the Vercel
origin. That is `CORS_ALLOW_ORIGINS` in `backend/fly.toml`, which is already set to
`https://skimreel.vercel.app`. Miss it and the request is blocked with no server
log to explain why.

Local file uploads are capped by the host's request size limit; Vercel and Fly
both sit well below a typical video file, so large uploads will fail in
production even though they work locally.

## Configuration

See `backend/.env.example` (API) and `frontend/.env.example` (SPA).
