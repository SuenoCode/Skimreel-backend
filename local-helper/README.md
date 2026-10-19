# SkimReel local YouTube helper

This helper keeps YouTube access on the Windows computer where Firefox is
signed in. It downloads the video locally, then uploads the video file to the
existing SkimReel Render API. Firefox cookies are never sent to Render.

Close Firefox before running it so yt-dlp can read the cookie database.

From `F:\apps\Forge`, run:

```cmd
local-helper\run_youtube.bat "https://www.youtube.com/watch?v=6KjRIoQtN2w"
```

Optional settings:

```cmd
local-helper\run_youtube.bat "URL" --length 30 --count 8 --quality 720 --keep
```

The helper uses the default Firefox profile from `profiles.ini`, Deno for
YouTube's JavaScript challenges, and the existing `/api/analyze` upload path.
