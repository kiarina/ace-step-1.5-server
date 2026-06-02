# ACE-Step 1.5 Music Generation Server

A **FastAPI** server for [ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5) music generation, optimized for **Apple Silicon (Mac Studio M4 Max)**.

- Accepts concurrent requests, processes one at a time via an internal queue
- Each request gets a job ID for async polling
- All files (uploads and generated output) managed via `/files` API with `file_id`
- Jobs and files are in-memory only — cleared on server restart

---

## Setup

> **Python 3.12 required.** Python 3.13 is not yet supported by ACE-Step dependencies.

### 1. Install dependencies

```bash
uv sync
```

### 2. Download checkpoints

```bash
# LLM (required for all generation)
# The 1.7B LLM is bundled inside the Ace-Step1.5 repo — download only that subfolder:
uv run hf download ACE-Step/Ace-Step1.5 \
  --include "acestep-5Hz-lm-1.7B/*" \
  --local-dir ./checkpoints

# DiT model — choose one or both:

# xl-base: highest quality (~19GB)
uv run hf download ACE-Step/acestep-v15-xl-base \
  --local-dir ./checkpoints/acestep-v15-xl-base

# turbo: fastest (~9GB) — also contains the LLM above, so one command covers both
uv run hf download ACE-Step/Ace-Step1.5 \
  --local-dir ./checkpoints/acestep-v15-turbo
```

### 3. Start the server

```bash
export PORT=8000  # change if the port is taken
uv run uvicorn main:app --host 0.0.0.0 --port $PORT
```

The server pre-loads `xl-base` and the LLM at startup (~30s). Interactive docs available at http://localhost:$PORT/docs.

---

## Endpoints

### Files

| Method | Path | Description |
|---|---|---|
| `POST` | `/files` | Upload an audio file → returns `file_id` |
| `GET` | `/files` | List all files |
| `GET` | `/files/{file_id}` | Get file metadata |
| `GET` | `/files/{file_id}/download` | Download WAV |
| `DELETE` | `/files/{file_id}` | Delete a file |

### Jobs

| Method | Path | Description |
|---|---|---|
| `POST` | `/jobs/text2music` | Generate music from text and lyrics |
| `POST` | `/jobs/cover` | Re-style an existing audio file |
| `POST` | `/jobs/repaint` | Edit a specific time range of an audio file |
| `POST` | `/jobs/extract` | Separate audio into stems |
| `GET` | `/jobs` | List all in-memory jobs |
| `GET` | `/jobs/{id}` | Get job status and metadata |

All generation endpoints return a job ID immediately (`202 Accepted`).
Poll `GET /jobs/{id}` until `status == "done"`, then download with `GET /files/{file_id}/download`.

### Server

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Server health check |
| `GET` | `/help` | LLM-friendly full reference |
| `GET` | `/docs` | Interactive Swagger UI |

---

## Usage examples

### text2music — Generate from text and lyrics

```bash
export PORT=8000

# Submit
JOB=$(curl -s -X POST http://localhost:$PORT/jobs/text2music \
  -H "Content-Type: application/json" \
  --data-binary @- <<'EOF' | jq -r .id
{
  "prompt": "Modern J-Pop, 132 BPM, bright piano, emotional electric guitar, upbeat drums, polished production",
  "lyrics": "[Intro]\n\n[Verse 1]\n加速する世界の中で\n君の声が聴こえてくる\n揺れる心抱えながら\n一歩ずつ前を向いて\n\n[Chorus]\n僕らは光を追いかける\n終わらない夢の向こうへ\n諦めないで走り続ける\nこの手を離さないで\n\n[Outro]\n光の中へ",
  "model": "xl-base",
  "duration": 60,
  "lang": "ja",
  "seed": 1
}
EOF
)

# Poll
until [ "$(curl -s http://localhost:$PORT/jobs/$JOB | jq -r .status)" = "done" ] || \
      [ "$(curl -s http://localhost:$PORT/jobs/$JOB | jq -r .status)" = "failed" ]; do
  echo "status: $(curl -s http://localhost:$PORT/jobs/$JOB | jq -r .status)"
  sleep 5
done

# Get file_id and download
FILE_ID=$(curl -s http://localhost:$PORT/jobs/$JOB | jq -r .file_id)
curl -o song.wav http://localhost:$PORT/files/$FILE_ID/download
```

---

### cover — Re-style an existing audio file

Transform the style of a song while preserving its musical structure.
Upload the source file first, then pass its `file_id` as `src`.

```bash
export PORT=8000

# Upload source file
SRC_ID=$(curl -s -X POST http://localhost:$PORT/files \
  -F "file=@/path/to/song.wav" | jq -r .id)

# Submit cover job
JOB=$(curl -s -X POST http://localhost:$PORT/jobs/cover \
  -H "Content-Type: application/json" \
  --data-binary @- <<EOF | jq -r .id
{
  "src": "$SRC_ID",
  "prompt": "City Pop, groovy bass, smooth guitar, laid-back drums, polished 80s production",
  "strength": 0.7,
  "model": "xl-base",
  "seed": 1
}
EOF
)

until [ "$(curl -s http://localhost:$PORT/jobs/$JOB | jq -r .status)" = "done" ] || \
      [ "$(curl -s http://localhost:$PORT/jobs/$JOB | jq -r .status)" = "failed" ]; do
  echo "status: $(curl -s http://localhost:$PORT/jobs/$JOB | jq -r .status)"
  sleep 5
done

FILE_ID=$(curl -s http://localhost:$PORT/jobs/$JOB | jq -r .file_id)
curl -o cover.wav http://localhost:$PORT/files/$FILE_ID/download
```

**`strength`**: how closely the output follows the source structure.
- `0.3` — creative reimagining, loosely based on the original
- `0.7` — (default) preserves structure, changes style
- `1.0` — strict adherence to the original

> **Tip:** Cover works best for **similar genre transfers** (e.g. J-Pop → City Pop, Rock → Blues Rock).
> Dramatic genre changes (e.g. J-Pop → Acoustic Folk) tend to produce unstable results.

---

### repaint — Edit a specific time range

Regenerate a section of an existing audio file with a new style.

```bash
export PORT=8000

# Upload source file
SRC_ID=$(curl -s -X POST http://localhost:$PORT/files \
  -F "file=@/path/to/song.wav" | jq -r .id)

JOB=$(curl -s -X POST http://localhost:$PORT/jobs/repaint \
  -H "Content-Type: application/json" \
  --data-binary @- <<EOF | jq -r .id
{
  "src": "$SRC_ID",
  "prompt": "Dramatic orchestral strings, emotional swell, cinematic",
  "start": 15,
  "end": 30,
  "strength": 0.6,
  "model": "xl-base",
  "seed": 1
}
EOF
)

until [ "$(curl -s http://localhost:$PORT/jobs/$JOB | jq -r .status)" = "done" ] || \
      [ "$(curl -s http://localhost:$PORT/jobs/$JOB | jq -r .status)" = "failed" ]; do
  echo "status: $(curl -s http://localhost:$PORT/jobs/$JOB | jq -r .status)"
  sleep 5
done

FILE_ID=$(curl -s http://localhost:$PORT/jobs/$JOB | jq -r .file_id)
curl -o repainted.wav http://localhost:$PORT/files/$FILE_ID/download
```

- `start` / `end`: time range in seconds. `end: -1` means until end of file.
- `strength`: repaint intensity. `0.5` (default) blends well with surrounding sections.

---

### extract — Separate audio into stems

Each target stem is a **separate job**. The response contains a list of IDs.

```bash
export PORT=8000

# Upload source file
SRC_ID=$(curl -s -X POST http://localhost:$PORT/files \
  -F "file=@/path/to/song.wav" | jq -r .id)

RESP=$(curl -s -X POST http://localhost:$PORT/jobs/extract \
  -H "Content-Type: application/json" \
  --data-binary @- <<EOF
{
  "src": "$SRC_ID",
  "targets": ["vocals", "drums", "bass", "other"],
  "model": "xl-base"
}
EOF
)

echo "$RESP" | jq .
# { "ids": ["aaa...", "bbb...", "ccc...", "ddd..."], "targets": [...], "status": "queued" }

# Poll and download each stem
for ID in $(echo "$RESP" | jq -r '.ids[]'); do
  TARGET=$(curl -s http://localhost:$PORT/jobs/$ID | jq -r '.request.targets[0]')
  until [ "$(curl -s http://localhost:$PORT/jobs/$ID | jq -r .status)" = "done" ] || \
        [ "$(curl -s http://localhost:$PORT/jobs/$ID | jq -r .status)" = "failed" ]; do
    echo "[$TARGET] status: $(curl -s http://localhost:$PORT/jobs/$ID | jq -r .status)"
    sleep 5
  done
  FILE_ID=$(curl -s http://localhost:$PORT/jobs/$ID | jq -r .file_id)
  curl -o "${TARGET}.wav" http://localhost:$PORT/files/$FILE_ID/download
  echo "Saved ${TARGET}.wav"
done
```

Available targets: `vocals`, `drums`, `bass`, `other` (and any stem name supported by ACE-Step).

---

## Parameters

### Common (all tasks)

| Parameter | Default | Description |
|---|---|---|
| `model` | `xl-base` | `xl-base` (quality) or `turbo` (speed) |
| `seed` | `-1` | Random seed. `-1` = random. Same seed → same output |
| `inference_steps` | *(preset)* | Override steps. More = higher quality, slower |
| `guidance_scale` | *(preset)* | CFG strength (xl-base only) |
| `shift` | `3.0` | Timestep shift. **Do not change from 3.0** unless experimenting |

### text2music

| Parameter | Default | Description |
|---|---|---|
| `prompt` | J-Pop preset | Music style description — genre, tempo, instruments, mood |
| `lyrics` | `[Instrumental]` | Lyrics with `[Section]` tags. Match length to `duration` |
| `duration` | `60` | Output length in seconds (5–300) |
| `lang` | `ja` | Vocal language: `ja` / `en` / `ko` / `zh` / `unknown` |

### cover

| Parameter | Default | Description |
|---|---|---|
| `src` | — | `file_id` of the source WAV |
| `prompt` | — | Target style description |
| `strength` | `0.7` | Source adherence (0.0–1.0) |
| `duration` | `null` | Output length. Defaults to source length |

### repaint

| Parameter | Default | Description |
|---|---|---|
| `src` | — | `file_id` of the source WAV |
| `prompt` | — | Style for the repainted section |
| `start` | — | Start time in seconds |
| `end` | `-1` | End time in seconds. `-1` = until end of file |
| `strength` | `0.5` | Repaint intensity (0.0–1.0) |

### extract

| Parameter | Default | Description |
|---|---|---|
| `src` | — | `file_id` of the source WAV |
| `targets` | `["vocals","drums","bass","other"]` | Stems to extract |

### Models

| Model | Steps | Quality | ~Speed (30s audio, M4 Max) |
|---|---|---|---|
| `xl-base` | 32 | ★★★★★ | ~25s |
| `turbo` | 8 | ★★★☆☆ | ~4s |

---

## Prompt tips

Describe the **sound**, not the story. Narrative prompts work better than keyword lists.

```
✗  piano, sad, female, slow

✓  A melancholic piano ballad where soft female vocals weave through gentle string
   accompaniment, creating an intimate and heartbreaking atmosphere. 80 BPM.
```

Key elements: genre · tempo/BPM · instruments · vocal style · mood · production style

---

## Lyrics format

Structure lyrics with `[Square Bracket]` section tags at the start of a line.

```
[Intro]          Opening — instrumental or atmospheric
[Verse 1]        First verse
[Pre-Chorus]     Build-up before the hook
[Chorus]         The hook — emotionally strongest part
[Bridge]         Contrasting section
[Instrumental]   No vocals
[Outro]          Closing
```

Modifiers: `[Verse 1 - Female]`, `[Chorus - Both]`, `[Bridge - Whispered]`, `[Outro - Fade out]`

> **Tip:** Match lyrics length to `duration`. Too few lines for a long duration causes the model to lose coherence in the second half.

See `GET /help` for a complete LLM-friendly reference.
