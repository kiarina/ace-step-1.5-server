# ACE-Step 1.5 Music Generation Server

A **FastAPI** server for [ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5) music generation, optimized for **Apple Silicon (Mac Studio M4 Max)**.

- Accepts concurrent requests, processes one at a time via an internal queue
- Each request gets a job ID for async polling and file download
- Jobs are in-memory only — cleared on server restart

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
uv run uvicorn main:app --host 0.0.0.0 --port 8000
```

The server pre-loads `xl-base` and the LLM at startup (~30s). Interactive docs available at http://localhost:8000/docs.

---

## API

### Submit a job

```
POST /jobs
```

```json
{
  "prompt": "Modern J-Pop, 132 BPM, bright piano, upbeat drums",
  "lyrics": "[Verse 1]\n光の中へ\n[Chorus]\n僕らは走り続ける",
  "model": "xl-base",
  "duration": 30,
  "lang": "ja",
  "seed": -1
}
```

Response `202`:

```json
{
  "id": "b3f1a2c4-...",
  "status": "queued",
  "position": 1
}
```

### Poll status

```
GET /jobs/{id}
```

```json
{
  "id": "b3f1a2c4-...",
  "status": "done",
  "position": 0,
  "created_at": 1700000000.0,
  "started_at": 1700000001.0,
  "completed_at": 1700000026.0,
  "duration_sec": 25.1,
  "output_path": "/path/to/outputs/xxx.wav",
  "error": null,
  "request": { ... }
}
```

`status` values: `queued` → `running` → `done` / `failed`

### Download WAV

```
GET /jobs/{id}/download
```

Returns the WAV file when `status == "done"`. Returns `409` if not done yet, `422` if failed.

### Other endpoints

| Endpoint | Description |
|---|---|
| `GET /jobs` | List all in-memory jobs |
| `GET /health` | Queue depth, loaded models |
| `GET /help` | LLM-friendly full reference (prompts, lyrics, params) |
| `GET /docs` | Interactive Swagger UI |

---

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `prompt` | J-Pop preset | Music style description — genre, tempo, instruments, mood |
| `lyrics` | `[Instrumental]` | Lyrics with `[Section]` tags. `[Instrumental]` = no vocals |
| `model` | `xl-base` | `xl-base` (quality) or `turbo` (speed) |
| `duration` | `30` | Output length in seconds (5–300) |
| `lang` | `ja` | Vocal language: `ja` / `en` / `ko` / `zh` / `unknown` |
| `seed` | `-1` | Random seed. `-1` = random. Same seed → same output |
| `inference_steps` | *(preset)* | Override steps. More = higher quality, slower |
| `guidance_scale` | *(preset)* | CFG strength (xl-base only) |
| `shift` | `3.0` | Timestep shift. **Do not change from 3.0** unless experimenting |

### Models

| Model | Steps | Quality | ~Speed (30s audio, M4 Max) |
|---|---|---|---|
| `xl-base` | 32 | ★★★★★ | ~25s |
| `turbo` | 8 | ★★★☆☆ | ~4s |

---

## Usage example

```bash
# Submit a job
JOB=$(curl -s -X POST http://localhost:8000/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Upbeat anime opening, orchestral brass, driving rock drums, powerful male vocal",
    "lyrics": "[Verse 1]\n夢を追いかけて\n[Chorus]\n諦めないで走り続ける",
    "model": "xl-base",
    "duration": 60,
    "lang": "ja"
  }' | jq -r .id)

echo "Job ID: $JOB"

# Poll until done
while true; do
  STATUS=$(curl -s http://localhost:8000/jobs/$JOB | jq -r .status)
  echo "Status: $STATUS"
  [ "$STATUS" = "done" ] && break
  [ "$STATUS" = "failed" ] && break
  sleep 5
done

# Download
curl -o song.wav http://localhost:8000/jobs/$JOB/download
```

---

## Tips

See `GET /help` for a comprehensive LLM-friendly reference including:
- Prompt writing techniques (narrative > list style)
- Lyrics section tags and modifiers
- Language codes
- Reproducibility with seeds
