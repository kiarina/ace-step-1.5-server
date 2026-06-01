"""
ACE-Step 1.5 Music Generation Server

A FastAPI server that accepts music generation requests, queues them, and
processes them one at a time using ACE-Step 1.5.

Endpoints:
  POST /jobs                  Submit a generation job → returns job ID
  GET  /jobs                  List all in-memory jobs
  GET  /jobs/{id}             Get job status and metadata
  GET  /jobs/{id}/download    Download the generated WAV file
  GET  /health                Server health check
  GET  /help                  LLM-friendly API reference
"""

import asyncio
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from loguru import logger
from pydantic import BaseModel, Field

from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
from acestep.inference import GenerationParams, GenerationConfig, generate_music


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
LLM_MODEL = "acestep-5Hz-lm-1.7B"


# ---------------------------------------------------------------------------
# Model presets
# ---------------------------------------------------------------------------

@dataclass
class ModelPreset:
    config_path: str
    inference_steps: int
    guidance_scale: float
    shift: float
    use_adg: bool
    dcw_enabled: bool


PRESETS: dict[str, ModelPreset] = {
    "turbo": ModelPreset(
        config_path="acestep-v15-turbo",
        inference_steps=8,
        guidance_scale=1.0,
        shift=3.0,
        use_adg=False,
        dcw_enabled=True,
    ),
    "xl-base": ModelPreset(
        config_path="acestep-v15-xl-base",
        inference_steps=32,
        guidance_scale=7.0,
        shift=3.0,
        use_adg=False,
        dcw_enabled=False,
    ),
}


# ---------------------------------------------------------------------------
# Job state
# ---------------------------------------------------------------------------

class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    done = "done"
    failed = "failed"


@dataclass
class Job:
    id: str
    request: "GenerateRequest"
    status: JobStatus = JobStatus.queued
    position: int = 0          # queue position (1-based; 0 when running/done)
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    output_path: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "status": self.status,
            "position": self.position,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_sec": (
                round(self.completed_at - self.started_at, 2)
                if self.started_at and self.completed_at else None
            ),
            "output_path": self.output_path,
            "error": self.error,
            "request": self.request.model_dump(),
        }


# In-memory store
jobs: dict[str, Job] = {}
job_queue: asyncio.Queue[Job] = asyncio.Queue()


# ---------------------------------------------------------------------------
# Request schema
# ---------------------------------------------------------------------------

class GenerateRequest(BaseModel):
    prompt: str = Field(
        default="Modern J-Pop, 132 BPM, bright piano, emotional electric guitar, upbeat drums",
        description=(
            "Music style description. Describe the SOUND itself — genre, tempo, instruments, mood, "
            "production style. Do NOT describe lyric content here."
        ),
        examples=["Acoustic jazz trio, brushed drums, walking bass, warm female vocal, intimate"],
    )
    lyrics: str = Field(
        default="[Instrumental]",
        description=(
            "Song lyrics with section tags: [Verse 1], [Chorus], [Bridge], [Outro], etc. "
            "Use '[Instrumental]' for no vocals."
        ),
        examples=["[Verse 1]\n静かな夜に\n[Chorus]\n光の中へ"],
    )
    model: str = Field(
        default="xl-base",
        description="Model preset. 'xl-base' = highest quality (32 steps). 'turbo' = fastest (8 steps).",
        pattern="^(xl-base|turbo)$",
    )
    duration: int = Field(
        default=30,
        ge=5,
        le=300,
        description="Output duration in seconds.",
    )
    lang: str = Field(
        default="ja",
        description="Vocal language code (ISO 639-1). e.g. 'ja', 'en', 'ko', 'zh'. Must match the lyrics language.",
    )
    seed: int = Field(
        default=-1,
        description="Random seed. -1 = random. Use the same seed to reproduce a generation exactly.",
    )
    # Advanced overrides (optional)
    inference_steps: Optional[int] = Field(
        default=None,
        description="Override model preset's inference steps.",
    )
    guidance_scale: Optional[float] = Field(
        default=None,
        description="Override CFG guidance scale (xl-base only; turbo ignores this).",
    )
    shift: Optional[float] = Field(
        default=None,
        description="Override timestep shift. 3.0 is strongly recommended; do not change unless experimenting.",
    )


# ---------------------------------------------------------------------------
# Model handlers (loaded lazily, cached after first load)
# ---------------------------------------------------------------------------

_dit_handlers: dict[str, AceStepHandler] = {}
_llm_handler: Optional[LLMHandler] = None


def get_dit_handler(preset: ModelPreset) -> AceStepHandler:
    if preset.config_path not in _dit_handlers:
        logger.info(f"Loading DiT model: {preset.config_path}")
        handler = AceStepHandler()
        msg, ok = handler.initialize_service(
            project_root=str(PROJECT_ROOT),
            config_path=preset.config_path,
            device="auto",
            offload_to_cpu=False,
        )
        if not ok:
            raise RuntimeError(f"DiT init failed: {msg}")
        _dit_handlers[preset.config_path] = handler
        logger.info(f"DiT loaded: {preset.config_path}")
    return _dit_handlers[preset.config_path]


def init_llm() -> LLMHandler:
    global _llm_handler
    if _llm_handler is None:
        logger.info(f"Loading LLM: {LLM_MODEL}")
        handler = LLMHandler()
        msg, ok = handler.initialize(
            checkpoint_dir=str(CHECKPOINT_DIR),
            lm_model_path=LLM_MODEL,
            backend="mlx",
            device="auto",
            offload_to_cpu=False,
            dtype=None,
        )
        if not ok:
            raise RuntimeError(f"LLM init failed: {msg}")
        _llm_handler = handler
        logger.info("LLM loaded")
    return _llm_handler


# ---------------------------------------------------------------------------
# Queue worker
# ---------------------------------------------------------------------------

def _update_queue_positions() -> None:
    """Recalculate queue positions for all queued jobs."""
    pos = 1
    for job in jobs.values():
        if job.status == JobStatus.queued:
            job.position = pos
            pos += 1


async def worker() -> None:
    """Single background worker — processes one job at a time."""
    logger.info("Queue worker started")
    while True:
        job = await job_queue.get()
        req = job.request
        preset = PRESETS[req.model]

        job.status = JobStatus.running
        job.position = 0
        job.started_at = time.time()
        _update_queue_positions()
        logger.info(f"[{job.id[:8]}] Starting (model={req.model}, duration={req.duration}s, seed={req.seed})")

        try:
            dit = get_dit_handler(preset)
            llm = init_llm()

            OUTPUT_DIR.mkdir(exist_ok=True)
            params = GenerationParams(
                task_type="text2music",
                thinking=True,
                caption=req.prompt,
                lyrics=req.lyrics,
                vocal_language=req.lang,
                duration=req.duration,
                inference_steps=req.inference_steps or preset.inference_steps,
                guidance_scale=req.guidance_scale if req.guidance_scale is not None else preset.guidance_scale,
                shift=req.shift if req.shift is not None else preset.shift,
                use_adg=preset.use_adg,
                dcw_enabled=preset.dcw_enabled,
                seed=req.seed,
            )
            config = GenerationConfig(batch_size=1, audio_format="wav")

            # Run sync generation in a thread pool so the event loop stays responsive
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: generate_music(dit, llm, params=params, config=config, save_dir=str(OUTPUT_DIR)),
            )

            if not result.success:
                raise RuntimeError(result.status_message)

            path = result.audios[0].get("path", "")
            job.output_path = path
            job.status = JobStatus.done
            job.completed_at = time.time()
            elapsed = job.completed_at - job.started_at
            logger.info(f"[{job.id[:8]}] Done in {elapsed:.1f}s → {path}")

        except Exception as exc:
            job.status = JobStatus.failed
            job.completed_at = time.time()
            job.error = str(exc)
            logger.error(f"[{job.id[:8]}] Failed: {exc}")

        finally:
            job_queue.task_done()
            _update_queue_positions()


# ---------------------------------------------------------------------------
# Lifespan: pre-load default model at startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        init_llm()
        get_dit_handler(PRESETS["xl-base"])
    except Exception as exc:
        logger.warning(f"Startup model preload failed (will retry on first request): {exc}")

    worker_task = asyncio.create_task(worker())
    yield
    worker_task.cancel()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="ACE-Step 1.5 Music Generation Server",
    description=__doc__,
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/jobs", status_code=202, summary="Submit a generation job")
async def submit_job(request: GenerateRequest) -> dict:
    """
    Submit a music generation request. Returns a job ID immediately.
    Poll `GET /jobs/{id}` for status. Download with `GET /jobs/{id}/download` when done.
    """
    job_id = str(uuid.uuid4())
    job = Job(id=job_id, request=request)
    jobs[job_id] = job
    await job_queue.put(job)
    _update_queue_positions()

    return {
        "id": job_id,
        "status": job.status,
        "position": job.position,
    }


@app.get("/jobs", summary="List all jobs")
async def list_jobs() -> list[dict]:
    """Returns all in-memory jobs. Cleared on server restart."""
    return [j.to_dict() for j in jobs.values()]


@app.get("/jobs/{job_id}", summary="Get job status")
async def get_job(job_id: str) -> dict:
    """Returns status, metadata, and timing for the given job ID."""
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return job.to_dict()


@app.get("/jobs/{job_id}/download", summary="Download generated WAV")
async def download_job(job_id: str) -> FileResponse:
    """
    Download the generated WAV file.
    - 404: job not found
    - 409: job not done yet (queued or running)
    - 422: job failed
    """
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    if job.status == JobStatus.failed:
        raise HTTPException(status_code=422, detail=f"Job failed: {job.error}")
    if job.status != JobStatus.done:
        raise HTTPException(status_code=409, detail=f"Job is '{job.status.value}', not done yet")
    if not job.output_path or not Path(job.output_path).exists():
        raise HTTPException(status_code=500, detail="Output file missing on server")

    filename = f"acestep_{job_id[:8]}.wav"
    return FileResponse(job.output_path, media_type="audio/wav", filename=filename)


@app.get("/health", summary="Health check")
async def health() -> dict:
    """Returns server health, queue depth, and loaded model info."""
    return {
        "status": "ok",
        "models_loaded": list(_dit_handlers.keys()),
        "llm_loaded": _llm_handler is not None,
        "queue_size": job_queue.qsize(),
        "running": sum(1 for j in jobs.values() if j.status == JobStatus.running),
        "total_jobs": len(jobs),
    }


@app.get("/help", summary="LLM-friendly API reference and tips")
async def help_endpoint() -> JSONResponse:
    """
    Comprehensive reference for LLMs and developers:
    endpoint descriptions, parameter tips, prompt writing guide, and lyrics format.
    """
    content = {
        "overview": (
            "ACE-Step 1.5 Music Generation Server. "
            "Submit jobs via POST /jobs, poll status via GET /jobs/{id}, "
            "download audio via GET /jobs/{id}/download. "
            "Requests are accepted concurrently but processed one at a time via an internal queue."
        ),
        "quick_start": [
            "1. POST /jobs  with {prompt, lyrics, model, duration}  →  get {id}",
            "2. GET /jobs/{id}  until status == 'done'",
            "3. GET /jobs/{id}/download  →  WAV file",
        ],
        "endpoints": {
            "POST /jobs": "Submit a job. Body: GenerateRequest. Returns {id, status, position}.",
            "GET /jobs": "List all in-memory jobs (cleared on restart).",
            "GET /jobs/{id}": "Get status, timing, and request params for a job.",
            "GET /jobs/{id}/download": "Download WAV. 409 if not done, 422 if failed.",
            "GET /health": "Health, queue depth, loaded models.",
            "GET /help": "This document.",
        },
        "models": {
            "xl-base": (
                "Highest quality. 32 inference steps, CFG guidance_scale=7.0. "
                "~25s generation time for 30s audio on M4 Max."
            ),
            "turbo": (
                "Fastest. 8 inference steps, no CFG. "
                "~4s generation time for 15s audio on M4 Max. Good for prototyping."
            ),
        },
        "prompt_tips": {
            "rule": "Describe the SOUND, not the story. Narrative prompts work better than lists.",
            "elements": {
                "genre": "Modern J-Pop / Acoustic Jazz / Dark Trap / Lo-fi Hip-hop / Anime OST",
                "tempo": "132 BPM / slow and intimate / driving beat / half-time feel",
                "instruments": "bright piano / smooth saxophone / 808 sub-bass / nylon guitar / brass section",
                "vocal_style": "emotional female vocal / whispered delivery / powerful male tenor / rap verse",
                "mood": "melancholic / triumphant / cozy / aggressive / dreamy / nostalgic",
                "production": "polished radio-ready / lo-fi vinyl texture / orchestral / live band recording",
            },
            "good_example": (
                "A melancholic piano ballad where soft female vocals weave through gentle string "
                "accompaniment, creating an intimate and heartbreaking atmosphere. 80 BPM."
            ),
            "bad_example": "piano, sad, female, slow",
            "avoid": "Don't mix contradictory instructions (e.g. 'slow and fast'). Don't describe lyric content in prompt.",
        },
        "lyrics_format": {
            "rule": "Use [Square Bracket] section tags at the start of a line to structure the song.",
            "tags": {
                "[Intro]": "Opening, often instrumental or atmospheric",
                "[Verse 1]": "First verse",
                "[Verse 2]": "Second verse (same melody, different lyrics)",
                "[Pre-Chorus]": "Build-up before chorus",
                "[Chorus]": "The hook — emotionally strongest part",
                "[Bridge]": "Contrasting section, different chord feel",
                "[Instrumental]": "No vocals (guitar solo, etc.)",
                "[Outro]": "Closing section",
            },
            "modifiers": (
                "Add style hints after a dash: [Verse 1 - Female], [Chorus - Both], "
                "[Bridge - Whispered], [Outro - Fade out], [Instrumental - Guitar Solo]"
            ),
            "no_vocals": "Set lyrics to '[Instrumental]' for a fully instrumental output.",
            "line_length": "Aim for 6–10 syllables per line for natural phrasing.",
            "emphasis": "ALL CAPS for shouts or strong emphasis (e.g. 'I WILL NEVER LOOK BACK').",
            "backing_vocals": "Use parentheses: '光を追いかける（Yeah, yeah）'",
        },
        "language_codes": {
            "ja": "Japanese",
            "en": "English",
            "ko": "Korean",
            "zh": "Mandarin Chinese",
            "yue": "Cantonese",
            "es": "Spanish",
            "fr": "French",
            "de": "German",
            "unknown": "Auto-detect (may reduce quality)",
        },
        "reproducibility": {
            "tip": "Set 'seed' to any integer to reproduce a generation. seed=-1 randomizes each time.",
            "note": "Same seed + same params + same model = identical output.",
        },
        "advanced_params": {
            "shift": "Timestep schedule parameter. 3.0 is the correct value; 1.0 causes noisy output.",
            "inference_steps": "More steps = higher quality but slower. Overrides the model preset.",
            "guidance_scale": "CFG strength (xl-base only). Higher = more prompt-adherent but less natural.",
        },
    }
    return JSONResponse(content=content)
