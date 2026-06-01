"""
ACE-Step 1.5 Music Generation Server

A FastAPI server that accepts music generation requests, queues them, and
processes them one at a time using ACE-Step 1.5.

Endpoints:
  POST /jobs/text2music           Generate music from text and lyrics
  POST /jobs/cover                Re-style an existing audio file
  POST /jobs/repaint              Edit a specific time range of an audio file
  POST /jobs/extract              Separate an audio file into stems

  GET  /jobs                      List all in-memory jobs
  GET  /jobs/{id}                 Get job status and metadata
  GET  /jobs/{id}/download        Download the generated WAV file
  GET  /health                    Server health check
  GET  /help                      LLM-friendly API reference
"""

import asyncio
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Annotated, Optional, Union

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


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------

class _ModelMixin(BaseModel):
    model: str = Field(
        default="xl-base",
        description="Model preset. 'xl-base' = highest quality (32 steps). 'turbo' = fastest (8 steps).",
        pattern="^(xl-base|turbo)$",
    )
    seed: int = Field(
        default=-1,
        description="Random seed. -1 = random. Same seed → same output.",
    )
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
        description="Override timestep shift. 3.0 is strongly recommended.",
    )


class Text2MusicRequest(_ModelMixin):
    task: str = Field(default="text2music", frozen=True, exclude=True)
    prompt: str = Field(
        default="Modern J-Pop, 132 BPM, bright piano, emotional electric guitar, upbeat drums",
        description=(
            "Music style description. Describe the SOUND — genre, tempo, instruments, mood, "
            "production style. Do NOT describe lyric content here."
        ),
    )
    lyrics: str = Field(
        default="[Instrumental]",
        description=(
            "Song lyrics with section tags: [Verse 1], [Chorus], [Bridge], [Outro], etc. "
            "Use '[Instrumental]' for no vocals. "
            "Match lyrics length to duration — too few lines for a long duration causes quality issues."
        ),
    )
    duration: int = Field(default=60, ge=5, le=300, description="Output duration in seconds.")
    lang: str = Field(
        default="ja",
        description="Vocal language code (ISO 639-1): 'ja', 'en', 'ko', 'zh', 'unknown', etc.",
    )


class CoverRequest(_ModelMixin):
    task: str = Field(default="cover", frozen=True, exclude=True)
    src: str = Field(description="Absolute path to the source audio file on the server.")
    prompt: str = Field(description="Target style description for the cover version.")
    strength: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="How closely to follow the source structure. 0.0 = free, 1.0 = strict.",
    )
    duration: Optional[int] = Field(
        default=None,
        ge=5,
        le=300,
        description="Output duration in seconds. Defaults to source audio length.",
    )


class RepaintRequest(_ModelMixin):
    task: str = Field(default="repaint", frozen=True, exclude=True)
    src: str = Field(description="Absolute path to the source audio file on the server.")
    prompt: str = Field(description="Style description for the repainted section.")
    start: float = Field(description="Start time of the section to repaint, in seconds.")
    end: float = Field(
        default=-1,
        description="End time of the section to repaint, in seconds. -1 = until end of file.",
    )
    strength: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Repaint strength. 0.0 = subtle, 1.0 = aggressive.",
    )


class ExtractRequest(_ModelMixin):
    task: str = Field(default="extract", frozen=True, exclude=True)
    src: str = Field(description="Absolute path to the source audio file on the server.")
    targets: list[str] = Field(
        default=["vocals", "drums", "bass", "other"],
        description="Stems to extract. Each target becomes a separate job result.",
    )


AnyRequest = Annotated[
    Union[Text2MusicRequest, CoverRequest, RepaintRequest, ExtractRequest],
    Field(discriminator=None),
]


# ---------------------------------------------------------------------------
# Job dataclass
# ---------------------------------------------------------------------------

@dataclass
class Job:
    id: str
    request: AnyRequest
    status: JobStatus = JobStatus.queued
    position: int = 0
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
# Build GenerationParams from request
# ---------------------------------------------------------------------------

def _build_params(req: AnyRequest) -> GenerationParams:
    preset = PRESETS[req.model]
    common = dict(
        inference_steps=req.inference_steps or preset.inference_steps,
        guidance_scale=req.guidance_scale if req.guidance_scale is not None else preset.guidance_scale,
        shift=req.shift if req.shift is not None else preset.shift,
        use_adg=preset.use_adg,
        dcw_enabled=preset.dcw_enabled,
        seed=req.seed,
    )

    if isinstance(req, Text2MusicRequest):
        return GenerationParams(
            task_type="text2music",
            thinking=True,
            caption=req.prompt,
            lyrics=req.lyrics,
            vocal_language=req.lang,
            duration=req.duration,
            **common,
        )
    elif isinstance(req, CoverRequest):
        return GenerationParams(
            task_type="cover",
            thinking=False,
            caption=req.prompt,
            lyrics="[Instrumental]",
            src_audio=req.src,
            audio_cover_strength=req.strength,
            duration=req.duration,
            **common,
        )
    elif isinstance(req, RepaintRequest):
        return GenerationParams(
            task_type="repaint",
            thinking=False,
            caption=req.prompt,
            lyrics="[Instrumental]",
            src_audio=req.src,
            repainting_start=req.start,
            repainting_end=req.end,
            repaint_strength=req.strength,
            **common,
        )
    elif isinstance(req, ExtractRequest):
        # targets must be a single-element list when going through the normal worker
        assert len(req.targets) == 1, "ExtractRequest in worker must have exactly one target"
        return GenerationParams(
            task_type="extract",
            thinking=False,
            caption=req.targets[0],
            lyrics="[Instrumental]",
            src_audio=req.src,
            **common,
        )
    else:
        raise ValueError(f"Unknown request type: {type(req)}")


# ---------------------------------------------------------------------------
# Queue worker
# ---------------------------------------------------------------------------

def _update_queue_positions() -> None:
    pos = 1
    for job in jobs.values():
        if job.status == JobStatus.queued:
            job.position = pos
            pos += 1


async def _run_job(job: Job) -> None:
    req = job.request
    preset = PRESETS[req.model]

    job.status = JobStatus.running
    job.position = 0
    job.started_at = time.time()
    _update_queue_positions()
    logger.info(f"[{job.id[:8]}] Starting task={req.model_dump().get('task', '?')} model={req.model}")

    try:
        dit = get_dit_handler(preset)
        llm = init_llm()
        OUTPUT_DIR.mkdir(exist_ok=True)

        params = _build_params(req)
        config = GenerationConfig(batch_size=1, audio_format="wav")

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
        logger.info(f"[{job.id[:8]}] Done in {job.completed_at - job.started_at:.1f}s → {path}")

    except Exception as exc:
        job.status = JobStatus.failed
        job.completed_at = time.time()
        job.error = str(exc)
        logger.error(f"[{job.id[:8]}] Failed: {exc}")


async def worker() -> None:
    """Single background worker — processes one job at a time."""
    logger.info("Queue worker started")
    while True:
        job = await job_queue.get()
        try:
            await _run_job(job)
        finally:
            job_queue.task_done()
            _update_queue_positions()


# ---------------------------------------------------------------------------
# Helper: enqueue a job
# ---------------------------------------------------------------------------

async def _enqueue(request: AnyRequest) -> dict:
    job_id = str(uuid.uuid4())
    job = Job(id=job_id, request=request)
    jobs[job_id] = job
    await job_queue.put(job)
    _update_queue_positions()
    return {"id": job_id, "status": job.status, "position": job.position}


# ---------------------------------------------------------------------------
# Lifespan
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
# Generation endpoints
# ---------------------------------------------------------------------------

@app.post("/jobs/text2music", status_code=202, summary="Generate music from text and lyrics")
async def submit_text2music(request: Text2MusicRequest) -> dict:
    """
    Generate a new song from a style prompt and lyrics.
    Returns a job ID immediately. Poll `GET /jobs/{id}` for status.
    """
    return await _enqueue(request)


@app.post("/jobs/cover", status_code=202, summary="Re-style an existing audio file")
async def submit_cover(request: CoverRequest) -> dict:
    """
    Transform the style of an existing audio file while preserving its structure.
    `src` must be an absolute path to a WAV file accessible on the server.
    `strength` controls how closely the output follows the source (0.0–1.0).
    """
    if not Path(request.src).exists():
        raise HTTPException(status_code=422, detail=f"src file not found: {request.src}")
    return await _enqueue(request)


@app.post("/jobs/repaint", status_code=202, summary="Edit a specific time range of an audio file")
async def submit_repaint(request: RepaintRequest) -> dict:
    """
    Regenerate a specific time range of an existing audio file with a new style.
    `start` and `end` are in seconds. `end=-1` means until the end of the file.
    """
    if not Path(request.src).exists():
        raise HTTPException(status_code=422, detail=f"src file not found: {request.src}")
    return await _enqueue(request)


@app.post("/jobs/extract", status_code=202, summary="Separate audio into stems")
async def submit_extract(request: ExtractRequest) -> dict:
    """
    Separate an audio file into individual stems (vocals, drums, bass, other, etc.).
    Each target stem is enqueued as a **separate job**. Returns a list of job IDs.
    Poll each ID individually via `GET /jobs/{id}`.
    """
    if not Path(request.src).exists():
        raise HTTPException(status_code=422, detail=f"src file not found: {request.src}")

    result_ids = []
    for target in request.targets:
        single = ExtractRequest(
            src=request.src,
            targets=[target],
            model=request.model,
            seed=request.seed,
            inference_steps=request.inference_steps,
            guidance_scale=request.guidance_scale,
            shift=request.shift,
        )
        job_id = str(uuid.uuid4())
        job = Job(id=job_id, request=single)
        jobs[job_id] = job
        await job_queue.put(job)
        result_ids.append(job_id)

    _update_queue_positions()
    return {
        "ids": result_ids,
        "targets": request.targets,
        "status": "queued",
        "note": "Each stem is a separate job. Poll GET /jobs/{id} for each.",
    }


# ---------------------------------------------------------------------------
# Job management endpoints
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Utility endpoints
# ---------------------------------------------------------------------------

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
            "Submit jobs via POST /jobs/<task>, poll via GET /jobs/{id}, "
            "download via GET /jobs/{id}/download. "
            "All requests are queued and processed one at a time."
        ),
        "quick_start": [
            "1. POST /jobs/text2music  with {prompt, lyrics, duration, model}  →  {id}",
            "2. GET /jobs/{id}  until status == 'done'",
            "3. GET /jobs/{id}/download  →  WAV file",
        ],
        "endpoints": {
            "POST /jobs/text2music": "Generate new music from prompt + lyrics.",
            "POST /jobs/cover": "Re-style an existing audio file. Requires 'src' (server path).",
            "POST /jobs/repaint": "Edit a time range of an existing audio file. Requires 'src', 'start', 'end'.",
            "POST /jobs/extract": "Separate audio into stems. Returns multiple job IDs, one per target.",
            "GET /jobs": "List all in-memory jobs (cleared on restart).",
            "GET /jobs/{id}": "Get status, timing, and request params for a job.",
            "GET /jobs/{id}/download": "Download WAV. 409 if not done, 422 if failed.",
            "GET /health": "Health, queue depth, loaded models.",
            "GET /help": "This document.",
        },
        "tasks": {
            "text2music": {
                "description": "Generate a new song from scratch.",
                "key_params": ["prompt", "lyrics", "duration", "lang", "seed"],
                "lyrics_tip": "Match lyrics length to duration. Too few lines for a long duration causes quality issues.",
            },
            "cover": {
                "description": "Transform the style of an existing audio file.",
                "key_params": ["src", "prompt", "strength"],
                "strength_tip": "strength=0.7 preserves structure well. Lower = more creative, higher = more faithful.",
            },
            "repaint": {
                "description": "Regenerate a specific time range with a new style.",
                "key_params": ["src", "prompt", "start", "end", "strength"],
                "note": "end=-1 means until the end of the file.",
            },
            "extract": {
                "description": "Separate audio into stems (vocals, drums, bass, other).",
                "key_params": ["src", "targets"],
                "note": "Each target becomes a separate job. Response contains a list of IDs.",
            },
        },
        "models": {
            "xl-base": "Highest quality. 32 steps, CFG guidance_scale=7.0. ~25s for 30s audio on M4 Max.",
            "turbo": "Fastest. 8 steps, no CFG. ~4s for 15s audio on M4 Max. Good for prototyping.",
        },
        "prompt_tips": {
            "rule": "Describe the SOUND, not the story. Narrative prompts work better than keyword lists.",
            "elements": {
                "genre": "Modern J-Pop / Acoustic Jazz / Dark Trap / Lo-fi Hip-hop / Anime OST",
                "tempo": "132 BPM / slow and intimate / driving beat / half-time feel",
                "instruments": "bright piano / smooth saxophone / 808 sub-bass / nylon guitar / brass section",
                "vocal_style": "emotional female vocal / whispered delivery / powerful male tenor / rap verse",
                "mood": "melancholic / triumphant / cozy / aggressive / dreamy / nostalgic",
                "production": "polished radio-ready / lo-fi vinyl texture / orchestral / live band recording",
            },
            "good_example": "A melancholic piano ballad where soft female vocals weave through gentle string accompaniment, creating an intimate and heartbreaking atmosphere. 80 BPM.",
            "bad_example": "piano, sad, female, slow",
        },
        "lyrics_format": {
            "rule": "Use [Square Bracket] section tags at the start of a line.",
            "tags": {
                "[Intro]": "Opening, often instrumental",
                "[Verse 1]": "First verse",
                "[Pre-Chorus]": "Build-up before chorus",
                "[Chorus]": "The hook",
                "[Bridge]": "Contrasting section",
                "[Instrumental]": "No vocals",
                "[Outro]": "Closing section",
            },
            "modifiers": "[Verse 1 - Female], [Chorus - Both], [Bridge - Whispered], [Outro - Fade out]",
            "no_vocals": "Set lyrics to '[Instrumental]' for a fully instrumental output.",
        },
        "language_codes": {
            "ja": "Japanese", "en": "English", "ko": "Korean",
            "zh": "Mandarin Chinese", "yue": "Cantonese",
            "es": "Spanish", "fr": "French", "de": "German",
            "unknown": "Auto-detect (may reduce quality)",
        },
        "advanced_params": {
            "shift": "Timestep schedule. 3.0 is correct; 1.0 causes noisy output.",
            "inference_steps": "More steps = higher quality, slower.",
            "guidance_scale": "CFG strength (xl-base only).",
            "seed": "-1 = random. Same seed + same params = identical output.",
        },
    }
    return JSONResponse(content=content)
