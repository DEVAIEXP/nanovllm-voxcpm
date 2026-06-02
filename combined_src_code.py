

############################################################
# FILE: __init__.py
############################################################



# --- END OF __init__.py ---


############################################################
# FILE: api\__init__.py
############################################################



# --- END OF api\__init__.py ---


############################################################
# FILE: api\api.py
############################################################

from __future__ import annotations

from fastapi import APIRouter

from app.api.routes.encode_latents import router as encode_latents_router
from app.api.routes.generate import router as generate_router
from app.api.routes.health import router as health_router
from app.api.routes.info import router as info_router
from app.api.routes.lora import router as lora_router
from app.api.routes.metrics import router as metrics_router

api_router = APIRouter()
api_router.include_router(health_router)
api_router.include_router(info_router)
api_router.include_router(lora_router)
api_router.include_router(metrics_router)
api_router.include_router(encode_latents_router)
api_router.include_router(generate_router)


# --- END OF api\api.py ---


############################################################
# FILE: api\deps.py
############################################################

from __future__ import annotations

from typing import Any, cast

from fastapi import HTTPException, Request


def get_server(request: Request) -> Any:
    server = getattr(request.app.state, "server", None)
    if server is None:
        raise HTTPException(status_code=503, detail="Model server not ready")
    # app.state is dynamically typed; normalize for type checkers.
    return cast(Any, server)


# --- END OF api\deps.py ---


############################################################
# FILE: api\routes\__init__.py
############################################################



# --- END OF api\routes\__init__.py ---


############################################################
# FILE: api\routes\encode_latents.py
############################################################

from __future__ import annotations

import base64
import time
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import get_server
from app.core.metrics import (
    ENCODE_LATENTS_DURATION_SECONDS,
    ENCODE_LATENTS_REQUESTS_TOTAL,
)
from app.schemas.http import EncodeLatentsRequest, EncodeLatentsResponse, ErrorResponse

router = APIRouter(tags=["latents"])


@router.post(
    "/encode_latents",
    response_model=EncodeLatentsResponse,
    summary="Encode prompt audio to prompt latents",
    responses={
        400: {"description": "Invalid input", "model": ErrorResponse},
        503: {"description": "Model server not ready", "model": ErrorResponse},
        500: {"description": "Internal error", "model": ErrorResponse},
    },
)
async def encode_latents(
    req: EncodeLatentsRequest,
    server: Any = Depends(get_server),
) -> EncodeLatentsResponse:
    """Decode an audio file and return serialized float32 prompt latents."""

    t0 = time.perf_counter()
    try:
        wav = base64.b64decode(req.wav_base64)
    except Exception as e:
        ENCODE_LATENTS_REQUESTS_TOTAL.labels(status="400").inc()
        raise HTTPException(status_code=400, detail=f"Invalid base64 in wav_base64: {e}") from e

    try:
        latents = await server.encode_latents(wav, req.wav_format)
        model_info = await server.get_model_info()
    except HTTPException:
        raise
    except Exception as e:
        ENCODE_LATENTS_REQUESTS_TOTAL.labels(status="500").inc()
        raise HTTPException(status_code=500, detail=str(e)) from e
    finally:
        ENCODE_LATENTS_DURATION_SECONDS.observe(time.perf_counter() - t0)

    ENCODE_LATENTS_REQUESTS_TOTAL.labels(status="200").inc()
    return EncodeLatentsResponse(
        prompt_latents_base64=base64.b64encode(latents).decode("utf-8"),
        feat_dim=int(model_info["feat_dim"]),
        sample_rate=int(model_info.get("encoder_sample_rate", model_info["sample_rate"])),
        channels=int(model_info["channels"]),
    )


# --- END OF api\routes\encode_latents.py ---


############################################################
# FILE: api\routes\generate.py
############################################################

from __future__ import annotations

import base64
import inspect
import time
from typing import Any, AsyncIterator

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from numpy.typing import NDArray

from app.api.deps import get_server
from app.core.metrics import (
    GENERATE_AUDIO_SECONDS_TOTAL,
    GENERATE_STREAM_BYTES_TOTAL,
    GENERATE_TTFB_SECONDS,
)
from app.schemas.http import ErrorResponse, GenerateRequest
from app.services.mp3 import stream_mp3

router = APIRouter(tags=["generation"])


def _decode_latents_base64(value: str, field_name: str, feat_dim: int) -> bytes:
    try:
        latents = base64.b64decode(value)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 in {field_name}: {e}") from e

    try:
        np.frombuffer(latents, dtype=np.float32).reshape(-1, feat_dim)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid latent payload in {field_name}: {e}") from e

    return latents


def _validate_generate_prompt(req: GenerateRequest) -> None:
    has_wav = req.prompt_wav_base64 is not None or req.prompt_wav_format is not None
    has_latents = req.prompt_latents_base64 is not None
    has_ref_wav = req.ref_audio_wav_base64 is not None or req.ref_audio_wav_format is not None
    has_ref_latents = req.ref_audio_latents_base64 is not None

    if has_wav and has_latents:
        raise HTTPException(
            status_code=400,
            detail="prompt_wav_* and prompt_latents_base64 are mutually exclusive",
        )

    if has_ref_wav and has_ref_latents:
        raise HTTPException(
            status_code=400,
            detail="ref_audio_wav_* and ref_audio_latents_base64 are mutually exclusive",
        )

    if has_ref_wav and (req.ref_audio_wav_base64 is None or req.ref_audio_wav_format is None):
        raise HTTPException(
            status_code=400,
            detail="reference wav requires ref_audio_wav_base64 + ref_audio_wav_format",
        )

    if has_wav:
        if req.prompt_wav_base64 is None or req.prompt_wav_format is None:
            raise HTTPException(
                status_code=400,
                detail="wav prompt requires prompt_wav_base64 + prompt_wav_format",
            )
        if req.prompt_text is None or req.prompt_text == "":
            raise HTTPException(status_code=400, detail="wav prompt requires non-empty prompt_text")
        return

    if has_latents:
        if req.prompt_text is None or req.prompt_text == "":
            raise HTTPException(status_code=400, detail="latents prompt requires non-empty prompt_text")
        return

    if req.prompt_text not in (None, ""):
        raise HTTPException(status_code=400, detail="prompt_text is not allowed for zero-shot")


@router.post(
    "/generate",
    response_class=StreamingResponse,
    summary="Generate audio (streaming MP3)",
    responses={
        200: {
            "description": "MP3 byte stream",
            "content": {
                "audio/mpeg": {
                    "schema": {"type": "string", "format": "binary"},
                }
            },
            "headers": {
                "X-Audio-Sample-Rate": {
                    "description": "Audio sample rate in Hz.",
                    "schema": {"type": "integer"},
                },
                "X-Audio-Channels": {
                    "description": "Number of audio channels.",
                    "schema": {"type": "integer"},
                },
            },
        },
        400: {"description": "Invalid input", "model": ErrorResponse},
        503: {"description": "Model server not ready", "model": ErrorResponse},
        500: {"description": "Internal error", "model": ErrorResponse},
    },
)
async def generate(
    req: GenerateRequest,
    request: Request,
    server: Any = Depends(get_server),
) -> StreamingResponse:
    """Generate speech audio as a streamed MP3 byte stream.

    The response is streamed and may terminate early if the client disconnects or
    an internal error occurs after streaming has started.
    """

    _validate_generate_prompt(req)

    cfg = getattr(request.app.state, "cfg", None)
    if cfg is None:
        raise HTTPException(status_code=500, detail="server misconfigured: missing app.state.cfg")

    model_info = await server.get_model_info()
    sample_rate = int(model_info["sample_rate"])
    channels = int(model_info["channels"])
    feat_dim = int(model_info["feat_dim"])
    if channels != 1:
        raise HTTPException(status_code=500, detail=f"Only mono is supported (channels={channels})")

    if req.lora_name is not None:
        registered_loras = {str(item["name"]) for item in await server.list_loras()}
        if req.lora_name not in registered_loras:
            raise HTTPException(status_code=400, detail=f"LoRA '{req.lora_name}' is not registered")

    prompt_latents: bytes | None = None
    ref_audio_latents: bytes | None = None
    prompt_text = ""
    if req.prompt_wav_base64 is not None:
        try:
            wav = base64.b64decode(req.prompt_wav_base64)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64 in prompt_wav_base64: {e}") from e
        assert req.prompt_wav_format is not None
        assert req.prompt_text is not None
        prompt_latents = await server.encode_latents(wav, req.prompt_wav_format)
        prompt_text = req.prompt_text
    elif req.prompt_latents_base64 is not None:
        prompt_latents = _decode_latents_base64(req.prompt_latents_base64, "prompt_latents_base64", feat_dim)
        assert req.prompt_text is not None
        prompt_text = req.prompt_text

    if req.ref_audio_wav_base64 is not None:
        try:
            wav = base64.b64decode(req.ref_audio_wav_base64)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64 in ref_audio_wav_base64: {e}") from e
        assert req.ref_audio_wav_format is not None
        ref_audio_latents = await server.encode_latents(wav, req.ref_audio_wav_format)
    elif req.ref_audio_latents_base64 is not None:
        ref_audio_latents = _decode_latents_base64(req.ref_audio_latents_base64, "ref_audio_latents_base64", feat_dim)

    generate_kwargs = {
        "target_text": req.target_text,
        "prompt_latents": prompt_latents,
        "prompt_text": prompt_text,
        "max_generate_length": req.max_generate_length,
        "temperature": req.temperature,
        "cfg_value": req.cfg_value,
        "lora_name": req.lora_name,
    }
    if ref_audio_latents is not None:
        generate_kwargs["ref_audio_latents"] = ref_audio_latents

    if ref_audio_latents is not None:
        generate_params = inspect.signature(server.generate).parameters
        if "ref_audio_latents" not in generate_params:
            raise HTTPException(status_code=400, detail="Reference audio is not supported by the loaded model")

    stream = server.generate(**generate_kwargs)

    first_chunk: NDArray[np.float32] | None = None
    stream_exhausted = False
    try:
        first_chunk = await anext(stream)
    except StopAsyncIteration:
        stream_exhausted = True
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    start_t = time.perf_counter()
    ttfb_recorded = False

    async def wav_chunks() -> AsyncIterator[NDArray[np.float32]]:
        if first_chunk is not None:
            GENERATE_AUDIO_SECONDS_TOTAL.inc(float(first_chunk.shape[0]) / float(sample_rate))
            yield first_chunk

        if stream_exhausted:
            return

        try:
            async for chunk in stream:
                GENERATE_AUDIO_SECONDS_TOTAL.inc(float(chunk.shape[0]) / float(sample_rate))
                yield chunk
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

    async def body() -> AsyncIterator[bytes]:
        nonlocal ttfb_recorded
        async for b in stream_mp3(
            request=request,
            wav_chunks=wav_chunks(),
            sample_rate=sample_rate,
            mp3=cfg.mp3,
        ):
            if not ttfb_recorded:
                GENERATE_TTFB_SECONDS.observe(time.perf_counter() - start_t)
                ttfb_recorded = True
            GENERATE_STREAM_BYTES_TOTAL.inc(len(b))
            yield b
        if not ttfb_recorded:
            GENERATE_TTFB_SECONDS.observe(time.perf_counter() - start_t)

    return StreamingResponse(
        body(),
        media_type="audio/mpeg",
        headers={
            "X-Audio-Sample-Rate": str(sample_rate),
            "X-Audio-Channels": str(channels),
        },
    )


# --- END OF api\routes\generate.py ---


############################################################
# FILE: api\routes\health.py
############################################################

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from app.schemas.http import ErrorResponse, HealthResponse

router = APIRouter(tags=["health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness probe",
)
async def health() -> HealthResponse:
    """Liveness probe."""

    return HealthResponse()


@router.get(
    "/ready",
    response_model=HealthResponse,
    summary="Readiness probe",
    responses={
        503: {
            "description": "Model is still loading",
            "model": ErrorResponse,
        }
    },
)
async def ready(request: Request) -> HealthResponse:
    """Return 200 only after the model server is ready."""

    if not getattr(request.app.state, "ready", False):
        raise HTTPException(status_code=503, detail="not ready")
    return HealthResponse()


# --- END OF api\routes\health.py ---


############################################################
# FILE: api\routes\info.py
############################################################

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request

from app.api.deps import get_server
from app.core.config import materialize_lora_config
from app.schemas.http import ErrorResponse, InfoResponse, LoRAInfo, ModelInfo, Mp3Info

router = APIRouter(tags=["info"])


@router.get(
    "/info",
    response_model=InfoResponse,
    summary="Get model and service metadata",
    responses={
        503: {
            "description": "Model server not ready",
            "model": ErrorResponse,
        }
    },
)
async def info(request: Request, server: Any = Depends(get_server)) -> InfoResponse:
    """Return model metadata and instance-level configuration."""

    cfg = getattr(request.app.state, "cfg", None)
    model_info = await server.get_model_info()
    registered_loras = [str(item["name"]) for item in await server.list_loras()]
    model_architecture = getattr(request.app.state, "model_architecture", None)
    lora_config = None
    cfg_lora = getattr(cfg, "lora", None)
    if cfg_lora is not None and model_architecture is not None:
        lora_config = materialize_lora_config(cfg_lora, model_architecture)
    return InfoResponse(
        model=ModelInfo(
            sample_rate=int(model_info["sample_rate"]),
            channels=int(model_info["channels"]),
            feat_dim=int(model_info["feat_dim"]),
            patch_size=int(model_info["patch_size"]),
            model_path=str(model_info["model_path"]),
        ),
        lora=LoRAInfo(
            enabled=lora_config is not None,
            enable_lm=bool(getattr(lora_config, "enable_lm", False)),
            enable_dit=bool(getattr(lora_config, "enable_dit", False)),
            enable_proj=bool(getattr(lora_config, "enable_proj", False)),
            max_loras=getattr(lora_config, "max_loras", None),
            max_lora_rank=getattr(lora_config, "max_lora_rank", None),
            target_modules_lm=list(getattr(lora_config, "target_modules_lm", ())),
            target_modules_dit=list(getattr(lora_config, "target_modules_dit", ())),
            target_proj_modules=list(getattr(lora_config, "target_proj_modules", ())),
            registered_names=registered_loras,
            loaded=bool(registered_loras),
        ),
        mp3=Mp3Info(
            bitrate_kbps=getattr(getattr(cfg, "mp3", None), "bitrate_kbps", None),
            quality=getattr(getattr(cfg, "mp3", None), "quality", None),
        ),
    )


# --- END OF api\routes\info.py ---


############################################################
# FILE: api\routes\lora.py
############################################################

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request

from app.api.deps import get_server
from app.schemas.http import (
    ErrorResponse,
    RegisterLoRARequest,
    RegisterLoRAResponse,
    RegisteredLoRA,
    UnregisterLoRAResponse,
)

router = APIRouter(tags=["lora"])


@router.get(
    "/loras",
    response_model=list[RegisteredLoRA],
    summary="List registered LoRA adapters",
    responses={503: {"description": "Model server not ready", "model": ErrorResponse}},
)
async def list_loras(server: Any = Depends(get_server)) -> list[RegisteredLoRA]:
    return [RegisteredLoRA(name=str(item["name"])) for item in await server.list_loras()]


@router.post(
    "/loras",
    response_model=RegisterLoRAResponse,
    summary="Register a LoRA adapter",
    responses={
        400: {"description": "Invalid input", "model": ErrorResponse},
        503: {"description": "Model server not ready", "model": ErrorResponse},
    },
)
async def register_lora(
    req: RegisterLoRARequest, request: Request, server: Any = Depends(get_server)
) -> RegisterLoRAResponse:
    cfg = getattr(request.app.state, "cfg", None)
    if getattr(cfg, "lora", None) is None:
        raise HTTPException(status_code=400, detail="Runtime LoRA is disabled; set NANOVLLM_LORA_ENABLED=true")
    try:
        result = await server.register_lora(req.name, req.path)
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return RegisterLoRAResponse(name=str(result["name"]))


@router.delete(
    "/loras/{name}",
    response_model=UnregisterLoRAResponse,
    summary="Unregister a LoRA adapter",
    responses={
        400: {"description": "Invalid input", "model": ErrorResponse},
        503: {"description": "Model server not ready", "model": ErrorResponse},
    },
)
async def unregister_lora(name: str, server: Any = Depends(get_server)) -> UnregisterLoRAResponse:
    try:
        result = await server.unregister_lora(name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return UnregisterLoRAResponse(name=str(result["name"]))


# --- END OF api\routes\lora.py ---


############################################################
# FILE: api\routes\metrics.py
############################################################

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import Response

from app.core.metrics import metrics_response

router = APIRouter(tags=["metrics"])


@router.get(
    "/metrics",
    response_class=Response,
    summary="Prometheus metrics",
    responses={
        200: {
            "description": "Prometheus text exposition format",
            "content": {
                "text/plain": {
                    "schema": {
                        "type": "string",
                        "description": "Prometheus metrics in text format",
                    }
                }
            },
        }
    },
)
async def metrics() -> Response:
    """Expose Prometheus metrics for scraping."""

    return metrics_response()


# --- END OF api\routes\metrics.py ---


############################################################
# FILE: core\__init__.py
############################################################



# --- END OF core\__init__.py ---


############################################################
# FILE: core\config.py
############################################################

from __future__ import annotations

import os
from dataclasses import dataclass

ALL_LINEAR_LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
VOXCPM_PROJ_LORA_TARGETS = ("enc_to_lm_proj", "lm_to_dit_proj", "res_to_dit_proj")
VOXCPM2_PROJ_LORA_TARGETS = (*VOXCPM_PROJ_LORA_TARGETS, "fusion_concat_proj")


def _get_int_env(name: str, default: int) -> int:
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    try:
        return int(v)
    except ValueError as e:
        raise RuntimeError(f"Invalid env {name}={v!r}; expected int") from e


def _get_float_env(name: str, default: float) -> float:
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    try:
        return float(v)
    except ValueError as e:
        raise RuntimeError(f"Invalid env {name}={v!r}; expected float") from e


def _get_bool_env(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None or v == "":
        return default

    s = v.strip().lower()
    if s in ("1", "true", "t", "yes", "y", "on"):
        return True
    if s in ("0", "false", "f", "no", "n", "off"):
        return False
    raise RuntimeError(f"Invalid env {name}={v!r}; expected bool")


def _get_int_list_env(name: str, default: tuple[int, ...]) -> tuple[int, ...]:
    v = os.environ.get(name)
    if v is None or v == "":
        return default

    parts = [p.strip() for p in v.split(",")]
    parts = [p for p in parts if p != ""]
    if len(parts) == 0:
        raise RuntimeError(f"Invalid env {name}={v!r}; expected comma-separated ints")

    out: list[int] = []
    for p in parts:
        try:
            out.append(int(p))
        except ValueError as e:
            raise RuntimeError(f"Invalid env {name}={v!r}; expected comma-separated ints") from e
    return tuple(out)


def _get_str_list_env(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    v = os.environ.get(name)
    if v is None or v == "":
        return default

    parts = tuple(p.strip() for p in v.split(",") if p.strip() != "")
    if len(parts) == 0:
        raise RuntimeError(f"Invalid env {name}={v!r}; expected comma-separated strings")
    return parts


@dataclass(frozen=True)
class Mp3Config:
    bitrate_kbps: int
    quality: int


@dataclass(frozen=True)
class ServerPoolStartupConfig:
    max_num_batched_tokens: int
    max_num_seqs: int
    max_model_len: int
    gpu_memory_utilization: float
    enforce_eager: bool
    devices: tuple[int, ...]


@dataclass(frozen=True)
class RuntimeLoRAConfig:
    enable_lm: bool | None
    enable_dit: bool | None
    enable_proj: bool | None
    max_loras: int
    max_lora_rank: int
    target_modules_lm: tuple[str, ...] | None
    target_modules_dit: tuple[str, ...] | None
    target_proj_modules: tuple[str, ...] | None


@dataclass(frozen=True)
class MaterializedRuntimeLoRAConfig:
    enable_lm: bool
    enable_dit: bool
    enable_proj: bool
    max_loras: int
    max_lora_rank: int
    target_modules_lm: tuple[str, ...]
    target_modules_dit: tuple[str, ...]
    target_proj_modules: tuple[str, ...]


@dataclass(frozen=True)
class ServiceConfig:
    model_path: str
    mp3: Mp3Config
    server_pool: ServerPoolStartupConfig
    lora: RuntimeLoRAConfig | None


def load_config() -> ServiceConfig:
    model_path = os.path.expanduser(os.environ.get("NANOVLLM_MODEL_PATH", "~/VoxCPM1.5"))

    mp3_bitrate_kbps = _get_int_env("NANOVLLM_MP3_BITRATE_KBPS", 192)
    mp3_quality = _get_int_env("NANOVLLM_MP3_QUALITY", 2)
    if mp3_bitrate_kbps <= 0:
        raise RuntimeError("NANOVLLM_MP3_BITRATE_KBPS must be > 0")
    if mp3_quality < 0 or mp3_quality > 2:
        raise RuntimeError("NANOVLLM_MP3_QUALITY must be in [0, 2]")

    lora_uri = os.environ.get("NANOVLLM_LORA_URI")
    lora_id = os.environ.get("NANOVLLM_LORA_ID")
    lora_sha256 = os.environ.get("NANOVLLM_LORA_SHA256")

    if lora_uri or lora_id or lora_sha256:
        raise RuntimeError(
            "LoRA startup preload env vars were removed; use the runtime LoRA API with NANOVLLM_LORA_ENABLED=true"
        )

    runtime_lora_enabled = _get_bool_env("NANOVLLM_LORA_ENABLED", False)
    runtime_lora_config: RuntimeLoRAConfig | None = None
    if runtime_lora_enabled:
        lora_max_loras = _get_int_env("NANOVLLM_LORA_MAX_LORAS", 1)
        lora_max_lora_rank = _get_int_env("NANOVLLM_LORA_MAX_LORA_RANK", 32)
        lora_enable_lm = _get_optional_bool_env("NANOVLLM_LORA_ENABLE_LM")
        lora_enable_dit = _get_optional_bool_env("NANOVLLM_LORA_ENABLE_DIT")
        lora_enable_proj = _get_optional_bool_env("NANOVLLM_LORA_ENABLE_PROJ")
        target_modules_lm = _get_optional_str_list_env("NANOVLLM_LORA_TARGET_MODULES_LM")
        target_modules_dit = _get_optional_str_list_env("NANOVLLM_LORA_TARGET_MODULES_DIT")
        target_proj_modules = _get_optional_str_list_env("NANOVLLM_LORA_TARGET_PROJ_MODULES")

        if lora_max_loras <= 0:
            raise RuntimeError("NANOVLLM_LORA_MAX_LORAS must be > 0")
        if lora_max_lora_rank <= 0:
            raise RuntimeError("NANOVLLM_LORA_MAX_LORA_RANK must be > 0")
        if lora_enable_lm is False and lora_enable_dit is False and lora_enable_proj is False:
            raise RuntimeError("At least one of NANOVLLM_LORA_ENABLE_LM/DIT/PROJ must be true")

        runtime_lora_config = RuntimeLoRAConfig(
            enable_lm=lora_enable_lm,
            enable_dit=lora_enable_dit,
            enable_proj=lora_enable_proj,
            max_loras=lora_max_loras,
            max_lora_rank=lora_max_lora_rank,
            target_modules_lm=target_modules_lm,
            target_modules_dit=target_modules_dit,
            target_proj_modules=target_proj_modules,
        )

    # Server pool startup config (read at startup).
    pool_max_num_batched_tokens = _get_int_env("NANOVLLM_SERVERPOOL_MAX_NUM_BATCHED_TOKENS", 8192)
    pool_max_num_seqs = _get_int_env("NANOVLLM_SERVERPOOL_MAX_NUM_SEQS", 16)
    pool_max_model_len = _get_int_env("NANOVLLM_SERVERPOOL_MAX_MODEL_LEN", 4096)
    pool_gpu_memory_utilization = _get_float_env("NANOVLLM_SERVERPOOL_GPU_MEMORY_UTILIZATION", 0.95)
    pool_enforce_eager = _get_bool_env("NANOVLLM_SERVERPOOL_ENFORCE_EAGER", False)
    pool_devices = _get_int_list_env("NANOVLLM_SERVERPOOL_DEVICES", (0,))

    if pool_max_num_batched_tokens <= 0:
        raise RuntimeError("NANOVLLM_SERVERPOOL_MAX_NUM_BATCHED_TOKENS must be > 0")
    if pool_max_num_seqs <= 0:
        raise RuntimeError("NANOVLLM_SERVERPOOL_MAX_NUM_SEQS must be > 0")
    if pool_max_model_len <= 0:
        raise RuntimeError("NANOVLLM_SERVERPOOL_MAX_MODEL_LEN must be > 0")
    if not (0.0 < pool_gpu_memory_utilization <= 1.0):
        raise RuntimeError("NANOVLLM_SERVERPOOL_GPU_MEMORY_UTILIZATION must be in (0, 1]")
    if len(pool_devices) == 0:
        raise RuntimeError("NANOVLLM_SERVERPOOL_DEVICES must be a non-empty list")
    if any(d < 0 for d in pool_devices):
        raise RuntimeError("NANOVLLM_SERVERPOOL_DEVICES entries must be >= 0")

    return ServiceConfig(
        model_path=model_path,
        mp3=Mp3Config(bitrate_kbps=mp3_bitrate_kbps, quality=mp3_quality),
        server_pool=ServerPoolStartupConfig(
            max_num_batched_tokens=pool_max_num_batched_tokens,
            max_num_seqs=pool_max_num_seqs,
            max_model_len=pool_max_model_len,
            gpu_memory_utilization=pool_gpu_memory_utilization,
            enforce_eager=pool_enforce_eager,
            devices=pool_devices,
        ),
        lora=runtime_lora_config,
    )


def _get_optional_bool_env(name: str) -> bool | None:
    if os.environ.get(name) in (None, ""):
        return None
    return _get_bool_env(name, False)


def _get_optional_str_list_env(name: str) -> tuple[str, ...] | None:
    if os.environ.get(name) in (None, ""):
        return None
    return _get_str_list_env(name, ())


def materialize_lora_config(config: RuntimeLoRAConfig, architecture: str) -> MaterializedRuntimeLoRAConfig:
    default_proj_targets: tuple[str, ...]
    if architecture == "voxcpm":
        default_proj_targets = VOXCPM_PROJ_LORA_TARGETS
    elif architecture == "voxcpm2":
        default_proj_targets = VOXCPM2_PROJ_LORA_TARGETS
    else:
        raise RuntimeError(f"Unsupported model architecture for runtime LoRA: {architecture}")

    enable_lm = True if config.enable_lm is None else config.enable_lm
    enable_dit = True if config.enable_dit is None else config.enable_dit
    enable_proj = True if config.enable_proj is None else config.enable_proj

    return MaterializedRuntimeLoRAConfig(
        enable_lm=enable_lm,
        enable_dit=enable_dit,
        enable_proj=enable_proj,
        max_loras=config.max_loras,
        max_lora_rank=config.max_lora_rank,
        target_modules_lm=config.target_modules_lm or ALL_LINEAR_LORA_TARGETS,
        target_modules_dit=config.target_modules_dit or ALL_LINEAR_LORA_TARGETS,
        target_proj_modules=config.target_proj_modules or default_proj_targets,
    )


# --- END OF core\config.py ---


############################################################
# FILE: core\lifespan.py
############################################################

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from huggingface_hub import snapshot_download

from nanovllm_voxcpm.llm import VoxCPM

from app.core.config import ServiceConfig, materialize_lora_config

SERVER_FACTORY = VoxCPM.from_pretrained


def _read_model_architecture(model_path: str) -> str:
    resolved_model_path = os.path.expanduser(model_path)
    if not os.path.isdir(resolved_model_path):
        resolved_model_path = snapshot_download(repo_id=model_path)
    config_file = os.path.join(resolved_model_path, "config.json")
    if not os.path.isfile(config_file):
        raise FileNotFoundError(f"Config file `{config_file}` not found")
    with open(config_file, encoding="utf-8") as f:
        config = json.load(f)
    architecture = config.get("architecture")
    if not isinstance(architecture, str) or architecture == "":
        raise RuntimeError(f"Config file `{config_file}` must define architecture")
    return architecture


def build_lifespan(cfg: ServiceConfig):
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        model_architecture = None
        lora_config = None
        if cfg.lora is not None:
            model_architecture = _read_model_architecture(cfg.model_path)
            lora_config = materialize_lora_config(cfg.lora, model_architecture)

        server = SERVER_FACTORY(
            model=cfg.model_path,
            max_num_batched_tokens=cfg.server_pool.max_num_batched_tokens,
            max_num_seqs=cfg.server_pool.max_num_seqs,
            max_model_len=cfg.server_pool.max_model_len,
            gpu_memory_utilization=cfg.server_pool.gpu_memory_utilization,
            enforce_eager=cfg.server_pool.enforce_eager,
            devices=list(cfg.server_pool.devices),
            lora_config=lora_config,
        )
        app.state.server = server
        app.state.model_architecture = model_architecture
        app.state.ready = False

        try:
            await server.wait_for_ready()

            app.state.ready = True
            yield
        finally:
            app.state.ready = False
            await server.stop()
            if getattr(app.state, "server", None) is server:
                delattr(app.state, "server")
            if getattr(app.state, "model_architecture", None) is model_architecture:
                delattr(app.state, "model_architecture")

    return lifespan


# --- END OF core\lifespan.py ---


############################################################
# FILE: core\metrics.py
############################################################

from __future__ import annotations

import time

from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

HTTP_REQUESTS_TOTAL = Counter(
    "nanovllm_http_requests_total",
    "Total HTTP requests",
    labelnames=["route", "method", "status"],
)
HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "nanovllm_http_request_duration_seconds",
    "HTTP request duration in seconds (includes streaming)",
    labelnames=["route", "method"],
)
INFLIGHT_REQUESTS = Gauge(
    "nanovllm_inflight_requests",
    "Number of in-flight HTTP requests",
    labelnames=["route"],
)
EXCEPTIONS_TOTAL = Counter(
    "nanovllm_exceptions_total",
    "Unhandled exceptions",
    labelnames=["route", "exception_type"],
)

GENERATE_TTFB_SECONDS = Histogram(
    "nanovllm_generate_ttfb_seconds",
    "Time-to-first-byte for /generate streaming responses",
)
GENERATE_AUDIO_SECONDS_TOTAL = Counter(
    "nanovllm_generate_audio_seconds_total",
    "Total generated audio duration in seconds",
)
GENERATE_STREAM_BYTES_TOTAL = Counter(
    "nanovllm_generate_stream_bytes_total",
    "Total bytes streamed by /generate",
)

AUDIO_ENCODE_FAILURES_TOTAL = Counter(
    "nanovllm_audio_encode_failures_total",
    "MP3 encoding failures",
)
AUDIO_ENCODE_SECONDS = Histogram(
    "nanovllm_audio_encode_seconds",
    "Time spent in MP3 encoder.encode() calls",
)

ENCODE_LATENTS_REQUESTS_TOTAL = Counter(
    "nanovllm_encode_latents_requests_total",
    "Total /encode_latents requests",
    labelnames=["status"],
)
ENCODE_LATENTS_DURATION_SECONDS = Histogram(
    "nanovllm_encode_latents_duration_seconds",
    "Latency of /encode_latents in seconds",
)


def install_metrics(app: FastAPI) -> None:
    @app.middleware("http")
    async def metrics_middleware(request: Request, call_next):
        route = request.url.path
        method = request.method
        start = time.perf_counter()

        INFLIGHT_REQUESTS.labels(route=route).inc()
        try:
            response = await call_next(request)
        except Exception as e:
            EXCEPTIONS_TOTAL.labels(route=route, exception_type=type(e).__name__).inc()
            dur = time.perf_counter() - start
            HTTP_REQUEST_DURATION_SECONDS.labels(route=route, method=method).observe(dur)
            HTTP_REQUESTS_TOTAL.labels(route=route, method=method, status="500").inc()
            INFLIGHT_REQUESTS.labels(route=route).dec()
            raise

        status = str(response.status_code)

        if isinstance(response, StreamingResponse):
            original_iter = response.body_iterator

            async def wrapped_iter():
                try:
                    async for chunk in original_iter:
                        yield chunk
                except Exception as e:
                    EXCEPTIONS_TOTAL.labels(route=route, exception_type=type(e).__name__).inc()
                    raise
                finally:
                    dur = time.perf_counter() - start
                    HTTP_REQUEST_DURATION_SECONDS.labels(route=route, method=method).observe(dur)
                    HTTP_REQUESTS_TOTAL.labels(route=route, method=method, status=status).inc()
                    INFLIGHT_REQUESTS.labels(route=route).dec()

            response.body_iterator = wrapped_iter()
            return response

        dur = time.perf_counter() - start
        HTTP_REQUEST_DURATION_SECONDS.labels(route=route, method=method).observe(dur)
        HTTP_REQUESTS_TOTAL.labels(route=route, method=method, status=status).inc()
        INFLIGHT_REQUESTS.labels(route=route).dec()
        return response


def metrics_response() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# --- END OF core\metrics.py ---


############################################################
# FILE: main.py
############################################################

from __future__ import annotations

from fastapi import FastAPI

from app.api.api import api_router
from app.core.config import load_config
from app.core.lifespan import build_lifespan
from app.core.metrics import install_metrics


def create_app() -> FastAPI:
    cfg = load_config()
    app = FastAPI(
        title="nano-vllm VoxCPM Service",
        version="0.1.0",
        description=(
            "Production-oriented FastAPI wrapper for nano-vllm-voxcpm. "
            "See /docs for interactive API docs and /openapi.json for the OpenAPI schema."
        ),
        openapi_tags=[
            {"name": "health", "description": "Liveness and readiness probes."},
            {"name": "info", "description": "Model and instance metadata."},
            {"name": "metrics", "description": "Prometheus metrics."},
            {"name": "lora", "description": "Runtime LoRA adapter management."},
            {
                "name": "latents",
                "description": "Encode prompt audio to prompt latents.",
            },
            {
                "name": "generation",
                "description": "Text-to-speech generation (streaming MP3).",
            },
        ],
        lifespan=build_lifespan(cfg),
    )
    app.state.cfg = cfg
    install_metrics(app)
    app.include_router(api_router)
    return app


app = create_app()


# --- END OF main.py ---


############################################################
# FILE: schemas\__init__.py
############################################################



# --- END OF schemas\__init__.py ---


############################################################
# FILE: schemas\http.py
############################################################

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Standard health response for liveness/readiness endpoints."""

    status: Literal["ok"] = "ok"


class ErrorResponse(BaseModel):
    """Default error response shape produced by FastAPI for HTTPException.

    Note: FastAPI's validation errors (422) use a different schema.
    """

    detail: str = Field(..., description="Human-readable error message.")


class ModelInfo(BaseModel):
    """Read-only model metadata returned by the engine."""

    sample_rate: int = Field(..., description="Audio sample rate in Hz.", examples=[16000])
    channels: int = Field(..., description="Number of audio channels.", examples=[1])
    feat_dim: int = Field(..., description="Latent feature dimension.", examples=[64])
    patch_size: int = Field(..., description="Model patch size.", examples=[2])
    model_path: str = Field(
        ...,
        description="Resolved model path used by this instance.",
        examples=["/models/VoxCPM1.5"],
    )


class Mp3Info(BaseModel):
    """MP3 encoder configuration used by /generate."""

    bitrate_kbps: int | None = Field(None, description="Constant bitrate used for MP3 encoding.", examples=[192])
    quality: int | None = Field(None, description="LAME quality preset (0 is best, 2 is fast).", examples=[2])


class LoRAInfo(BaseModel):
    """Runtime LoRA registration state."""

    enabled: bool = Field(..., description="Whether runtime LoRA capacity is enabled for this deployment instance.")
    enable_lm: bool = Field(..., description="Whether LM LoRA capacity is enabled.")
    enable_dit: bool = Field(..., description="Whether DiT LoRA capacity is enabled.")
    enable_proj: bool = Field(..., description="Whether projection-layer LoRA capacity is enabled.")
    max_loras: int | None = Field(None, description="Maximum concurrently resident LoRA adapters per layer.")
    max_lora_rank: int | None = Field(None, description="Maximum supported LoRA rank per layer slot.")
    target_modules_lm: list[str] = Field(default_factory=list, description="Enabled LM LoRA target modules.")
    target_modules_dit: list[str] = Field(default_factory=list, description="Enabled DiT LoRA target modules.")
    target_proj_modules: list[str] = Field(default_factory=list, description="Enabled projection LoRA target modules.")
    registered_names: list[str] = Field(default_factory=list, description="Currently registered LoRA adapter names.")
    loaded: bool = Field(
        ...,
        description="Whether at least one LoRA adapter is currently registered.",
        examples=[False],
    )


class RegisteredLoRA(BaseModel):
    """Registered LoRA adapter metadata."""

    name: str = Field(..., description="Logical LoRA adapter name.", examples=["demo-lora"])


class RegisterLoRARequest(BaseModel):
    """Request body for POST /loras."""

    name: str = Field(..., description="Logical LoRA adapter name.", examples=["demo-lora"])
    path: str = Field(..., description="Filesystem path to the LoRA checkpoint directory.")


class RegisterLoRAResponse(BaseModel):
    """Response body for POST /loras."""

    name: str = Field(..., description="Registered LoRA adapter name.")


class UnregisterLoRAResponse(BaseModel):
    """Response body for DELETE /loras/{name}."""

    name: str = Field(..., description="Unregistered LoRA adapter name.")


class InfoResponse(BaseModel):
    """Response for GET /info."""

    model: ModelInfo
    lora: LoRAInfo
    mp3: Mp3Info


class EncodeLatentsRequest(BaseModel):
    """Request body for POST /encode_latents."""

    wav_base64: str = Field(
        ...,
        description="Base64-encoded audio file bytes (entire file contents). Do not include a data URI prefix.",
        examples=["UklGRiQAAABXQVZFZm10IBAAAAABAAEA..."],
    )
    wav_format: str = Field(
        ...,
        description="Audio container format for decoding (e.g. 'wav', 'flac', 'mp3'); passed to torchaudio.",
        examples=["wav"],
    )


class EncodeLatentsResponse(BaseModel):
    """Response body for POST /encode_latents."""

    prompt_latents_base64: str
    feat_dim: int
    latents_dtype: Literal["float32"] = "float32"
    sample_rate: int
    channels: int


class GenerateRequest(BaseModel):
    """Request body for POST /generate.

    Prompt forms (mutually exclusive):

    - Zero-shot: omit all prompt_* fields.
    - WAV prompt: set prompt_wav_base64 + prompt_wav_format + prompt_text.
    - Latents prompt: set prompt_latents_base64 + prompt_text.

    Reference audio (optional, mutually exclusive within the ref_audio_* group):

    - WAV reference: set ref_audio_wav_base64 + ref_audio_wav_format.
    - Latents reference: set ref_audio_latents_base64.
    """

    target_text: str = Field(..., description="Text to synthesize.")

    # Prompt forms (mutually exclusive):
    prompt_wav_base64: str | None = Field(
        None,
        description="(wav prompt) Base64-encoded audio file bytes (entire file contents).",
    )
    prompt_wav_format: str | None = Field(
        None,
        description="(wav prompt) Audio container format for decoding (e.g. 'wav', 'flac', 'mp3').",
    )
    prompt_latents_base64: str | None = Field(
        None,
        description="(latents prompt) Base64-encoded float32 bytes returned by /encode_latents.",
    )
    prompt_text: str | None = Field(
        None,
        description="Prompt transcript text. Required for wav/latents prompt; omitted for zero-shot.",
    )

    ref_audio_wav_base64: str | None = Field(
        None,
        description="(reference audio) Base64-encoded audio file bytes (entire file contents).",
    )
    ref_audio_wav_format: str | None = Field(
        None,
        description="(reference audio) Audio container format for decoding (e.g. 'wav', 'flac', 'mp3').",
    )
    ref_audio_latents_base64: str | None = Field(
        None,
        description="(reference audio) Base64-encoded float32 bytes returned by /encode_latents.",
    )
    lora_name: str | None = Field(None, description="Registered LoRA adapter name to apply for this request.")

    max_generate_length: int = Field(2000, ge=1, description="Maximum number of model generation steps.")
    temperature: float = Field(1.0, ge=0.0, description="Sampling temperature.")
    cfg_value: float = Field(1.5, ge=0.0, description="Classifier-free guidance scale.")


# --- END OF schemas\http.py ---


############################################################
# FILE: services\__init__.py
############################################################



# --- END OF services\__init__.py ---


############################################################
# FILE: services\mp3.py
############################################################

from __future__ import annotations

import asyncio
import concurrent.futures
import queue
import threading
import time
from typing import AsyncIterator, Protocol

import numpy as np

from app.core.config import Mp3Config
from app.core.metrics import AUDIO_ENCODE_FAILURES_TOTAL, AUDIO_ENCODE_SECONDS


class _DisconnectableRequest(Protocol):
    async def is_disconnected(self) -> bool: ...


def float32_to_s16le_bytes(wav: np.ndarray) -> bytes:
    wav_f32 = wav.astype(np.float32, copy=False)
    wav_f32 = np.clip(wav_f32, -1.0, 1.0)
    wav_i16 = (wav_f32 * 32767.0).astype(np.int16, copy=False)
    return wav_i16.tobytes(order="C")


async def stream_mp3(
    *,
    request: _DisconnectableRequest,
    wav_chunks: AsyncIterator[np.ndarray],
    sample_rate: int,
    mp3: Mp3Config,
) -> AsyncIterator[bytes]:
    """Encode float32 mono waveform chunks to MP3 and stream bytes.

    Encoding is done in a background thread to avoid blocking the event loop.
    """

    pcm_q: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=8)
    mp3_q: queue.Queue[bytes | None] = queue.Queue(maxsize=8)
    stop_evt = threading.Event()
    thread_exc: list[BaseException] = []
    loop = asyncio.get_running_loop()
    io_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="mp3-queue")

    async def put_pcm(item: np.ndarray | None) -> None:
        await loop.run_in_executor(io_executor, pcm_q.put, item)

    def drain_queue(q: queue.Queue) -> None:
        while True:
            try:
                q.get_nowait()
            except queue.Empty:
                return

    def encoder_thread() -> None:
        try:
            import lameenc

            enc = lameenc.Encoder()
            enc.set_bit_rate(mp3.bitrate_kbps)
            enc.set_in_sample_rate(sample_rate)
            enc.set_channels(1)
            enc.set_quality(mp3.quality)

            encoded_any = False

            while True:
                item = pcm_q.get()
                if item is None or stop_evt.is_set():
                    break

                pcm_bytes = float32_to_s16le_bytes(item)
                t0 = time.perf_counter()
                out = enc.encode(pcm_bytes)
                encoded_any = True
                AUDIO_ENCODE_SECONDS.observe(time.perf_counter() - t0)

                # lameenc may return bytearray; StreamingResponse requires bytes.
                if out:
                    if isinstance(out, (bytearray, memoryview)):
                        out = bytes(out)
                    mp3_q.put(out)

            # Some lameenc builds raise if flush is called without encoding any samples.
            if encoded_any:
                out = enc.flush()
                if out:
                    if isinstance(out, (bytearray, memoryview)):
                        out = bytes(out)
                    mp3_q.put(out)
            mp3_q.put(None)
        except BaseException as e:
            AUDIO_ENCODE_FAILURES_TOTAL.inc()
            thread_exc.append(e)
            stop_evt.set()
            try:
                mp3_q.put(None)
            except Exception:
                pass

    enc_thread = threading.Thread(target=encoder_thread, name="mp3-encoder", daemon=True)
    enc_thread.start()

    async def pcm_producer() -> None:
        try:
            async for chunk in wav_chunks:
                if await request.is_disconnected():
                    stop_evt.set()
                    break
                await put_pcm(chunk)
        finally:
            try:
                await put_pcm(None)
            except Exception:
                pass

    producer_task = asyncio.create_task(pcm_producer())

    try:
        while True:
            item = await loop.run_in_executor(io_executor, mp3_q.get)
            if item is None:
                break
            yield item
        if thread_exc:
            raise RuntimeError(f"MP3 encoder failed: {thread_exc[0]}")
    finally:
        stop_evt.set()
        producer_task.cancel()
        drain_queue(pcm_q)
        drain_queue(mp3_q)
        try:
            pcm_q.put_nowait(None)
        except Exception:
            pass
        try:
            mp3_q.put_nowait(None)
        except Exception:
            pass
        try:
            await producer_task
        except asyncio.CancelledError:
            pass
        except Exception:
            pass
        try:
            await loop.run_in_executor(io_executor, enc_thread.join, 2.0)
        finally:
            io_executor.shutdown(wait=True, cancel_futures=True)


# --- END OF services\mp3.py ---
