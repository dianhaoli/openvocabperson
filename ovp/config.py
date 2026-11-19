"""Runtime configuration.

Everything is driven by environment variables (optionally from a ``.env`` file) so the
same code runs on a laptop CPU, an Apple Silicon Mac, a CUDA box, or fully against cloud
APIs. Model choices are *specs* of the form ``provider:model`` and can also be changed at
runtime through ``PATCH /api/config`` or per request.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import overload

try:  # optional
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover
    pass


@overload
def _env(name: str, default: str) -> str: ...


@overload
def _env(name: str, default: None = None) -> str | None: ...


def _env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value.strip()


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


def _env_list(name: str, default: str = "") -> list[str]:
    return [x.strip() for x in _env(name, default).split(",") if x.strip()]


def _env_bool(name: str, default: bool) -> bool:
    raw = _env(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


@dataclass
class Thresholds:
    """Re-ID cosine thresholds; they differ per embedding space."""

    match: float
    review: float


# Sensible defaults per Re-ID backend. OSNet features are discriminative; CLIP and the colour
# descriptor (``reid:color:v2``, see ``ColorReID``: same person >= 0.85, typically >= 0.91, different
# people <= 0.82 on the measured photos/videos) need tighter match thresholds.
REID_THRESHOLDS: dict[str, Thresholds] = {
    "osnet": Thresholds(match=0.75, review=0.60),
    "clip": Thresholds(match=0.88, review=0.80),
    "color": Thresholds(match=0.87, review=0.78),
}


@dataclass
class Settings:
    database_url: str = field(
        default_factory=lambda: _env(
            "DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/vision_analysis"
        )
    )
    storage_dir: Path = field(default_factory=lambda: Path(_env("OVP_STORAGE_DIR", "storage")))
    device: str = field(default_factory=lambda: _env("OVP_DEVICE", "auto"))

    # ── model specs ("auto" picks the best available at startup) ──
    detector: str = field(default_factory=lambda: _env("OVP_DETECTOR", "auto"))
    segmenter: str = field(default_factory=lambda: _env("OVP_SEGMENTER", "auto"))
    vlm: str = field(default_factory=lambda: _env("OVP_VLM", "auto"))
    llm: str = field(default_factory=lambda: _env("OVP_LLM", "auto"))
    embedder: str = field(default_factory=lambda: _env("OVP_EMBEDDER", "auto"))
    text_embedder: str = field(default_factory=lambda: _env("OVP_TEXT_EMBEDDER", "auto"))
    reid: str = field(default_factory=lambda: _env("OVP_REID", "auto"))

    # ── provider credentials / endpoints ──
    openai_api_key: str | None = field(default_factory=lambda: _env("OPENAI_API_KEY"))
    openai_base_url: str = field(
        default_factory=lambda: _env("OPENAI_BASE_URL", "https://api.openai.com/v1")
    )
    anthropic_api_key: str | None = field(default_factory=lambda: _env("ANTHROPIC_API_KEY"))
    anthropic_base_url: str = field(default_factory=lambda: _env("ANTHROPIC_BASE_URL", "https://api.anthropic.com"))
    gemini_api_key: str | None = field(
        default_factory=lambda: _env("GEMINI_API_KEY", _env("GOOGLE_API_KEY"))
    )
    gemini_base_url: str = field(
        default_factory=lambda: _env("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai")
    )
    openrouter_api_key: str | None = field(default_factory=lambda: _env("OPENROUTER_API_KEY"))
    ollama_base_url: str = field(
        default_factory=lambda: _env("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    )
    # Any OpenAI-compatible server (vLLM, LM Studio, TGI, llama.cpp ...)
    compat_base_url: str | None = field(default_factory=lambda: _env("OVP_COMPAT_BASE_URL"))
    compat_api_key: str | None = field(default_factory=lambda: _env("OVP_COMPAT_API_KEY"))

    # default cloud model names
    openai_vlm_model: str = field(default_factory=lambda: _env("OVP_OPENAI_VLM_MODEL", "gpt-4.1-mini"))
    openai_llm_model: str = field(default_factory=lambda: _env("OVP_OPENAI_LLM_MODEL", "gpt-4.1-mini"))
    anthropic_vlm_model: str = field(
        default_factory=lambda: _env("OVP_ANTHROPIC_VLM_MODEL", "claude-haiku-5-5")
    )
    anthropic_llm_model: str = field(
        default_factory=lambda: _env("OVP_ANTHROPIC_LLM_MODEL", "claude-sonnet-5-5")
    )
    gemini_model: str = field(default_factory=lambda: _env("OVP_GEMINI_MODEL", "gemini-2.5-flash"))
    local_vlm_model: str = field(
        default_factory=lambda: _env("OVP_LOCAL_VLM_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct")
    )

    # ── detection / segmentation ──
    detection_conf: float = field(default_factory=lambda: _env_float("OVP_CONF", 0.25))
    low_conf: float = field(default_factory=lambda: _env_float("OVP_LOW_CONF", 0.3))
    nms_iou: float = field(default_factory=lambda: _env_float("OVP_NMS_IOU", 0.7))
    expand_ratio: float = field(default_factory=lambda: _env_float("OVP_EXPAND_RATIO", 0.1))
    max_detections: int = field(default_factory=lambda: _env_int("OVP_MAX_DETECTIONS", 100))
    weights_dir: Path = field(default_factory=lambda: Path(_env("OVP_WEIGHTS_DIR", ".")))

    # ── VLM ──
    vlm_concurrency: int = field(default_factory=lambda: _env_int("OVP_VLM_CONCURRENCY", 8))
    vlm_batch_size: int = field(default_factory=lambda: _env_int("OVP_VLM_BATCH", 4))
    vlm_max_tokens: int = field(default_factory=lambda: _env_int("OVP_VLM_MAX_TOKENS", 260))
    max_vlm_crops: int = field(default_factory=lambda: _env_int("OVP_MAX_VLM_CROPS", 40))
    vlm_quantization: str = field(default_factory=lambda: _env("OVP_VLM_QUANT", "none"))  # none|8bit|4bit
    request_timeout: float = field(default_factory=lambda: _env_float("OVP_REQUEST_TIMEOUT", 90.0))

    # ── video ──
    video_fps: float = field(default_factory=lambda: _env_float("OVP_VIDEO_FPS", 2.0))
    max_frames: int = field(default_factory=lambda: _env_int("OVP_MAX_FRAMES", 600))
    min_track_hits: int = field(default_factory=lambda: _env_int("OVP_MIN_TRACK_HITS", 2))

    # ── jobs ──
    job_concurrency: int = field(default_factory=lambda: _env_int("OVP_JOB_CONCURRENCY", 2))
    media_concurrency: int = field(default_factory=lambda: _env_int("OVP_MEDIA_CONCURRENCY", 3))

    # ── identity ──
    reid_match_threshold: float | None = field(
        default_factory=lambda: float(v) if (v := _env("OVP_REID_MATCH")) else None
    )
    reid_review_threshold: float | None = field(
        default_factory=lambda: float(v) if (v := _env("OVP_REID_REVIEW")) else None
    )

    # ── server / security ──
    # Server-side folders that POST /api/analyze/paths (and the agent) may import from; empty
    # disables server-side path imports. Paths are resolved, so symlinks cannot escape a root.
    import_roots: list[Path] = field(
        default_factory=lambda: [Path(p).expanduser().resolve() for p in _env_list("OVP_IMPORT_ROOTS")]
    )
    # Browser origins allowed to call the API cross-origin (the UI served by the API itself is
    # same-origin and needs no entry); "*" allows any origin.
    cors_origins: list[str] = field(
        default_factory=lambda: _env_list("OVP_CORS_ORIGINS", "http://localhost:5173,http://localhost:3000")
    )
    max_upload_mb: int = field(default_factory=lambda: _env_int("OVP_MAX_UPLOAD_MB", 2048))

    # ── misc ──
    load_models_on_startup: bool = field(default_factory=lambda: _env_bool("OVP_PRELOAD", True))
    frontend_dist: Path = field(
        default_factory=lambda: Path(_env("OVP_FRONTEND_DIST", str(Path(__file__).resolve().parent.parent / "frontend" / "dist")))
    )

    def thresholds_for(self, reid_kind: str) -> Thresholds:
        base = REID_THRESHOLDS.get(reid_kind, REID_THRESHOLDS["osnet"])
        return Thresholds(
            match=self.reid_match_threshold if self.reid_match_threshold is not None else base.match,
            review=self.reid_review_threshold if self.reid_review_threshold is not None else base.review,
        )

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    def public(self) -> dict:
        """Settings safe to expose (no secrets)."""
        data = asdict(self)
        for key in list(data):
            if "api_key" in key:
                data[key] = bool(data[key])
        data["storage_dir"] = str(self.storage_dir)
        data["weights_dir"] = str(self.weights_dir)
        data["frontend_dist"] = str(self.frontend_dist)
        data["import_roots"] = [str(p) for p in self.import_roots]
        data.pop("database_url", None)
        return data


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings(settings: Settings | None = None) -> Settings:
    """Used by tests."""
    global _settings
    _settings = settings or Settings()
    return _settings
