"""Model registry: resolves specs, lazily constructs and caches model instances, and reports
what is available on this machine. ``auto`` picks the best option for the hardware/keys present.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any, Optional

from ..config import Settings, get_settings
from ..runtime import get_runtime
from . import detectors as D
from . import embedders as E
from . import llm as L
from . import segmenters as S

log = logging.getLogger("ovp.registry")

ROLES = ("detector", "segmenter", "vlm", "llm", "embedder", "text_embedder", "reid")


def _has(module: str) -> bool:
    try:
        __import__(module)
        return True
    except Exception:
        return False


_NETWORK_ERRORS = ("Tunnel connection failed", "Max retries exceeded", "urlopen error", "ProxyError", "Name or service not known", "Temporary failure in name resolution")


def describe_error(err: Exception | str) -> str:
    """A short, user-facing reason for a model that failed to load."""
    text = " ".join(str(err).split())
    if any(marker in text for marker in _NETWORK_ERRORS):
        host = re.search(r"host='([^']+)'", text)
        return "model weights could not be downloaded (offline or host blocked" + (f": {host.group(1)}" if host else "") + ")"
    return text[:500]


class ModelUnavailable(RuntimeError):
    def __init__(self, role: str, spec: str, reason: str):
        super().__init__(f"{role} {spec!r} is unavailable: {reason}")
        self.role, self.spec, self.reason = role, spec, reason


class ModelRegistry:
    def __init__(self, settings: Optional[Settings] = None):
        self.s = settings or get_settings()
        self._cache: dict[tuple[str, str], Any] = {}
        self.errors: dict[str, str] = {}
        self.defaults: dict[str, str] = {}
        self._locks: dict[tuple[str, str], asyncio.Lock] = {}
        self.resolve_defaults()

    # ── defaults ──────────────────────────────────────────────────────────────
    @property
    def device(self) -> str:
        return get_runtime().device

    def _auto(self, role: str) -> str:
        s, dev = self.s, self.device
        gpu = dev != "cpu"
        if role == "detector":
            if _has("ultralytics"):
                if D.clip_pkg_available():
                    return "yoloe:yoloe-11l-seg.pt" if dev == "cuda" else "yoloe:yoloe-11m-seg.pt"
                return "yolo:yolo11l.pt"
            vlm = self._auto("vlm")
            return f"vlm:{vlm}" if vlm != "none" else "none"
        if role == "segmenter":
            if not _has("ultralytics"):
                return "none"
            return "sam:sam2.1_b.pt" if dev == "cuda" else "sam:mobile_sam.pt"
        if role == "vlm":
            if gpu and _has("transformers") and _has("torch"):
                return f"qwen:{s.local_vlm_model}"
            if s.openai_api_key:
                return f"openai:{s.openai_vlm_model}"
            if s.anthropic_api_key:
                return f"anthropic:{s.anthropic_vlm_model}"
            if s.gemini_api_key:
                return f"gemini:{s.gemini_model}"
            if s.compat_base_url:
                return "compat:default"
            return "none"
        if role == "llm":
            if s.openai_api_key:
                return f"openai:{s.openai_llm_model}"
            if s.anthropic_api_key:
                return f"anthropic:{s.anthropic_llm_model}"
            if s.gemini_api_key:
                return f"gemini:{s.gemini_model}"
            if s.compat_base_url:
                return "compat:default"
            return "none"
        if role == "embedder":
            return "clip:ViT-B-32" if E.clip_available() else "none"
        if role == "text_embedder":
            if s.openai_api_key:
                return "openai:text-embedding-3-small"
            if s.gemini_api_key:
                return "gemini:gemini-embedding-001"
            if _has("sentence_transformers"):
                return "st:all-MiniLM-L6-v2"
            return "none"
        if role == "reid":
            if E.torchreid_available():
                return "osnet:osnet_x1_0"
            if E.clip_available():
                return "clip"
            return "color"
        raise KeyError(role)

    def open_vocab_fallbacks(self) -> list[str]:
        """Open-vocabulary detector specs to try when the default detector is closed-set (best first)."""
        # an open-vocab detector that is already loaded costs nothing to switch to
        specs = [spec for (role, spec), inst in self._cache.items() if role == "detector" and inst.open_vocab and inst.local]
        if _has("ultralytics") and D.clip_pkg_available():
            specs += ["yoloe:yoloe-11l-seg.pt" if self.device == "cuda" else "yoloe:yoloe-11m-seg.pt", "yolo-world:yolov8l-worldv2.pt"]
        vlm = self.defaults.get("vlm", "none")
        if vlm != "none":
            specs.append(f"vlm:{vlm}")
        return list(dict.fromkeys(specs))

    def resolve_defaults(self) -> None:
        for role in ROLES:
            raw = getattr(self.s, role)
            self.defaults[role] = self._auto(role) if raw in (None, "", "auto") else raw

    def set_default(self, role: str, spec: str) -> None:
        if role not in ROLES:
            raise KeyError(role)
        self.defaults[role] = self._auto(role) if spec == "auto" else spec

    # ── construction ──────────────────────────────────────────────────────────
    def _build(self, role: str, spec: str) -> Any:
        s = self.s
        if role == "detector":
            return D.create_detector(spec, s, vlm_factory=lambda vs: self.get("vlm", vs or self.defaults["vlm"]))
        if role == "segmenter":
            return S.create_segmenter(spec, s)
        if role in ("vlm", "llm"):
            return L.create_chat_model(spec, s)
        if role == "embedder":
            _, _, name = spec.partition(":")
            return E.ClipEmbedder(name or "ViT-B-32")
        if role == "text_embedder":
            provider, _, model = spec.partition(":")
            if provider == "openai":
                return E.ApiTextEmbedder("openai", model or "text-embedding-3-small", s.openai_base_url, s.openai_api_key)
            if provider == "gemini":
                return E.ApiTextEmbedder("gemini", model or "gemini-embedding-001", s.gemini_base_url, s.gemini_api_key)
            if provider == "ollama":
                return E.ApiTextEmbedder("ollama", model or "nomic-embed-text", s.ollama_base_url, None)
            if provider == "compat":
                return E.ApiTextEmbedder("compat", model, s.compat_base_url or "", s.compat_api_key)
            if provider == "st":
                return E.SentenceTransformerEmbedder(model or "all-MiniLM-L6-v2")
            if provider == "clip":
                return E.ClipTextEmbedder(self.get("embedder", spec if model else self.defaults["embedder"]))
            raise ValueError(f"unknown text embedder {spec!r}")
        if role == "reid":
            kind, _, name = spec.partition(":")
            if kind == "osnet":
                return E.OSNetReID(name or "osnet_x1_0")
            if kind == "clip":
                clip_spec = self.defaults["embedder"] if self.defaults["embedder"] != "none" else "clip:ViT-B-32"
                clip = self.get("embedder", clip_spec)
                if clip is None:
                    raise RuntimeError("CLIP embedder unavailable")
                return E.ClipReID(clip)
            if kind == "color":
                return E.ColorReID()
            raise ValueError(f"unknown reid model {spec!r}")
        raise KeyError(role)

    def get(self, role: str, spec: Optional[str] = None) -> Any:
        """Return a model for a role or ``None`` when the role is disabled/unavailable."""
        spec = spec or self.defaults.get(role, "none")
        if not spec or spec == "none":
            return None
        key = (role, spec)
        if f"{role}|{spec}" in self.errors:
            return None
        inst = self._cache.get(key)
        if inst is None:
            try:
                inst = self._build(role, spec)
            except Exception as e:
                self.errors[f"{role}|{spec}"] = describe_error(e)
                log.warning("cannot create %s %s: %s", role, spec, e)
                return None
            self._cache[key] = inst
        return inst

    def mark_failed(self, role: str, spec: str, err: Exception | str) -> None:
        self.errors[f"{role}|{spec}"] = describe_error(err)
        self._cache.pop((role, spec), None)

    def clear_error(self, role: str, spec: str) -> None:
        self.errors.pop(f"{role}|{spec}", None)

    async def ensure(self, role: str, spec: Optional[str] = None) -> Any:
        """Get a model and make sure it actually loads (weights downloaded etc.)."""
        spec = spec or self.defaults.get(role, "none")
        inst = self.get(role, spec)
        if inst is None:
            return None
        warm = getattr(inst, "warmup", None)
        if warm is None or getattr(inst, "_ovp_warm", False):
            return inst
        lock = self._locks.setdefault((role, spec), asyncio.Lock())
        async with lock:
            if getattr(inst, "_ovp_warm", False):
                return inst
            try:
                await warm()
                inst._ovp_warm = True
            except Exception as e:
                log.warning("%s %s failed to load: %s", role, spec, e)
                self.mark_failed(role, spec, e)
                return None
        return inst

    async def warmup_defaults(self, roles: tuple[str, ...] = ("detector", "embedder", "reid", "text_embedder")) -> None:
        for role in roles:
            spec = self.defaults.get(role)
            if spec and spec != "none":
                await self.ensure(role, spec)
        # Re-ID falls back gracefully: osnet → clip → color
        if self.defaults.get("reid") and self.get("reid") is None and self.defaults["reid"] != "color":
            self.defaults["reid"] = "clip" if self.get("embedder") is not None else "color"
            await self.ensure("reid")
            if self.get("reid") is None:
                self.defaults["reid"] = "color"

    async def switch_defaults(self, changes: dict[str, str]) -> None:
        """Change runtime defaults atomically: every new model must load, else nothing changes.

        Raises ``ModelUnavailable`` naming the first role that could not be loaded.
        """
        previous = dict(self.defaults)
        for role, spec in changes.items():
            self.clear_error(role, spec)  # allow retrying a model that failed earlier
            self.set_default(role, spec)
        try:
            for role in changes:
                spec = self.defaults[role]
                if spec == "none":
                    continue
                provider = spec.split(":", 1)[0]
                if role in ("vlm", "llm"):
                    ok, note = L.provider_available(provider, self.s)
                    if not ok:
                        raise ModelUnavailable(role, spec, note or "unavailable")
                if await self.ensure(role, spec) is None:
                    raise ModelUnavailable(role, spec, self.errors.get(f"{role}|{spec}", "unavailable"))
        except ModelUnavailable:
            self.defaults = previous
            raise

    async def aclose(self) -> None:
        for inst in self._cache.values():
            close = getattr(inst, "aclose", None)
            if close:
                try:
                    await close()
                except Exception:
                    pass

    # ── reporting ─────────────────────────────────────────────────────────────
    def reid_kind(self) -> Optional[str]:
        spec = self.defaults.get("reid", "none")
        if spec == "none" or self.get("reid") is None:
            return None
        return spec.split(":")[0]

    def _status(self, role: str, spec: str, label: str, available: bool, note: Optional[str] = None, **extra: Any) -> dict:
        err = self.errors.get(f"{role}|{spec}")
        return {
            "spec": spec,
            "label": label,
            "available": available and err is None,
            "note": err or note,
            **extra,
        }

    def catalog(self) -> dict[str, list[dict]]:
        s = self.s
        ul = _has("ultralytics")
        clip_pkg = D.clip_pkg_available()
        hf = _has("transformers") and _has("torch")
        out: dict[str, list[dict]] = {r: [] for r in ROLES}

        det = out["detector"]
        for w, lab in (("yoloe-11l-seg.pt", "YOLOE-11L"), ("yoloe-11m-seg.pt", "YOLOE-11M"), ("yoloe-11s-seg.pt", "YOLOE-11S")):
            det.append(self._status("detector", f"yoloe:{w}", f"{lab} (open-vocab + masks)", ul and clip_pkg, None if clip_pkg else "pip install git+https://github.com/ultralytics/CLIP.git", local=True, open_vocab=True, masks=True))
        det.append(self._status("detector", "yolo-world:yolov8l-worldv2.pt", "YOLO-World v2 L (open-vocab)", ul and clip_pkg, None, local=True, open_vocab=True, masks=False))
        det.append(self._status("detector", "yolo:yolo11l.pt", "YOLO11-L (COCO 80 classes)", ul, None, local=True, open_vocab=False, masks=False))
        det.append(self._status("detector", "yolo:yolo11l-seg.pt", "YOLO11-L-seg (COCO + masks)", ul, None, local=True, open_vocab=False, masks=True))
        det.append(self._status("detector", "owlv2:google/owlv2-base-patch16-ensemble", "OWLv2 (HF, open-vocab)", hf, None if hf else "pip install transformers", local=True, open_vocab=True, masks=False))
        det.append(self._status("detector", "grounding-dino:IDEA-Research/grounding-dino-tiny", "Grounding DINO tiny (HF, open-vocab)", hf, None if hf else "pip install transformers", local=True, open_vocab=True, masks=False))
        for vspec in self._vlm_specs():
            ok, note = L.provider_available(vspec.split(":")[0], s)
            det.append(self._status("detector", f"vlm:{vspec}", f"VLM grounding · {vspec}", ok, note or "boxes from a VLM; slower, best with Gemini / Qwen-VL", local=False, open_vocab=True, masks=False))

        seg = out["segmenter"]
        seg.append(self._status("segmenter", "detector", "Detector masks (YOLOE-seg, free)", True, "only when the detector produces masks", local=True))
        for w, lab in (("sam2.1_b.pt", "SAM 2.1 base"), ("sam2.1_t.pt", "SAM 2.1 tiny"), ("sam2.1_l.pt", "SAM 2.1 large"), ("mobile_sam.pt", "MobileSAM"), ("sam_b.pt", "SAM ViT-B")):
            seg.append(self._status("segmenter", f"sam:{w}", lab, ul, None if ul else "pip install ultralytics", local=True))
        seg.append(self._status("segmenter", "fastsam:FastSAM-s.pt", "FastSAM-s", ul, None, local=True))

        for role in ("vlm", "llm"):
            for vspec in self._vlm_specs(role):
                provider = vspec.split(":")[0]
                ok, note = L.provider_available(provider, s)
                local = provider in L.LOCAL_PROVIDERS
                if role == "llm" and local:
                    continue
                if local and self.device == "cpu":
                    note = (note or "") + " (CPU: very slow)"
                out[role].append(self._status(role, vspec, vspec, ok, note, local=local or provider == "ollama"))

        clip_ok = E.clip_available()
        for name in ("ViT-B-32", "ViT-B-16", "ViT-L-14"):
            out["embedder"].append(self._status("embedder", f"clip:{name}", f"CLIP {name}", clip_ok, None if clip_ok else "pip install open_clip_torch", local=True))

        te = out["text_embedder"]
        te.append(self._status("text_embedder", "openai:text-embedding-3-small", "OpenAI text-embedding-3-small", bool(s.openai_api_key), None if s.openai_api_key else "set OPENAI_API_KEY", local=False))
        te.append(self._status("text_embedder", "gemini:gemini-embedding-001", "Gemini embedding", bool(s.gemini_api_key), None if s.gemini_api_key else "set GEMINI_API_KEY", local=False))
        te.append(self._status("text_embedder", "ollama:nomic-embed-text", "Ollama nomic-embed-text", True, "requires a running Ollama server", local=True))
        st = _has("sentence_transformers")
        te.append(self._status("text_embedder", "st:all-MiniLM-L6-v2", "sentence-transformers MiniLM", st, None if st else "pip install sentence-transformers", local=True))
        te.append(self._status("text_embedder", "clip", "CLIP text tower", clip_ok, None if clip_ok else "needs CLIP", local=True))

        rid = out["reid"]
        tr = E.torchreid_available()
        rid.append(self._status("reid", "osnet:osnet_x1_0", "OSNet x1.0 (person Re-ID)", tr, None if tr else "pip install torchreid gdown", local=True))
        rid.append(self._status("reid", "clip", "CLIP image features", clip_ok, None if clip_ok else "needs CLIP", local=True))
        rid.append(self._status("reid", "color", "Colour-stripe histogram (fallback)", True, "weak; use only without OSNet/CLIP", local=True))
        return out

    def _vlm_specs(self, role: str = "vlm") -> list[str]:
        s = self.s
        specs = [
            f"openai:{s.openai_vlm_model if role == 'vlm' else s.openai_llm_model}",
            "openai:gpt-4.1",
            "openai:gpt-5-mini",
            f"anthropic:{s.anthropic_vlm_model if role == 'vlm' else s.anthropic_llm_model}",
            "anthropic:claude-sonnet-5-5" if role == "vlm" else "anthropic:claude-opus-5-5",
            f"gemini:{s.gemini_model}",
            "gemini:gemini-2.5-pro",
            "ollama:qwen2.5vl:7b" if role == "vlm" else "ollama:qwen3:8b",
            f"qwen:{s.local_vlm_model}",
            "qwen:Qwen/Qwen2.5-VL-7B-Instruct",
        ]
        if s.compat_base_url:
            specs.append("compat:default")
        seen: dict[str, None] = {}
        for x in specs:
            seen.setdefault(x, None)
        cur = self.defaults.get(role)
        if cur and cur != "none":
            seen.setdefault(cur, None)
        return list(seen)

    def capabilities(self) -> dict[str, Any]:
        def ok(role: str) -> bool:
            spec = self.defaults.get(role, "none")
            return spec != "none" and f"{role}|{spec}" not in self.errors

        det = self.defaults.get("detector", "none")
        return {
            "detector": ok("detector"),
            "open_vocab": det.split(":")[0] in ("yoloe", "yolo-world", "owlv2", "grounding-dino", "vlm") and ok("detector"),
            "vlm": ok("vlm"),
            "llm": ok("llm"),
            "agent": ok("llm"),
            "segmentation": ok("segmenter") or det.startswith("yoloe") or "-seg" in det,
            "clip": ok("embedder"),
            "text_search": ok("text_embedder"),
            "reid": self.reid_kind(),
            "video": _has("cv2") and det.split(":")[0] in ("yoloe", "yolo-world", "yolo"),
        }

    def report(self) -> dict[str, Any]:
        return {
            "device": self.device,
            "defaults": dict(self.defaults),
            "providers": self.catalog(),
            "capabilities": self.capabilities(),
        }


_registry: Optional[ModelRegistry] = None


def get_registry() -> ModelRegistry:
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry


def reset_registry(reg: Optional[ModelRegistry] = None) -> ModelRegistry:
    global _registry
    _registry = reg or ModelRegistry()
    return _registry
