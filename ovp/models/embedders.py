"""Embedding models.

* Image/text joint embeddings (CLIP / SigLIP via ``open_clip``, or OpenAI ``clip``) — enable
  text→image and image→image search over crops. Replaces the old "mean of layer-0 hidden
  states" VLM embedding, which was dominated by prompt tokens and is not a meaningful image
  representation.
* Text embeddings for VLM descriptions (OpenAI / Gemini / Ollama / any OpenAI-compatible
  ``/embeddings`` endpoint, or local sentence-transformers) — semantic search over what the
  VLM saw.
* Person Re-ID embeddings: OSNet (torchreid) → CLIP → colour-stripe histogram fallback.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any, Optional

import httpx
import numpy as np
from PIL import Image

from ..runtime import get_runtime

log = logging.getLogger("ovp.embed")


def l2n(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        return x / (np.linalg.norm(x) + 1e-8)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


# ──────────────────────────────────────────────────────────────────────────────
# CLIP-style joint embeddings
# ──────────────────────────────────────────────────────────────────────────────


class ClipEmbedder:
    """``clip:ViT-B-32`` (open_clip, pretrained=laion2b_s34b_b79k) or ``clip:ViT-B-16-SigLIP``..."""

    def __init__(self, model: str = "ViT-B-32", pretrained: Optional[str] = None):
        self.model_name = model or "ViT-B-32"
        self.pretrained = pretrained
        self._model: Any = None
        self._preprocess: Any = None
        self._tokenizer: Any = None
        self._backend = ""
        self._lock = threading.Lock()
        self.dim: Optional[int] = None

    @property
    def space(self) -> str:
        return f"clip:{self.model_name}"

    def _load(self) -> None:
        with self._lock:
            if self._model is not None:
                return
            import torch

            device = get_runtime().device
            try:
                import open_clip

                pretrained = self.pretrained or {
                    "ViT-B-32": "laion2b_s34b_b79k",
                    "ViT-B-16": "laion2b_s34b_b88k",
                    "ViT-L-14": "laion2b_s32b_b82k",
                }.get(self.model_name, None)
                if pretrained is None:
                    tags = open_clip.list_pretrained_tags_by_model(self.model_name)
                    pretrained = tags[0] if tags else None
                model, _, preprocess = open_clip.create_model_and_transforms(self.model_name, pretrained=pretrained, device=device)
                self._tokenizer = open_clip.get_tokenizer(self.model_name)
                self._backend = "open_clip"
            except ImportError:
                import clip  # OpenAI / ultralytics fork

                name = self.model_name.replace("ViT-B-32", "ViT-B/32").replace("ViT-B-16", "ViT-B/16").replace("ViT-L-14", "ViT-L/14")
                model, preprocess = clip.load(name, device=device)
                self._tokenizer = lambda texts: clip.tokenize(texts, truncate=True)
                self._backend = "clip"
            model.eval()
            self._model, self._preprocess = model, preprocess
            with torch.inference_mode():
                self.dim = int(model.encode_text(self._tokenizer(["a photo"]).to(device)).shape[-1])
            log.info("CLIP embedder %s ready (%s, dim=%s)", self.model_name, self._backend, self.dim)

    def _images_sync(self, images: list[Image.Image]) -> np.ndarray:
        import torch

        self._load()
        device = get_runtime().device
        out = []
        for i in range(0, len(images), 64):
            batch = torch.stack([self._preprocess(im.convert("RGB")) for im in images[i : i + 64]]).to(device)
            with torch.inference_mode():
                feats = self._model.encode_image(batch).float()
            out.append(feats.cpu().numpy())
        return l2n(np.concatenate(out, 0))

    def _texts_sync(self, texts: list[str]) -> np.ndarray:
        import torch

        self._load()
        with torch.inference_mode():
            feats = self._model.encode_text(self._tokenizer(texts).to(get_runtime().device)).float()
        return l2n(feats.cpu().numpy())

    async def embed_images(self, images: list[Image.Image]) -> np.ndarray:
        if not images:
            return np.zeros((0, self.dim or 512), np.float32)
        return await get_runtime().run_model(self._images_sync, images)

    async def embed_texts(self, texts: list[str]) -> np.ndarray:
        return await get_runtime().run_model(self._texts_sync, [f"a photo of {t}" if len(t.split()) <= 4 else t for t in texts])

    async def warmup(self) -> None:
        await get_runtime().run_model(self._load)


# ──────────────────────────────────────────────────────────────────────────────
# Text embeddings for descriptions
# ──────────────────────────────────────────────────────────────────────────────


class ApiTextEmbedder:
    def __init__(self, provider: str, model: str, base_url: str, api_key: Optional[str], timeout: float = 60.0):
        self.provider, self.model, self.base_url, self.api_key = provider, model, base_url.rstrip("/"), api_key
        self._client = httpx.AsyncClient(timeout=timeout)
        self.dim: Optional[int] = None

    @property
    def space(self) -> str:
        return f"text:{self.provider}:{self.model}"

    async def embed_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim or 1), np.float32)
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        vecs: list[list[float]] = []
        for i in range(0, len(texts), 256):
            chunk = [t[:8000] or " " for t in texts[i : i + 256]]
            for attempt in range(4):
                r = await self._client.post(f"{self.base_url}/embeddings", json={"model": self.model, "input": chunk}, headers=headers)
                if r.status_code == 200:
                    break
                if r.status_code in (429, 500, 502, 503) and attempt < 3:
                    await asyncio.sleep(2**attempt)
                    continue
                raise RuntimeError(f"embeddings HTTP {r.status_code}: {r.text[:300]}")
            data = sorted(r.json()["data"], key=lambda d: d["index"])
            vecs.extend(d["embedding"] for d in data)
        arr = l2n(np.asarray(vecs, np.float32))
        self.dim = arr.shape[1]
        return arr

    async def warmup(self) -> None:
        await self.embed_texts(["warmup"])


class SentenceTransformerEmbedder:
    def __init__(self, model: str = "all-MiniLM-L6-v2"):
        self.model_name = model
        self._model: Any = None
        self.dim: Optional[int] = None

    @property
    def space(self) -> str:
        return f"text:st:{self.model_name}"

    def _sync(self, texts: list[str]) -> np.ndarray:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=get_runtime().device)
        arr = self._model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        self.dim = arr.shape[1]
        return arr.astype(np.float32)

    async def embed_texts(self, texts: list[str]) -> np.ndarray:
        return await get_runtime().run_model(self._sync, texts)

    async def warmup(self) -> None:
        await self.embed_texts(["warmup"])


class ClipTextEmbedder:
    """Fallback: use the CLIP text tower for descriptions (weak but local)."""

    def __init__(self, clip: ClipEmbedder):
        self.clip = clip

    @property
    def space(self) -> str:
        return f"text:{self.clip.space}"

    @property
    def dim(self) -> Optional[int]:
        return self.clip.dim

    async def embed_texts(self, texts: list[str]) -> np.ndarray:
        return await get_runtime().run_model(self.clip._texts_sync, texts)

    async def warmup(self) -> None:
        await self.clip.warmup()


# ──────────────────────────────────────────────────────────────────────────────
# Person Re-ID
# ──────────────────────────────────────────────────────────────────────────────


class ReIDModel:
    kind = "base"
    dim = 0

    @property
    def space(self) -> str:
        return f"reid:{self.kind}"

    async def embed(self, images: list[Image.Image]) -> np.ndarray:
        raise NotImplementedError

    async def warmup(self) -> None:
        await self.embed([Image.new("RGB", (64, 128))])


class OSNetReID(ReIDModel):
    kind = "osnet"
    dim = 512

    def __init__(self, model_name: str = "osnet_x1_0"):
        self.model_name = model_name
        self._model: Any = None
        self._tf: Any = None

    @property
    def space(self) -> str:
        return f"reid:osnet:{self.model_name}"

    def _load(self) -> None:
        if self._model is not None:
            return
        from torchreid.reid.models import build_model  # type: ignore
        from torchvision import transforms as T

        device = get_runtime().device
        model = build_model(self.model_name, num_classes=1, loss="softmax", pretrained=True, use_gpu=device.startswith("cuda"))
        self._model = model.eval().to(device)
        self._tf = T.Compose([T.Resize((256, 128)), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def _sync(self, images: list[Image.Image]) -> np.ndarray:
        import torch

        self._load()
        out = []
        for i in range(0, len(images), 64):
            t = torch.stack([self._tf(im.convert("RGB")) for im in images[i : i + 64]]).to(get_runtime().device)
            with torch.inference_mode():
                out.append(self._model(t).float().cpu().numpy())
        return l2n(np.concatenate(out, 0))

    async def embed(self, images: list[Image.Image]) -> np.ndarray:
        if not images:
            return np.zeros((0, self.dim), np.float32)
        return await get_runtime().run_model(self._sync, images)


class ClipReID(ReIDModel):
    kind = "clip"

    def __init__(self, clip: ClipEmbedder):
        self.clip = clip

    async def embed(self, images: list[Image.Image]) -> np.ndarray:
        arr = await self.clip.embed_images(images)
        self.dim = arr.shape[1] if arr.size else self.dim
        return arr


class ColorReID(ReIDModel):
    """Dependency-free appearance descriptor: HSV histograms over 6 horizontal body stripes.

    It is the classic hand-crafted Re-ID baseline: robust to pose, weak against clothing
    changes. Used only when neither OSNet nor CLIP is installed.
    """

    kind = "color"
    STRIPES = 6
    dim = STRIPES * (16 + 4 + 4)

    @classmethod
    def describe(cls, img: Image.Image) -> np.ndarray:
        import cv2

        arr = np.asarray(img.convert("RGB").resize((64, 128), Image.BILINEAR))
        hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
        feats = []
        h = hsv.shape[0]
        for s in range(cls.STRIPES):
            band = hsv[s * h // cls.STRIPES : (s + 1) * h // cls.STRIPES, 8:56]  # trim background at the sides
            hh = cv2.calcHist([band], [0], None, [16], [0, 180]).ravel()
            ss = cv2.calcHist([band], [1], None, [4], [0, 256]).ravel()
            vv = cv2.calcHist([band], [2], None, [4], [0, 256]).ravel()
            v = np.concatenate([hh, ss, vv])
            feats.append(np.sqrt(v / (v.sum() + 1e-6)))
        return l2n(np.concatenate(feats))

    async def embed(self, images: list[Image.Image]) -> np.ndarray:
        if not images:
            return np.zeros((0, self.dim), np.float32)
        return await get_runtime().run_cpu(lambda: np.stack([self.describe(im) for im in images]).astype(np.float32))

    async def warmup(self) -> None:
        return None


def torchreid_available() -> bool:
    try:
        import torchreid  # noqa: F401

        return True
    except Exception:
        return False


def clip_available() -> bool:
    try:
        import open_clip  # noqa: F401

        return True
    except Exception:
        try:
            import clip  # noqa: F401

            return True
        except Exception:
            return False
