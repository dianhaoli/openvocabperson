"""Deterministic stand-ins for the model roles (no weights, no network).

Test images are black canvases with solid colour blocks. ``FakeDetector`` finds each palette
colour and reports it with the palette label, ``FakeReID`` embeds a crop as its mean colour, and
``FakeVLM`` describes a crop by naming its dominant colour, so identity linking and search
behave like the real pipeline would on these synthetic scenes.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import re
from typing import Any, Optional

import numpy as np
from PIL import Image

from ovp.models.embedders import l2n
from ovp.config import get_settings
from ovp.models.registry import ROLES, normalize_spec
from ovp.types import Detection

Color = tuple[int, int, int]

RED: Color = (220, 30, 30)
GREEN: Color = (30, 200, 30)
BLUE: Color = (30, 30, 220)
YELLOW: Color = (230, 220, 40)

DEFAULT_PALETTE: dict[Color, str] = {RED: "person", GREEN: "person", BLUE: "knife", YELLOW: "dog"}
COLOR_NAMES: dict[Color, str] = {RED: "red", GREEN: "green", BLUE: "blue", YELLOW: "yellow"}


def make_image(blocks: list[tuple[Color, tuple[int, int, int, int]]], size: tuple[int, int] = (320, 240)) -> Image.Image:
    img = Image.new("RGB", size, (0, 0, 0))
    for color, box in blocks:
        img.paste(color, box)
    return img


def png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def mean_color(img: Image.Image) -> np.ndarray:
    """Mean colour of the non-background pixels."""
    arr = np.asarray(img.convert("RGB"), dtype=np.float32).reshape(-1, 3)
    fg = arr[arr.sum(1) > 60]
    return fg.mean(0) if len(fg) else np.zeros(3, np.float32)


def dominant_name(img: Image.Image) -> str:
    m = mean_color(img)
    return min(COLOR_NAMES.items(), key=lambda kv: float(np.abs(np.asarray(kv[0]) - m).sum()))[1]


class FakeDetector:
    spec = "fake:detector"
    open_vocab = True
    supports_masks = False

    def __init__(self, palette: Optional[dict[Color, str]] = None, score: float = 0.9):
        self.palette = palette or DEFAULT_PALETTE
        self.score = score
        self.calls: list[Optional[list[str]]] = []
        self.gate: Optional[asyncio.Event] = None  # when set, detect() waits for it

    async def detect(self, image: Image.Image, prompts: Optional[list[str]], conf: float) -> list[Detection]:
        self.calls.append(prompts)
        if self.gate is not None:
            await self.gate.wait()
        arr = np.asarray(image.convert("RGB"), dtype=np.int16)
        out = []
        for color, label in self.palette.items():
            if prompts is not None and label not in prompts:
                continue
            mask = (np.abs(arr - np.asarray(color, np.int16)) < 30).all(-1)
            if mask.sum() < 50:
                continue
            ys, xs = np.nonzero(mask)
            box = (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))
            out.append(Detection(box=box, score=self.score, label=label))
        return out


class FakeReID:
    spec = "fake:reid"
    space = "reid:osnet:fake"

    async def embed(self, images: list[Image.Image]) -> np.ndarray:
        return np.stack([l2n(mean_color(im) + 1e-3) for im in images]).astype(np.float32)


class FakeTextEmbedder:
    """Hashed bag of words: texts sharing words are similar."""

    spec = "fake:text"
    space = "text:fake"
    dim = 64

    async def embed_texts(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), np.float32)
        for i, t in enumerate(texts):
            for w in re.findall(r"[a-z]+", t.lower()):
                out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1.0
        return l2n(out + 1e-6)


class FakeVLM:
    """Answers every verification question with ``answer`` and names the crop's colour."""

    spec = "fake:vlm"
    supports_tools = False
    supports_vision = True

    def __init__(self, answer: Any = True, fail: bool = False):
        self.answer = answer
        self.fail = fail
        self.requests: list[list[dict]] = []

    async def complete_many(self, batch: list[list[dict]], **_: Any) -> list[str | Exception]:
        return [await self._reply(msgs) for msgs in batch]

    async def _reply(self, messages: list[dict]) -> str | Exception:
        self.requests.append(messages)
        if self.fail:
            return RuntimeError("vlm down")
        parts = messages[-1]["content"]
        text = " ".join(p["text"] for p in parts if p["type"] == "text")
        image = next(p["image"] for p in parts if p["type"] == "image")
        label = re.search(r"detected '([^']+)'", text)
        reply: dict[str, Any] = {}
        n_questions = len(re.findall(r"^\s*q\d+:", text, flags=re.M))
        if n_questions:
            reply["checks"] = [{"id": f"q{i + 1}", "answer": self.answer, "reason": "clearly visible"} for i in range(n_questions)]
        if "Describe the subject" in text:
            color = dominant_name(image)
            noun = "jacket" if label and label.group(1) == "person" else "object"
            reply["summary"] = f"A {label.group(1) if label else 'thing'} in a {color} {noun}"
            reply["clothing"] = f"{color} {noun}"
        return json.dumps(reply)

    async def complete(self, messages: list[dict], **_: Any) -> str:
        return "It looks like a test scene."


class FakeLLM:
    """Planner LLM returning a canned reply (or raising it when it is an exception)."""

    supports_tools = False

    def __init__(self, reply: str | Exception, spec: str = "fake:llm"):
        self.reply = reply
        self.spec = spec
        self.calls = 0

    async def complete(self, messages: list[dict], **_: Any) -> str:
        self.calls += 1
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


class FakeRegistry:
    """The subset of ``ModelRegistry`` the services and routes use."""

    device = "cpu"

    def __init__(self, **models: Any):
        unknown = set(models) - set(ROLES)
        if unknown:
            raise KeyError(unknown)
        self.models: dict[str, Any] = {role: models.get(role) for role in ROLES}
        self.defaults = {role: (m.spec if m is not None else "none") for role, m in self.models.items()}
        self.errors: dict[str, str] = {}

    def resolve(self, role: str, spec: Optional[str] = None) -> str:
        """Like ``ModelRegistry.resolve``: the default for None/""/"auto", else a validated spec."""
        if spec is None or spec.strip() in ("", "auto"):
            return self.defaults[role]
        if spec.strip() == self.defaults[role]:
            return self.defaults[role]
        return normalize_spec(role, spec, get_settings())

    def get(self, role: str, spec: Optional[str] = None) -> Any:
        return self.models[role]

    async def ensure(self, role: str, spec: Optional[str] = None) -> Any:
        return self.models[role]

    def capabilities(self) -> dict[str, Any]:
        return {
            "vlm": self.models["vlm"] is not None,
            "llm": self.models["llm"] is not None,
            "segmentation": False,
            "clip": self.models["embedder"] is not None,
            "text_search": True,
            "reid": "fake" if self.models["reid"] is not None else None,
            "video": False,
            "agent": False,
        }

    def report(self) -> dict[str, Any]:
        return {"device": self.device, "defaults": dict(self.defaults), "providers": {}, "capabilities": self.capabilities()}

    async def aclose(self) -> None:
        return None
