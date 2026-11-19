"""Box-prompted segmentation.

Specs (weights auto-download through Ultralytics):
* ``sam:sam2.1_b.pt`` (default), ``sam:sam2.1_t.pt``, ``sam:sam2.1_l.pt``, ``sam:sam_b.pt``
* ``sam:mobile_sam.pt``  – very fast, good enough for most crops
* ``fastsam:FastSAM-s.pt`` – CNN based, fastest
* ``detector`` – use masks produced by the detector itself (YOLOE-seg), no extra model

Every detection box becomes a prompt; all boxes of an image go through the model in one call.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ..runtime import get_runtime
from ..types import Detection
from .detectors import box_iou, mask_to_polygons

log = logging.getLogger("ovp.segment")


class Segmenter:
    kind = "base"

    def __init__(self, weights: str):
        self.weights = weights

    @property
    def spec(self) -> str:
        return f"{self.kind}:{self.weights}" if self.weights else self.kind

    async def segment(self, image: Image.Image, dets: list[Detection]) -> None:
        """Fill ``polygons`` / ``mask_area`` on detections in place."""
        raise NotImplementedError


class UltralyticsSegmenter(Segmenter):
    def __init__(self, kind: str, weights: str, weights_dir: Path):
        super().__init__(weights)
        self.kind = kind
        self.weights_dir = weights_dir
        self._model: Any = None
        self._lock = threading.Lock()

    def _load(self) -> Any:
        with self._lock:
            if self._model is None:
                p = Path(self.weights)
                path = str(p) if p.exists() else (str(self.weights_dir / self.weights) if (self.weights_dir / self.weights).exists() else self.weights)
                if self.kind == "fastsam":
                    from ultralytics import FastSAM

                    self._model = FastSAM(path)
                else:
                    from ultralytics import SAM

                    self._model = SAM(path)
                log.info("loaded segmenter %s", self.spec)
            return self._model

    def unload(self) -> None:
        with self._lock:
            self._model = None

    def _segment_sync(self, image: Image.Image, dets: list[Detection]) -> None:
        if not dets:
            return
        model = self._load()
        bgr = np.ascontiguousarray(np.asarray(image.convert("RGB"))[:, :, ::-1])
        boxes = [[float(v) for v in d.box] for d in dets]
        dev = get_runtime().device
        r = model(bgr, bboxes=boxes, device="0" if dev == "cuda" else dev, verbose=False)[0]
        if r.masks is None:
            return
        masks = r.masks.data.cpu().numpy()
        mh, mw = masks.shape[1:]
        h, w = bgr.shape[:2]
        # SAM returns one mask per prompt, in prompt order. FastSAM segments everything and returns
        # the selected masks in its own order (de-duplicated), so those are matched to boxes by IoU.
        if self.kind == "sam" and len(masks) == len(dets):
            pairs = list(zip(dets, masks))
        else:
            pairs = _match_masks_to_boxes(dets, masks, w / mw, h / mh)
        for d, m in pairs:
            polys, area = mask_to_polygons(m, w / mw, h / mh)
            if polys:
                d.polygons, d.mask_area = polys, area

    async def segment(self, image: Image.Image, dets: list[Detection]) -> None:
        await get_runtime().run_model(self._segment_sync, image, dets)

    async def warmup(self) -> None:
        await get_runtime().run_model(self._load)


MIN_MASK_BOX_IOU = 0.3


def _match_masks_to_boxes(dets: list[Detection], masks: np.ndarray, sx: float, sy: float) -> list[tuple[Detection, np.ndarray]]:
    """Greedy one-to-one assignment of masks to detections by box IoU (best pairs first)."""
    mask_boxes: list[tuple[int, tuple[float, float, float, float]]] = []
    for j, m in enumerate(masks):
        ys, xs = np.nonzero(m > 0.5)
        if len(xs):
            mask_boxes.append((j, (xs.min() * sx, ys.min() * sy, (xs.max() + 1) * sx, (ys.max() + 1) * sy)))
    scored = sorted(
        ((box_iou(d.box, mb), i, j) for i, d in enumerate(dets) for j, mb in mask_boxes),
        reverse=True,
    )
    used_dets: set[int] = set()
    used_masks: set[int] = set()
    pairs: list[tuple[Detection, np.ndarray]] = []
    for iou, i, j in scored:
        if iou < MIN_MASK_BOX_IOU:
            break
        if i in used_dets or j in used_masks:
            continue
        used_dets.add(i)
        used_masks.add(j)
        pairs.append((dets[i], masks[j]))
    return pairs


class DetectorMasks(Segmenter):
    """No-op: the detector (YOLOE-seg) already produced masks."""

    kind = "detector"

    def __init__(self) -> None:
        super().__init__("")

    async def segment(self, image: Image.Image, dets: list[Detection]) -> None:
        return None


DEFAULT_SEG_WEIGHTS = {"sam": "sam2.1_b.pt", "fastsam": "FastSAM-s.pt"}


def create_segmenter(spec: str, settings: Any) -> Segmenter:
    kind, _, weights = spec.partition(":")
    kind = kind.lower()
    if kind in ("sam", "sam2", "mobilesam"):
        return UltralyticsSegmenter("sam", weights or ("mobile_sam.pt" if kind == "mobilesam" else DEFAULT_SEG_WEIGHTS["sam"]), settings.weights_dir)
    if kind == "fastsam":
        return UltralyticsSegmenter("fastsam", weights or DEFAULT_SEG_WEIGHTS["fastsam"], settings.weights_dir)
    if kind == "detector":
        return DetectorMasks()
    raise ValueError(f"unknown segmenter {spec!r}")
