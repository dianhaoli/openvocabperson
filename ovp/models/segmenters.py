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
from .detectors import mask_to_polygons

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
        # SAM returns one mask per prompt in order; FastSAM may return a different count → match by IoU
        if len(masks) == len(dets):
            pairs = list(zip(dets, masks))
        else:
            pairs = []
            for d in dets:
                best, best_iou = None, 0.0
                for m in masks:
                    ys, xs = np.nonzero(m > 0.5)
                    if len(xs) == 0:
                        continue
                    mb = (xs.min() * w / mw, ys.min() * h / mh, xs.max() * w / mw, ys.max() * h / mh)
                    from .detectors import box_iou

                    iou = box_iou(d.box, mb)
                    if iou > best_iou:
                        best, best_iou = m, iou
                if best is not None and best_iou > 0.3:
                    pairs.append((d, best))
        for d, m in pairs:
            polys, area = mask_to_polygons(m, w / mw, h / mh)
            if polys:
                d.polygons, d.mask_area = polys, area

    async def segment(self, image: Image.Image, dets: list[Detection]) -> None:
        await get_runtime().run_model(self._segment_sync, image, dets)


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
