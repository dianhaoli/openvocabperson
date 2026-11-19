"""Object detectors (closed-set and open-vocabulary).

Specs:
* ``yoloe:yoloe-11l-seg.pt``      – YOLOE: open-vocabulary text prompts + instance masks (default)
* ``yolo-world:yolov8l-worldv2.pt`` – YOLO-World open-vocabulary boxes
* ``yolo:yolo11l.pt``             – classic closed-set COCO YOLO (prompts filter COCO classes)
* ``owlv2:google/owlv2-base-patch16-ensemble`` – HF zero-shot detection
* ``grounding-dino:IDEA-Research/grounding-dino-tiny`` – HF zero-shot detection
* ``vlm:<vlm spec>``                – ask a VLM for boxes (Gemini / Qwen-VL are good at this)

All local inference runs on the device thread (see ``runtime.py``).
"""

from __future__ import annotations

import logging
import math
import tempfile
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import numpy as np
from PIL import Image

from ..runtime import get_runtime
from ..types import Detection, Polygon

log = logging.getLogger("ovp.detect")

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]

COCO_SYNONYMS = {
    "people": "person", "persons": "person", "human": "person", "humans": "person", "man": "person",
    "woman": "person", "men": "person", "women": "person", "child": "person", "kid": "person", "pedestrian": "person",
    "phone": "cell phone", "mobile phone": "cell phone", "smartphone": "cell phone", "cellphone": "cell phone",
    "television": "tv", "sofa": "couch", "table": "dining table", "plant": "potted plant", "motorbike": "motorcycle",
    "bike": "bicycle", "puppy": "dog", "dogs": "dog", "cats": "cat", "kitten": "cat", "cars": "car", "ball": "sports ball",
    "bag": "handbag", "purse": "handbag", "bat": "baseball bat", "glass": "wine glass", "mug": "cup", "plane": "airplane",
}


MIN_PART_FRACTION = 0.03  # mask fragments smaller than this share of the largest part are speckle
MIN_PART_PIXELS = 16.0


def simplify_polygons(polys: Iterable[np.ndarray], max_polys: int = 4, min_points: int = 3) -> list[Polygon]:
    """Douglas-Peucker simplify, keep the largest few contours (dropping speckle), round to 1 decimal."""
    import cv2

    out: list[tuple[float, Polygon]] = []
    for p in polys:
        p = np.asarray(p, dtype=np.float32).reshape(-1, 2)
        if len(p) < min_points:
            continue
        peri = cv2.arcLength(p, True)
        approx = cv2.approxPolyDP(p, max(0.6, 0.004 * peri), True).reshape(-1, 2)
        if len(approx) < min_points:
            continue
        area = float(cv2.contourArea(approx))
        out.append((area, [[round(float(x), 1), round(float(y), 1)] for x, y in approx]))
    if not out:
        return []
    out.sort(key=lambda x: x[0], reverse=True)
    min_area = max(MIN_PART_PIXELS, out[0][0] * MIN_PART_FRACTION)
    return [p for i, (area, p) in enumerate(out[:max_polys]) if i == 0 or area >= min_area]


def mask_to_polygons(mask: np.ndarray, scale_x: float = 1.0, scale_y: float = 1.0) -> tuple[list[Polygon], int]:
    import cv2

    m = (mask > 0.5).astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    scaled = [c.reshape(-1, 2).astype(np.float32) * np.array([scale_x, scale_y], dtype=np.float32) for c in contours]
    return simplify_polygons(scaled), int(m.sum() * scale_x * scale_y)


def box_iou(a: tuple, b: tuple) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def cross_label_nms(dets: list[Detection], iou: float = 0.7, protect: Optional[Callable[[Detection], bool]] = None) -> list[Detection]:
    """Class-agnostic NMS: an object matched by several synonyms ("knife", "blade") is kept once."""
    dets = sorted(dets, key=lambda d: d.score, reverse=True)
    kept: list[Detection] = []
    for d in dets:
        if any(box_iou(d.box, k.box) > iou and (protect is None or not (protect(d) ^ protect(k))) for k in kept):
            continue
        kept.append(d)
    return kept


def _ul_device() -> str:
    d = get_runtime().device
    return "0" if d == "cuda" else d


TRACK_LOST_SECONDS = 2.0  # how long ByteTrack keeps a lost track alive; longer gaps are re-joined by Re-ID stitching


def _bytetrack_config(sample_fps: float, conf: float) -> tuple[str, float]:
    """A ByteTrack config for the *sampled* frame rate and detection threshold.

    Ultralytics' defaults assume every frame of a ~30 fps video (track_buffer=30 frames ≈ 1 s) and a
    fixed 0.25 track threshold. At 2 fps sampling the default would keep lost tracks for 15 s and
    hand a vanished person's ID to whoever appears near the predicted box. Returns the YAML path and
    the (lower) detector confidence that feeds ByteTrack's second, low-score association stage.
    """
    buffer = max(2, math.ceil(TRACK_LOST_SECONDS * sample_fps))
    low = round(max(0.05, conf * 0.4), 3)
    conf = round(conf, 3)
    path = Path(tempfile.gettempdir()) / "ovp-trackers" / f"bytetrack-b{buffer}-c{conf}.yaml"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{threading.get_ident()}.tmp")
        tmp.write_text(
            "tracker_type: bytetrack\n"
            f"track_high_thresh: {conf}\ntrack_low_thresh: {low}\nnew_track_thresh: {conf}\n"
            f"track_buffer: {buffer}\nmatch_thresh: 0.8\nfuse_score: True\n"
        )
        tmp.replace(path)
    return str(path), low


class Detector:
    kind = "base"
    open_vocab = False
    supports_masks = False
    supports_tracking = False
    local = True

    def __init__(self, weights: str):
        self.weights = weights

    @property
    def spec(self) -> str:
        return f"{self.kind}:{self.weights}"

    async def detect(self, image: Image.Image, prompts: Optional[list[str]], conf: float) -> list[Detection]:
        raise NotImplementedError

    async def warmup(self) -> None:
        pass


# ──────────────────────────────────────────────────────────────────────────────
# Ultralytics family (YOLO / YOLOE / YOLO-World): detection, masks, tracking
# ──────────────────────────────────────────────────────────────────────────────


PE_CACHE_SIZE = 2048  # YOLOE prompt embeddings kept in memory (a few KB each)


class UltralyticsDetector(Detector):
    supports_tracking = True

    def __init__(self, kind: str, weights: str, weights_dir: Path, max_det: int = 100):
        super().__init__(weights)
        self.kind = kind
        self.open_vocab = kind in ("yoloe", "yolo-world")
        self.supports_masks = kind in ("yoloe", "yolo") and "-seg" in weights
        self.weights_dir = weights_dir
        self.max_det = max_det
        self._model: Any = None
        self._classes: Optional[tuple[str, ...]] = None
        self._pe_cache: "OrderedDict[str, Any]" = OrderedDict()  # YOLOE text embedding per prompt
        self._lock = threading.RLock()

    def _path(self) -> str:
        p = Path(self.weights)
        if p.exists():
            return str(p)
        local = self.weights_dir / self.weights
        return str(local) if local.exists() else self.weights  # ultralytics auto-downloads known names

    def _load(self) -> Any:
        with self._lock:
            if self._model is None:
                from ultralytics import YOLO, YOLOE, YOLOWorld

                model_cls: type[Any] = {"yoloe": YOLOE, "yolo-world": YOLOWorld}.get(self.kind, YOLO)
                log.info("loading detector %s", self.spec)
                self._model = model_cls(self._path())
            return self._model

    def unload(self) -> None:
        """Drop the weights (the registry evicted this model); the next call reloads them."""
        with self._lock:
            self._model, self._classes = None, None
            self._pe_cache.clear()

    def _set_vocab(self, prompts: Optional[list[str]]) -> Optional[set[str]]:
        """Configure vocabulary. Returns the label whitelist for closed-set models."""
        model = self._load()
        if not self.open_vocab:
            if not prompts:
                return None
            allowed = set()
            for p in prompts:
                key = p.lower().strip()
                key = COCO_SYNONYMS.get(key, key)
                if key in COCO_CLASSES:
                    allowed.add(key)
            return allowed
        names = tuple(dict.fromkeys(p.strip() for p in prompts or () if p.strip())) or tuple(COCO_CLASSES)
        if names == self._classes:
            return None
        if self.kind == "yoloe":
            model.set_classes(list(names), self._text_embeddings(model, names))
        else:
            model.set_classes(list(names))
        self._classes = names
        return None

    def _class_ids(self, allowed: Optional[set[str]]) -> Optional[list[int]]:
        """Class indices for a closed-set label whitelist (None = all classes)."""
        if allowed is None:
            return None
        return sorted(i for i, name in self._model.names.items() if name in allowed)

    def _text_embeddings(self, model: Any, names: tuple[str, ...]) -> Any:
        """YOLOE prompt embeddings, cached per prompt so overlapping vocabularies only embed new words."""
        import torch

        missing = [n for n in names if n not in self._pe_cache]
        if missing:
            fresh = model.get_text_pe(missing)  # (1, len(missing), dim)
            for i, name in enumerate(missing):
                self._pe_cache[name] = fresh[0, i]
        for name in names:
            self._pe_cache.move_to_end(name)
        while len(self._pe_cache) > PE_CACHE_SIZE:
            self._pe_cache.popitem(last=False)
        return torch.stack([self._pe_cache[n] for n in names])[None]

    def _parse(self, r: Any, allowed: Optional[set[str]], track: bool = False) -> list[Detection]:
        dets: list[Detection] = []
        if r.boxes is None or len(r.boxes) == 0:
            return dets
        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        clss = r.boxes.cls.cpu().numpy().astype(int)
        ids = r.boxes.id.cpu().numpy().astype(int) if (track and r.boxes.id is not None) else None
        polys = r.masks.xy if (r.masks is not None) else None
        for i in range(len(boxes)):
            label = r.names[int(clss[i])]
            if allowed is not None and label not in allowed:
                continue
            d = Detection(
                box=tuple(float(v) for v in boxes[i]),  # type: ignore[arg-type]
                score=float(confs[i]),
                label=label,
                track_id=int(ids[i]) if ids is not None else None,
            )
            if polys is not None and i < len(polys) and len(polys[i]) >= 3:
                d.polygons = simplify_polygons([polys[i]])
                import cv2

                d.mask_area = int(abs(cv2.contourArea(np.asarray(polys[i], dtype=np.float32))))
            dets.append(d)
        return dets

    def _detect_sync(self, image: Image.Image, prompts: Optional[list[str]], conf: float) -> list[Detection]:
        with self._lock:
            allowed = self._set_vocab(prompts)
            if allowed is not None and not allowed:
                return []
            bgr = np.ascontiguousarray(np.asarray(image.convert("RGB"))[:, :, ::-1])
            r = self._model.predict(
                bgr, conf=conf, iou=0.5, max_det=self.max_det, classes=self._class_ids(allowed), device=_ul_device(), verbose=False
            )[0]
            return self._parse(r, allowed)

    async def detect(self, image: Image.Image, prompts: Optional[list[str]], conf: float) -> list[Detection]:
        return await get_runtime().run_model(self._detect_sync, image, prompts, conf)

    def track_frames_sync(
        self,
        frames: Iterable[tuple[int, float, np.ndarray]],
        prompts: Optional[list[str]],
        conf: float,
        on_frame: Callable[[int, float, np.ndarray, list[Detection]], None],
        should_stop: Callable[[], bool] = lambda: False,
        sample_fps: float = 30.0,
    ) -> None:
        """Run ByteTrack over sampled BGR frames. ``on_frame`` gets every frame's tracked detections.

        The tracker is re-created on the first frame of every call (``persist=False``), so no
        state leaks between videos, and the shared predictor stays usable for plain ``predict``.
        """
        with self._lock:
            allowed = self._set_vocab(prompts)
            if allowed is not None and not allowed:
                return
            model = self._load()
            classes = self._class_ids(allowed)
            tracker, low_conf = _bytetrack_config(sample_fps, conf)
            try:
                for i, (idx, t, frame) in enumerate(frames):
                    if should_stop():
                        break
                    r = model.track(
                        frame, persist=i > 0, conf=low_conf, iou=0.5, max_det=self.max_det, classes=classes,
                        tracker=tracker, device=_ul_device(), verbose=False,
                    )[0]
                    on_frame(idx, t, frame, self._parse(r, allowed, track=True))
            finally:
                predictor = getattr(model, "predictor", None)
                if predictor is not None:
                    predictor.trackers = []  # drop the finished video's track state (frames, Kalman filters)

    async def warmup(self) -> None:
        await get_runtime().run_model(self._detect_sync, Image.new("RGB", (64, 64)), None, 0.5)


# ──────────────────────────────────────────────────────────────────────────────
# Hugging Face zero-shot detectors (OWLv2 / Grounding DINO)
# ──────────────────────────────────────────────────────────────────────────────


class HFZeroShotDetector(Detector):
    open_vocab = True

    def __init__(self, kind: str, weights: str):
        super().__init__(weights)
        self.kind = kind
        self._model: Any = None
        self._processor: Any = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch  # noqa: F401
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self._processor = AutoProcessor.from_pretrained(self.weights)
        self._model = AutoModelForZeroShotObjectDetection.from_pretrained(self.weights).to(get_runtime().device).eval()

    def unload(self) -> None:
        self._model, self._processor = None, None

    def _detect_sync(self, image: Image.Image, prompts: Optional[list[str]], conf: float) -> list[Detection]:
        import torch

        self._load()
        labels = list(prompts or COCO_CLASSES)
        image = image.convert("RGB")
        if self.kind == "grounding-dino":
            text = ". ".join(labels) + "."
            inputs = self._processor(images=image, text=text, return_tensors="pt").to(self._model.device)
        else:
            inputs = self._processor(images=image, text=[labels], return_tensors="pt").to(self._model.device)
        with torch.inference_mode():
            outputs = self._model(**inputs)
        target = torch.tensor([image.size[::-1]], device=self._model.device)
        if self.kind == "grounding-dino":
            res = self._processor.post_process_grounded_object_detection(
                outputs, inputs["input_ids"], threshold=conf, text_threshold=conf, target_sizes=target
            )[0]
            names = res.get("text_labels") or res.get("labels")
        else:
            res = self._processor.post_process_grounded_object_detection(outputs, threshold=conf, target_sizes=target, text_labels=[labels])[0]
            names = res.get("text_labels") or [labels[int(i)] for i in res["labels"]]
        dets = []
        for box, score, name in zip(res["boxes"].tolist(), res["scores"].tolist(), names):
            name = str(name).strip() or "object"
            # grounding-dino may return merged phrases; snap to the closest prompt
            if name not in labels:
                name = next((l for l in labels if l in name or name in l), name)
            dets.append(Detection(box=tuple(box), score=float(score), label=name))  # type: ignore[arg-type]
        return cross_label_nms(dets, 0.8)

    async def detect(self, image: Image.Image, prompts: Optional[list[str]], conf: float) -> list[Detection]:
        return await get_runtime().run_model(self._detect_sync, image, prompts, conf)


# ──────────────────────────────────────────────────────────────────────────────
# VLM grounding (cloud or local VLM returns boxes)
# ──────────────────────────────────────────────────────────────────────────────


class VLMGroundingDetector(Detector):
    kind = "vlm"
    open_vocab = True
    local = False

    def __init__(self, vlm: Any):
        super().__init__(vlm.spec)
        self.vlm = vlm

    async def detect(self, image: Image.Image, prompts: Optional[list[str]], conf: float) -> list[Detection]:
        from .llm import extract_json, user

        labels = list(prompts or ["person"])
        w, h = image.size
        gemini = self.vlm.provider == "gemini"
        # Coordinate convention per model family: Gemini answers [ymin, xmin, ymax, xmax] on a 0-1000 grid;
        # Qwen2/2.5-VL were trained on absolute pixels of their resized input; everything else gets 0-1000.
        pixel_frame = await self.vlm.grounding_frame(image) if hasattr(self.vlm, "grounding_frame") else None
        if gemini:
            fmt = 'Return a JSON list like [{"label": "...", "box_2d": [ymin, xmin, ymax, xmax], "confidence": 0.0-1.0}] with coordinates normalized to 0-1000.'
        elif pixel_frame:
            fmt = (f'Return JSON {{"objects": [{{"label": "...", "bbox_2d": [x1, y1, x2, y2], "confidence": 0.0-1.0}}]}} '
                   f"with absolute pixel coordinates in the {pixel_frame[0]}x{pixel_frame[1]} image.")
        else:
            fmt = 'Return JSON {"objects": [{"label": "...", "box": [x1, y1, x2, y2], "confidence": 0.0-1.0}]} with coordinates normalized to 0-1000 (0,0 = top-left).'
        sx, sy = (w / pixel_frame[0], h / pixel_frame[1]) if pixel_frame else (w / 1000, h / 1000)
        prompt = (
            "Detect every instance of the following in the image: "
            + ", ".join(labels)
            + ". Use exactly one of those strings as each label. Include partially visible instances; "
            "return an empty list if none are present. " + fmt
        )
        text = await self.vlm.complete([user(image, prompt)], max_tokens=1500, json_mode=not gemini)
        data = extract_json(text)
        items = data if isinstance(data, list) else data.get("objects", data.get("detections", []))
        dets = []
        for it in items or []:
            try:
                label = str(it.get("label", "")).strip().lower() or labels[0]
                if "box_2d" in it:
                    y1, x1, y2, x2 = [float(v) for v in it["box_2d"]]
                else:
                    x1, y1, x2, y2 = [float(v) for v in it.get("bbox_2d") or it["box"]]
                box = (x1 * sx, y1 * sy, x2 * sx, y2 * sy)
                box = (max(0, min(box[0], box[2])), max(0, min(box[1], box[3])), min(w, max(box[0], box[2])), min(h, max(box[1], box[3])))
                if box[2] - box[0] < 2 or box[3] - box[1] < 2:
                    continue
                score = float(it.get("confidence", 0.6))
                if score < conf:
                    continue
                if label not in labels:
                    label = next((l for l in labels if l in label or label in l), label)
                dets.append(Detection(box=box, score=score, label=label))
            except (KeyError, TypeError, ValueError, AttributeError):
                continue
        return dets


# ──────────────────────────────────────────────────────────────────────────────
# factory
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_WEIGHTS = {
    "yoloe": "yoloe-11l-seg.pt",
    "yolo-world": "yolov8l-worldv2.pt",
    "yolo": "yolo11l.pt",
    "owlv2": "google/owlv2-base-patch16-ensemble",
    "grounding-dino": "IDEA-Research/grounding-dino-tiny",
}


def ultralytics_available() -> bool:
    try:
        import ultralytics  # noqa: F401

        return True
    except Exception:
        return False


def clip_pkg_available() -> bool:
    try:
        import clip  # noqa: F401

        return True
    except Exception:
        return False


def create_detector(spec: str, settings: Any, vlm_factory: Optional[Callable[[str], Any]] = None) -> Detector:
    kind, _, weights = spec.partition(":")
    kind = kind.lower()
    if kind in ("yoloe", "yolo-world", "yolo"):
        return UltralyticsDetector(kind, weights or DEFAULT_WEIGHTS[kind], settings.weights_dir, settings.max_detections)
    if kind in ("owlv2", "grounding-dino"):
        return HFZeroShotDetector(kind, weights or DEFAULT_WEIGHTS[kind])
    if kind == "vlm":
        if vlm_factory is None:
            raise ValueError("vlm detector needs a VLM")
        return VLMGroundingDetector(vlm_factory(weights))
    raise ValueError(f"unknown detector {spec!r}")
