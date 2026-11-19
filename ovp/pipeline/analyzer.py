"""The analysis pipeline.

image ─► plan (LLM / lexicon) ─► open-vocab detect (+masks) ─► cross-prompt NMS
       ─► route ─┬─ VLM: verify abstract concepts + describe (one JSON call per crop, concurrent/batched)
                 ├─ segmentation (SAM, box prompts, one call per image)          ┐ run
                 ├─ CLIP crop embeddings                                       ├ concurrently
                 └─ person Re-ID embeddings                                    ┘
       ─► text embeddings of descriptions ─► one DB transaction ─► identity linking ─► events

video ─► sampled frames ─► detector + ByteTrack ─► per-track aggregation (trajectory, best crop,
         Re-ID crops) ─► track stitching (same person re-entering) ─► same crop pipeline as images
"""

from __future__ import annotations

import asyncio
import logging
import math
import queue
import re
import threading
import time
import weakref
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Optional, Sequence, Union

import numpy as np
from PIL import Image

from .. import serialize
from ..config import get_settings
from ..models.detectors import COCO_CLASSES, box_iou, cross_label_nms
from ..models.embedders import l2n
from ..models.llm import extract_json, user
from ..models.registry import ModelRegistry
from ..services.identity import IdentityService, presence_segments, segments_overlap
from ..storage.db import Database, new_id
from ..storage.files import FileStore
from ..types import Candidate, Concept, Detection, QueryPlan
from .planner import QueryPlanner

log = logging.getLogger("ovp.pipeline")

Emit = Callable[..., None]
Counts = dict[str, int]  # {"accepted", "rejected", "updated"}

DEFAULT_PLAN_CONF = 0.2  # recall for the default "everything" vocabulary
QUERY_CONF = 0.15  # open-vocabulary text prompts score low (YOLOE often 0.1-0.25); verification filters the rest


class Cancelled(Exception):
    pass


@dataclass
class AnalyzeOptions:
    query: Optional[str] = None
    expand: Union[bool, str] = "auto"
    verify: Union[bool, str] = "auto"
    segment: bool = False
    describe: str = "persons"  # persons | all | none
    detector: Optional[str] = None
    segmenter: Optional[str] = None
    vlm: Optional[str] = None
    llm: Optional[str] = None
    conf: Optional[float] = None
    video_fps: Optional[float] = None
    max_frames: Optional[int] = None
    reid: bool = True

    @classmethod
    def from_dict(cls, d: Optional[dict[str, Any]]) -> "AnalyzeOptions":
        d = dict(d or {})
        known = {k: d[k] for k in cls.__dataclass_fields__ if k in d}
        for k in ("expand", "verify"):
            v = known.get(k)
            if isinstance(v, str) and v.lower() in ("true", "false"):
                known[k] = v.lower() == "true"
        opts = cls(**known)
        if opts.describe not in ("persons", "all", "none"):
            opts.describe = "persons"
        return opts

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ──────────────────────────────────────────────────────────────────────────────
# prompts
# ──────────────────────────────────────────────────────────────────────────────

PERSON_FIELDS = {
    "summary": "one sentence describing the person and what they are doing",
    "actions": "visible actions/posture, <= 8 words",
    "clothing": "clothing and colours, <= 12 words",
    "held_objects": "objects held or interacted with, or 'none'",
    "appearance": "visible hair, accessories, bag, glasses etc., <= 10 words",
}
OBJECT_FIELDS = {
    "summary": "one sentence describing the object, its state and context",
    "color": "main colours",
    "details": "notable visible details, <= 12 words",
}


def build_crop_prompt(label: str, questions: list[str], describe: bool) -> str:
    lines = [
        f"This image is a crop around a detected '{label}'.",
        "Answer ONLY from what is clearly visible. Do not guess identity. If something is not visible, say 'unclear'.",
    ]
    schema: dict[str, Any] = {}
    if questions:
        lines.append("Answer each yes/no question about the main subject of the crop:")
        for i, q in enumerate(questions):
            lines.append(f'  q{i + 1}: {q}')
        schema["checks"] = [{"id": "q1", "answer": "true|false", "reason": "short visual evidence"}]
    if describe:
        fields = PERSON_FIELDS if label == "person" else OBJECT_FIELDS
        lines.append("Describe the subject with these fields:")
        for k, v in fields.items():
            lines.append(f"  {k}: {v}")
            schema[k] = "..."
    import json as _json

    lines.append("Reply with JSON only, shaped like: " + _json.dumps(schema))
    return "\n".join(lines)


_CHECK_ID = re.compile(r"q\d+")


def parse_crop_reply(text: str, n_questions: int, describe: bool) -> tuple[list[tuple[Optional[bool], str]], Optional[str], Optional[dict]]:
    checks: list[tuple[Optional[bool], str]] = [(None, "")] * n_questions
    summary, attrs = None, None
    try:
        data = extract_json(text)
        if not isinstance(data, dict):
            raise ValueError("not an object")
    except Exception:
        data = None
    if data is None:
        if describe and text and text.strip():
            summary = text.strip()[:600]
        if n_questions:
            low = (text or "").strip().lower()
            ans = True if low.startswith("yes") else False if low.startswith("no") else None
            checks = [(ans, (text or "")[:200])] * n_questions
        return checks, summary, attrs
    raw_checks = data.get("checks")
    if isinstance(raw_checks, list):
        items = [(str(c.get("id", f"q{i + 1}")) if isinstance(c, dict) else f"q{i + 1}", c) for i, c in enumerate(raw_checks)]
    elif isinstance(raw_checks, dict):
        items = [(str(k), v) for k, v in raw_checks.items()]
    else:  # answers at the top level: {"q1": true, ...}
        items = [(k, v) for k, v in data.items() if _CHECK_ID.fullmatch(k)]
    answers = [(k, v if isinstance(v, dict) else {"answer": v}) for k, v in items]
    by_id = dict(answers)
    for i in range(n_questions):
        c = by_id.get(f"q{i + 1}") or (answers[i][1] if i < len(answers) else None)
        if c is None:
            continue
        a = c.get("answer")
        if isinstance(a, str):
            a = {"true": True, "yes": True, "false": False, "no": False}.get(a.strip().lower())
        checks[i] = (bool(a) if a is not None else None, str(c.get("reason") or "")[:300])
    if describe:
        summary = data.get("summary")
        summary = str(summary).strip() if summary else None
        attrs = {k: str(v).strip() for k, v in data.items() if k not in ("checks", "summary") and not _CHECK_ID.fullmatch(k) and v not in (None, "", [], {})}
        attrs = {k: v for k, v in attrs.items() if v.lower() not in ("unclear", "none", "n/a")} or None
        if not summary and attrs:  # fields without the summary line still describe the subject
            summary = "; ".join(f"{k.replace('_', ' ')}: {v}" for k, v in attrs.items())
    return checks, summary, attrs


def expand_box(box: tuple, w: int, h: int, ratio: float) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    dx, dy = (x2 - x1) * ratio / 2, (y2 - y1) * ratio / 2
    return (max(0, int(x1 - dx)), max(0, int(y1 - dy)), min(w, int(math.ceil(x2 + dx))), min(h, int(math.ceil(y2 + dy))))


# ──────────────────────────────────────────────────────────────────────────────
# video tracking aggregation
# ──────────────────────────────────────────────────────────────────────────────


REID_CROPS_PER_TRACK = 8
MAX_TRAJECTORY_POINTS = 240
TRACK_LOST_SECONDS = 2.0  # how long ByteTrack keeps a lost track alive; longer gaps are re-joined by Re-ID stitching
BORDER_PX = 2  # a box edge this close to the frame edge is cut off by it
BORDER_PENALTY = 0.5  # crop quality factor per clipped side: a partly visible body makes a poor crop / Re-ID sample


def _crop_quality(d: Detection, w: int, h: int) -> float:
    """Confidence x size of a tracked detection, discounted for each side the frame border clips."""
    x1, y1, x2, y2 = d.box
    clipped = (x1 <= BORDER_PX) + (y1 <= BORDER_PX) + (x2 >= w - BORDER_PX) + (y2 >= h - BORDER_PX)
    return d.score * math.sqrt(max(d.area, 1.0)) * BORDER_PENALTY**clipped


@dataclass
class _Track:
    tid: int
    labels: dict[str, int] = field(default_factory=dict)
    hits: list[tuple[float, tuple, float, int]] = field(default_factory=list)  # (t, box, score, frame)
    best_q: float = -1.0
    best_crop: Optional[Image.Image] = None
    best_box: Optional[tuple] = None
    best_frame: int = 0
    best_polys: Any = None
    reid: list[tuple[float, Image.Image]] = field(default_factory=list)  # (quality, tight crop), best first

    @property
    def label(self) -> str:
        return max(self.labels.items(), key=lambda x: x[1])[0]

    @property
    def t_start(self) -> float:
        return self.hits[0][0]

    @property
    def t_end(self) -> float:
        return self.hits[-1][0]

    def add(self, idx: int, t: float, frame: np.ndarray, d: Detection, ratio: float) -> None:
        """Record one tracked detection; keeps the best crop and a few good Re-ID crops."""
        self.labels[d.label] = self.labels.get(d.label, 0) + 1
        self.hits.append((round(t, 3), tuple(round(v, 1) for v in d.box), d.score, idx))
        q = _crop_quality(d, frame.shape[1], frame.shape[0])
        if q > self.best_q:
            self.best_q, self.best_box, self.best_frame, self.best_polys = q, d.box, idx, d.polygons
            self.best_crop = _crop_bgr(frame, d.box, ratio)
        # every 3rd hit (consecutive frames look alike), keeping the highest-quality crops
        if d.label == "person" and len(self.hits) % 3 == 1 and (len(self.reid) < REID_CROPS_PER_TRACK or q > self.reid[-1][0]):
            self.reid.append((q, _crop_bgr(frame, d.box, 0.0)))
            self.reid.sort(key=lambda x: -x[0])
            del self.reid[REID_CROPS_PER_TRACK:]


def _crop_bgr(frame: np.ndarray, box: tuple, ratio: float) -> Image.Image:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = expand_box(box, w, h, ratio)
    return Image.fromarray(np.ascontiguousarray(frame[y1:y2, x1:x2, ::-1]))


def _presence(c: Candidate) -> list[tuple[float, float]]:
    return presence_segments(c.trajectory, c.t_start, c.t_end)


def _is_person(d: Detection) -> bool:
    return d.label == "person"


def _same_box(c: Candidate, e: Any) -> bool:
    """Whether a new image detection duplicates a stored entity (same label, nearly the same box)."""
    return bool(e["label"] == c.det.label and box_iou(c.det.box, (e["x1"], e["y1"], e["x2"], e["y2"])) > 0.7)


def _split_existing(cands: list[Candidate], existing: Optional[list[Any]], same: Callable[[Candidate, Any], bool]) -> tuple[list[Candidate], list[tuple[Candidate, Any]]]:
    """Split a scan's candidates into new ones and ``(candidate, stored entity)`` duplicates (one per entity)."""
    new: list[Candidate] = []
    matched: list[tuple[Candidate, Any]] = []
    taken: set[str] = set()
    for c in cands:
        e = next((e for e in existing or () if same(c, e)), None)
        if e is None:
            new.append(c)
        elif e["id"] not in taken:
            taken.add(e["id"])
            matched.append((c, e))
    return new, matched


def _same_track(c: Candidate, e: Any) -> bool:
    """Whether a new track duplicates a stored video entity (same label, overlapping in time and space)."""
    if e["label"] != c.det.label or e["t_start"] is None or c.t_start is None:
        return False
    if c.t_end < e["t_start"] or e["t_end"] < c.t_start:  # type: ignore[operator]
        return False
    return box_iou(c.det.box, (e["x1"], e["y1"], e["x2"], e["y2"])) > 0.5


class _ByteTracker:
    """ByteTrack over per-frame detections. The state lives here rather than in the detector, so the
    device thread is free between frames and any detector can track."""

    def __init__(self, conf: float, sample_fps: float):
        from ultralytics.trackers.byte_tracker import BYTETracker
        from ultralytics.utils import IterableSimpleNamespace

        self.low = round(max(0.05, conf * 0.4), 3)  # detection threshold: feeds ByteTrack's low-score association
        # the lost-track buffer is in *sampled* frames: at 2 fps the default (30) would keep a vanished track for 15 s
        args = IterableSimpleNamespace(
            tracker_type="bytetrack", track_high_thresh=conf, track_low_thresh=self.low, new_track_thresh=conf,
            track_buffer=max(2, math.ceil(TRACK_LOST_SECONDS * sample_fps)), match_thresh=0.8, fuse_score=True,
        )
        self._tracker = BYTETracker(args)
        self._classes: dict[str, int] = {}

    def update(self, dets: list[Detection], shape: tuple[int, int]) -> list[Detection]:
        """Feed one frame; returns its tracked detections with ``track_id`` set."""
        from ultralytics.engine.results import Boxes

        rows = [[*d.box, d.score, self._classes.setdefault(d.label, len(self._classes))] for d in dets]
        out = []
        for row in self._tracker.update(Boxes(np.asarray(rows, np.float32).reshape(-1, 6), shape)):
            d = dets[int(row[-1])]
            d.track_id = int(row[4])
            out.append(d)
        return out


@dataclass
class _Job:
    """VLM work for one candidate."""

    cand: Candidate
    verify: list[Concept]
    describe: bool
    verify_only: bool  # only verification concepts claim it: denied by all checks → rejected


@dataclass
class _VideoInfo:
    fps: float
    frame_count: int  # 0 when the container does not say
    width: int
    height: int
    poster: Image.Image

    @property
    def duration(self) -> Optional[float]:
        return self.frame_count / self.fps if self.frame_count else None


def _probe_video(path: Path) -> Optional[_VideoInfo]:
    import cv2

    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return None
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not (0 < fps <= 240):  # missing or bogus (VFR webm can report 1000)
            fps = 25.0
        n = max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0))
        ok, first = cap.read()
        if not ok:
            return None
        h, w = first.shape[:2]
        return _VideoInfo(fps=fps, frame_count=n, width=w, height=h, poster=_bgr_to_image(first))
    finally:
        cap.release()


_END = object()


class _FrameReader:
    """Decodes every ``step``-th frame on a background thread so decoding overlaps inference.

    Iterates ``(frame_index, seconds, bgr)``; timestamps come from the container (correct for
    variable frame rate). Skipped frames are only ``grab``-bed, never converted. After the
    iteration, ``reached_end`` / ``frame_count`` / ``duration`` describe the whole file (useful when
    the container does not state its length, e.g. browser-recorded WebM).
    """

    def __init__(self, path: Path, fps: float, step: int, max_frames: int, depth: int = 4):
        self.reached_end = False
        self.frame_count = 0
        self.duration = 0.0
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=depth)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, args=(path, fps, step, max_frames), name="ovp-video-decode", daemon=True)
        self._thread.start()

    def _put(self, item: Any) -> bool:
        while not self._stop.is_set():
            try:
                self._queue.put(item, timeout=0.2)
                return True
            except queue.Full:
                continue
        return False

    def _run(self, path: Path, fps: float, step: int, max_frames: int) -> None:
        import cv2

        cap = cv2.VideoCapture(str(path))
        try:
            idx = sampled = 0
            while sampled < max_frames:
                if idx % step:
                    ok = cap.grab()
                else:
                    ok, frame = cap.read()
                if not ok:
                    self.reached_end = True
                    break
                t = _frame_time(cap, idx, fps)
                self.frame_count, self.duration = idx + 1, t + 1.0 / fps
                if idx % step == 0:
                    if not self._put((idx, t, frame)):
                        return
                    sampled += 1
                idx += 1
        except Exception as e:  # surfaced to the consumer
            self._put(e)
        finally:
            cap.release()
            self._put(_END)

    def __iter__(self) -> Iterator[tuple[int, float, np.ndarray]]:
        while True:
            try:
                item = self._queue.get(timeout=0.2)
            except queue.Empty:
                if self._stop.is_set():  # closed early (e.g. the awaiting task was cancelled): never block the reading thread
                    return
                continue
            if item is _END:
                return
            if isinstance(item, Exception):
                raise item
            yield item

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)


def _frame_time(cap: Any, idx: int, fps: float) -> float:
    """Seconds of the frame just read: container timestamp (VFR-safe), else index / fps."""
    import cv2

    pos = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
    return pos if (pos > 0 or idx == 0) else idx / fps


def _bgr_to_image(frame: np.ndarray) -> Image.Image:
    return Image.fromarray(np.ascontiguousarray(frame[:, :, ::-1]))


def _read_frames(path: Path, indices: list[int]) -> list[tuple[int, float, np.ndarray]]:
    """Read specific frames as ``(index, seconds, bgr)``. Seeks when the container is indexed;
    otherwise (e.g. browser-recorded WebM without cues) decodes forward, which is always exact."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = fps if 0 < fps <= 240 else 25.0
        wanted = sorted(set(indices))
        out: list[tuple[int, float, np.ndarray]] = []
        if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0:
            for idx in wanted:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ok, frame = cap.read()
                if ok:
                    out.append((idx, _frame_time(cap, idx, fps), frame))
            return out
        idx = 0
        for target in wanted:
            while idx < target and cap.grab():
                idx += 1
            if idx < target:
                break
            ok, frame = cap.read()
            if not ok:
                break
            out.append((idx, _frame_time(cap, idx, fps), frame))
            idx += 1
        return out
    finally:
        cap.release()


# ──────────────────────────────────────────────────────────────────────────────
# analyzer
# ──────────────────────────────────────────────────────────────────────────────


class Analyzer:
    def __init__(self, db: Database, store: FileStore, registry: ModelRegistry, planner: QueryPlanner, identity: IdentityService):
        self.db, self.store, self.reg, self.planner, self.identity = db, store, registry, planner, identity
        self.s = get_settings()
        self._media_locks: "weakref.WeakValueDictionary[str, asyncio.Lock]" = weakref.WeakValueDictionary()

    # ── planning ──────────────────────────────────────────────────────────────
    async def make_plan(self, opts: AnalyzeOptions) -> QueryPlan:
        llm = None
        if opts.query and opts.expand is not False:
            llm = await self.reg.ensure("llm", opts.llm) or await self.reg.ensure("vlm", opts.vlm)
        return await self.planner.plan(opts.query, opts.expand, llm)

    def _detector_supports(self, det: Any, plan: QueryPlan) -> list[str]:
        """Prompts the detector can handle (closed-set detectors only know COCO)."""
        if getattr(det, "open_vocab", False) or plan.source == "default":
            return plan.prompts
        return [p for p in plan.prompts if p in COCO_CLASSES]

    async def _models(self, opts: AnalyzeOptions, plan: QueryPlan) -> dict[str, Any]:
        det_spec = opts.detector
        det = await self.reg.ensure("detector", det_spec)
        if det is None:
            raise RuntimeError("no detector available – install ultralytics (and CLIP for open-vocabulary) or configure a VLM detector")
        needs_open_vocab = plan.source != "default" and any(p not in COCO_CLASSES for p in plan.prompts)
        if needs_open_vocab and not det.open_vocab and not det_spec:
            # the default detector is closed-set (COCO) but the query is not: switch to an open-vocab one
            for alt in self.reg.open_vocab_fallbacks():
                cand = await self.reg.ensure("detector", alt)
                if cand is not None:
                    log.info("query needs open-vocabulary detection: using %s instead of %s", cand.spec, det.spec)
                    det = cand
                    break
        need_vlm = opts.describe != "none" or any(c.verify for c in plan.concepts) or opts.verify is True
        vlm = await self.reg.ensure("vlm", opts.vlm) if need_vlm else None
        seg = None
        if opts.segment:
            spec = opts.segmenter
            if spec in (None, "", "auto"):
                spec = "detector" if getattr(det, "supports_masks", False) else self.reg.defaults.get("segmenter")
            seg = await self.reg.ensure("segmenter", spec) if spec and spec != "none" else None
        return {
            "detector": det,
            "vlm": vlm,
            "segmenter": seg,
            "clip": await self.reg.ensure("embedder"),
            "text": await self.reg.ensure("text_embedder"),
            "reid": await self.reg.ensure("reid") if opts.reid else None,
        }

    # ── image ─────────────────────────────────────────────────────────────────
    async def analyze_image(self, img: Image.Image, name: str, opts: AnalyzeOptions, plan: QueryPlan, emit: Emit, cancel: Optional[threading.Event] = None) -> str:
        models = await self._models(opts, plan)  # an invalid model spec fails before anything is stored
        media_id = new_id("m")
        await self.store.save_image_original(media_id, img)
        await self.store.save_thumb(media_id, img)
        await self.db.insert_media(
            id=media_id, kind="image", name=name, path=str(self.store.media_path(media_id, ".jpg")), thumb_path=str(self.store.thumb_path(media_id)),
            width=img.width, height=img.height, query=opts.query, plan=plan.to_dict(), options=opts.to_dict(), status="processing",
        )
        emit("media_created", media=serialize.media(await self.db.get_media(media_id)))
        try:
            await self._image_entities(media_id, img, models, opts, plan, emit, cancel, existing=None)
            await self.db.update_media(media_id, status="done")
        except Cancelled:
            await self.db.update_media(media_id, status="cancelled")
            raise
        except BaseException:
            await self.db.update_media(media_id, status="error")
            raise
        return media_id

    def _conf(self, opts: AnalyzeOptions, plan: QueryPlan) -> float:
        """Detection threshold: the user's, else a recall-oriented default (lower for queries)."""
        if opts.conf is not None:
            return opts.conf
        return min(self.s.detection_conf, DEFAULT_PLAN_CONF if plan.source == "default" else QUERY_CONF)

    async def _detect_image(self, img: Image.Image, models: dict[str, Any], opts: AnalyzeOptions, plan: QueryPlan) -> list[Detection]:
        det = models["detector"]
        prompts = self._detector_supports(det, plan)
        if not prompts:
            return []
        dets = await det.detect(img, prompts if plan.source != "default" or det.open_vocab else None, self._conf(opts, plan))
        return cross_label_nms(dets, self.s.nms_iou, protect=_is_person)

    async def _image_entities(
        self, media_id: str, img: Image.Image, models: dict[str, Any], opts: AnalyzeOptions, plan: QueryPlan, emit: Emit, cancel: Optional[threading.Event], existing: Optional[list[Any]],
    ) -> Counts:
        emit("stage", media_id=media_id, stage="detect", message=f"Detecting {len(plan.prompts)} prompt(s) with {models['detector'].spec}")
        t0 = time.perf_counter()
        dets = await self._detect_image(img, models, opts, plan)
        if not opts.segment:
            for d in dets:
                d.polygons, d.mask_area = None, None
        w, h = img.size
        cands = [Candidate(det=d, crop=img.crop(expand_box(d.box, w, h, self.s.expand_ratio))) for d in dets]
        cands, matched = _split_existing(cands, existing, _same_box)
        emit("detections", media_id=media_id, count=len(cands), labels=_count(c.det.label for c in cands), elapsed=round(time.perf_counter() - t0, 2))
        segment_task = None
        seg = models["segmenter"]
        if seg is not None and seg.kind != "detector" and cands:
            emit("stage", media_id=media_id, stage="segment", message=f"Segmenting {len(cands)} object(s) with {seg.spec}")
            segment_task = asyncio.create_task(seg.segment(img, [c.det for c in cands]))
        try:
            return await self._process(media_id, cands, opts, plan, models, emit, cancel, extra=[segment_task] if segment_task else [], matched=matched)
        finally:
            if segment_task is not None and not segment_task.done():  # cancelled before the stages were awaited
                segment_task.cancel()
                await asyncio.gather(segment_task, return_exceptions=True)

    # ── shared crop processing ────────────────────────────────────────────────
    def _route(self, c: Candidate, plan: QueryPlan, opts: AnalyzeOptions, vlm: Any) -> _Job:
        """Assign the candidate's concept (and low-confidence stage) and decide its VLM work."""
        concepts = plan.concepts_for_label(c.det.label) if plan.source != "default" else []
        plain = [k for k in concepts if not k.verify]
        to_verify = [k for k in concepts if k.verify] if opts.verify is not False else []
        if opts.verify is True and not to_verify:
            to_verify = [Concept(name=k.name, prompts=k.prompts, verify=True, question=f"Is this really a {c.det.label} (part of '{k.name}')?") for k in plain]
            plain = []
        claimed = plain or to_verify or concepts
        c.concept = claimed[0].name if claimed else None  # verified matches may re-assign it
        if c.is_person and c.det.score < self.s.low_conf:
            c.stage = "low_confidence"
        # without a VLM we cannot verify: keep candidates, verification stays unknown
        if vlm is None:
            to_verify = []
        if plain:
            to_verify = [k for k in to_verify if k.kind == "person_attribute"] if c.is_person else []
        describe = opts.describe == "all" or (opts.describe == "persons" and c.is_person)
        describe = describe and c.stage != "low_confidence" and vlm is not None
        if c.crop.width * c.crop.height < 20 * 20:
            describe, to_verify = False, []
        return _Job(c, to_verify, describe, verify_only=bool(to_verify) and not plain)

    async def _run_vlm(self, media_id: str, vlm: Any, jobs: list[_Job], emit: Emit) -> None:
        """Verify + describe crops in one request each; verify-only candidates that every check denies are rejected."""
        if not jobs:
            return
        emit("stage", media_id=media_id, stage="describe", message=f"{vlm.spec}: {sum(1 for j in jobs if j.verify)} verification(s), {sum(1 for j in jobs if j.describe)} description(s)")
        batch = [[user(j.cand.crop, build_crop_prompt(j.cand.det.label, [k.question or f"Is this {k.name}?" for k in j.verify], j.describe))] for j in jobs]
        t = time.perf_counter()
        replies = await vlm.complete_many(batch, max_tokens=self.s.vlm_max_tokens, json_mode=True)
        failures = 0
        for j, reply in zip(jobs, replies):
            c = j.cand
            if isinstance(reply, Exception):
                failures += 1
                continue
            checks, summary, attrs = parse_crop_reply(reply, len(j.verify), j.describe)
            yes = [(k, r) for k, (a, r) in zip(j.verify, checks) if a]
            if yes:
                c.verified, c.verify_reason = True, yes[0][1] or None
                c.concept = yes[0][0].name
            elif j.verify_only and any(a is not None for a, _ in checks):
                # only verification concepts claimed this detection and they said no: kept (with the checked concept) but filtered out
                c.verified = False
                c.verify_reason = next((r for a, r in checks if r), None)
                c.stage = "rejected"
            if j.describe and summary:
                c.description, c.attributes = summary, attrs
                if c.stage == "detected":
                    c.stage = "described"
        if failures:
            emit("log", level="warning", message=f"{failures} VLM request(s) failed")
        emit("log", level="info", message=f"VLM finished {len(jobs)} crop(s) in {time.perf_counter() - t:.1f}s")

    async def _process(
        self, media_id: str, cands: list[Candidate], opts: AnalyzeOptions, plan: QueryPlan, models: dict[str, Any], emit: Emit,
        cancel: Optional[threading.Event], extra: Sequence[asyncio.Task] = (), matched: Sequence[tuple[Candidate, Any]] = (),
    ) -> Counts:
        """Run the crop stages on new candidates and save them; re-check stored entities a scan detected again (``matched``)."""

        def check_cancel() -> None:
            if cancel is not None and cancel.is_set():
                raise Cancelled()

        vlm = models["vlm"]
        # 1) routing: concepts, low-confidence persons; stored entities are only verified again (no description)
        jobs = [self._route(c, plan, opts, vlm) for c in cands]
        rechecks = [(self._route(c, plan, opts, vlm), e) for c, e in matched]
        for j, _ in rechecks:
            j.describe = False

        # cap the number of VLM calls (largest, most confident first)
        vlm_jobs = [j for j in jobs + [j for j, _ in rechecks] if j.verify or j.describe]
        if len(vlm_jobs) > self.s.max_vlm_crops:
            vlm_jobs.sort(key=lambda j: (bool(j.verify), j.cand.det.score * math.sqrt(j.cand.det.area)), reverse=True)
            emit("log", level="warning", message=f"{len(vlm_jobs) - self.s.max_vlm_crops} crop(s) not sent to the VLM (OVP_MAX_VLM_CROPS={self.s.max_vlm_crops})")
            vlm_jobs = vlm_jobs[: self.s.max_vlm_crops]
        check_cancel()

        async def run_clip() -> None:
            clip = models["clip"]
            if clip is None or not cands:
                return
            try:
                vecs = await clip.embed_images([c.crop for c in cands])
                for c, v in zip(cands, vecs):
                    c.image_embedding = v
            except Exception as e:
                emit("log", level="warning", message=f"CLIP embedding failed: {e}")

        async def run_reid() -> None:
            reid = models["reid"]
            people = [c for c in cands if c.is_person and c.stage != "low_confidence" and c.reid_embedding is None]  # stitched tracks already have one
            if reid is None or not people:
                return
            emit("stage", media_id=media_id, stage="embed", message=f"Re-ID embeddings for {len(people)} person crop(s)")
            try:
                flat: list[Image.Image] = []
                owners: list[int] = []
                for i, c in enumerate(people):
                    crops = c.reid_crops or [c.crop]
                    flat += crops
                    owners += [i] * len(crops)
                vecs = await reid.embed(flat)
                for i, c in enumerate(people):
                    sel = vecs[[k for k, o in enumerate(owners) if o == i]]
                    c.reid_embedding = l2n(sel.mean(0))
            except Exception as e:
                emit("log", level="warning", message=f"Re-ID failed: {e}")

        results = await asyncio.gather(self._run_vlm(media_id, vlm, vlm_jobs, emit), run_clip(), run_reid(), *extra, return_exceptions=True)
        for r in results:
            if isinstance(r, Exception):
                if isinstance(r, Cancelled):
                    raise r
                log.exception("pipeline stage failed", exc_info=r)
                emit("log", level="error", message=f"stage failed: {r}")
        check_cancel()

        # rejected and low-confidence candidates are kept (for transparency) but excluded from identity linking
        for c in cands:
            if c.stage in ("rejected", "low_confidence"):
                c.reid_embedding = None
        if cands:
            await self._save(media_id, cands, models, emit)
        n_rejected = sum(1 for c in cands if c.stage == "rejected")
        counts = {"accepted": len(cands) - n_rejected, "rejected": n_rejected, "updated": 0}
        for key, n in (await self._apply_rechecks(rechecks, models, emit)).items():
            counts[key] += n
        return counts

    async def _save(self, media_id: str, cands: list[Candidate], models: dict[str, Any], emit: Emit) -> None:
        """Text-embed descriptions, store the candidates in one transaction, link person sightings to identities."""
        temb = models["text"]
        described = [c for c in cands if c.description]
        if temb is not None and described:
            try:
                texts = []
                for c in described:
                    extra_t = " ".join(f"{k}: {v}." for k, v in (c.attributes or {}).items())
                    texts.append(f"{c.det.label}. {c.description} {extra_t}".strip())
                vecs = await temb.embed_texts(texts)
                for c, v in zip(described, vecs):
                    c.text_embedding = v
            except Exception as e:
                emit("log", level="warning", message=f"text embedding failed: {e}")

        n_rejected = sum(1 for c in cands if c.stage == "rejected")
        emit("stage", media_id=media_id, stage="save", message=f"Saving {len(cands)} entities" + (f" ({n_rejected} rejected by verification)" if n_rejected else ""))
        rows, embs = [], []
        for c in cands:
            c.entity_id = new_id("e")
        await self.store.save_crops([(c.entity_id, c.crop) for c in cands])  # type: ignore[misc]
        for c in cands:
            x1, y1, x2, y2 = c.det.box
            rows.append({
                "id": c.entity_id, "media_id": media_id, "label": c.det.label, "concept": c.concept, "score": float(c.det.score),
                "x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2),
                "polygon": c.det.polygons, "mask_area": c.det.mask_area, "crop_path": str(self.store.crop_path(c.entity_id)),  # type: ignore[arg-type]
                "description": c.description, "attributes": c.attributes, "verified": c.verified, "verify_reason": c.verify_reason,
                "stage": c.stage, "track_id": c.det.track_id, "t_start": c.t_start, "t_end": c.t_end, "frame_index": c.frame_index,
                "trajectory": c.trajectory,
            })
            if c.image_embedding is not None and models["clip"] is not None:
                embs.append((c.entity_id, models["clip"].space, c.image_embedding))
            if c.text_embedding is not None:
                embs.append((c.entity_id, temb.space, c.text_embedding))
            if c.reid_embedding is not None:
                embs.append((c.entity_id, models["reid"].space, c.reid_embedding))
        await self.db.insert_entities(rows, embs)  # type: ignore[arg-type]

        people = [(c.entity_id, c.reid_embedding) for c in cands if c.reid_embedding is not None]
        if people and models["reid"] is not None:
            emit("stage", media_id=media_id, stage="identity", message=f"Linking {len(people)} person sighting(s)")
            conflicts = None
            if any(c.t_start is not None for c in cands):
                presence = {c.entity_id: _presence(c) for c in cands}

                def conflicts(a: str, b: str) -> bool:  # two tracks visible at the same time are two people
                    return segments_overlap(presence[a], presence[b])

            await self.identity.assign_media(people, models["reid"].space, conflicts=conflicts)  # type: ignore[arg-type]

        for row in await self.db.get_entities([c.entity_id for c in cands]):  # type: ignore[misc]
            emit("entity", entity=serialize.entity(row))

    async def _apply_rechecks(self, rechecks: list[tuple[_Job, Any]], models: dict[str, Any], emit: Emit) -> Counts:
        """Re-label / re-verify stored entities a scan detected again.

        * verified for the scan's concept: ``accepted``, stored with that concept as verified;
        * claimed by a plain (unverified) concept: ``accepted`` and re-labelled, unless the entity is
          verified for another concept, which an unverified label never replaces;
        * denied by the VLM for the concept it is stored with: ``rejected``, and stored as rejected
          (``verified`` false with the reason), as a new detection would be; a person sighting is
          also unlinked from its person, since rejected entities are never identity-linked;
        * denied for another concept: ``rejected`` but left untouched, since it is still a valid
          detection that just does not match this query. Unknown verification changes nothing.

        A stored person sighting that was rejected and is accepted again is linked to a person.
        ``updated`` counts the stored entities that changed.
        """
        counts = {"accepted": 0, "rejected": 0, "updated": 0}
        changed_ids = []
        unlink: list[str] = []
        relink: list[tuple[Candidate, Any]] = []
        for j, e in rechecks:
            c = j.cand
            if c.concept is None:  # the scan's plan has no concept for this label (e.g. the default vocabulary)
                continue
            fields: dict[str, Any]
            if c.stage == "rejected":
                counts["rejected"] += 1
                if e["concept"] != c.concept or e["stage"] == "rejected":
                    continue
                fields = {"verified": False, "verify_reason": c.verify_reason, "stage": "rejected"}
            else:
                if c.verified:
                    fields = {"concept": c.concept, "verified": True, "verify_reason": c.verify_reason}
                elif j.verify_only:
                    continue
                elif e["concept"] == c.concept or (e["verified"] and e["stage"] != "rejected"):
                    fields = {}  # already labelled, or verified for a concept an unverified label never replaces
                else:  # the old verdict was about the old concept
                    fields = {"concept": c.concept, "verified": None, "verify_reason": None}
                if e["stage"] == "rejected":  # rejected for an earlier query's concept, matches this one
                    fields.update(concept=c.concept, stage="described" if e["description"] else "detected")
                    fields.setdefault("verified", None)
                    fields.setdefault("verify_reason", None)
                counts["accepted"] += 1
            changes = {k: v for k, v in fields.items() if e[k] != v}
            if changes:
                await self.db.update_entity(e["id"], **changes)
                changed_ids.append(e["id"])
                if e["label"] == "person" and changes.get("stage") == "rejected" and (e["person_id"] or e["match_status"]):
                    unlink.append(e["id"])
                elif e["label"] == "person" and e["stage"] == "rejected" and "stage" in changes:
                    relink.append((c, e))
        counts["updated"] = len(changed_ids)
        for eid in unlink:
            await self.identity.unlink(eid)
        await self._relink(relink, models["reid"], emit)
        for row in await self.db.get_entities(changed_ids) if changed_ids else []:
            emit("entity", entity=serialize.entity(row))
        return counts

    async def _relink(self, items: list[tuple[Candidate, Any]], reid: Any, emit: Emit) -> None:
        """Link stored person sightings that a scan accepted again after an earlier rejection: their
        Re-ID embedding is the stored one, else one computed from this scan's crops."""
        if reid is None or not items:
            return
        people: list[tuple[str, np.ndarray]] = []
        missing: list[tuple[str, Candidate]] = []
        for c, e in items:
            vec = await self.db.get_embedding(e["id"], reid.space)
            if vec is None and c.reid_embedding is not None:
                vec = c.reid_embedding
            if vec is None:
                missing.append((e["id"], c))
            else:
                people.append((e["id"], vec))
        if missing:
            try:
                for eid, c in missing:
                    people.append((eid, l2n((await reid.embed(c.reid_crops or [c.crop])).mean(0))))
            except Exception as ex:
                emit("log", level="warning", message=f"Re-ID failed: {ex}")
        await self.db.upsert_embeddings([(eid, reid.space, v) for eid, v in people])
        presence = {e["id"]: presence_segments(e["trajectory"], e["t_start"], e["t_end"]) for _, e in items}
        video = any(e["t_start"] is not None for _, e in items)

        def conflicts(a: str, b: str) -> bool:
            return not video or segments_overlap(presence[a], presence[b])

        await self.identity.assign_media(people, reid.space, conflicts=conflicts)

    # ── video ─────────────────────────────────────────────────────────────────
    async def analyze_video(
        self, path: Path, name: str, opts: AnalyzeOptions, plan: QueryPlan, emit: Emit, cancel: Optional[threading.Event] = None, move: bool = False,
    ) -> str:
        """Analyse a video file; ``move`` moves it into storage instead of copying (temporary uploads)."""
        from ..runtime import get_runtime

        info = await get_runtime().run_cpu(_probe_video, path)
        if info is None:
            raise RuntimeError(f"cannot decode video {name}")
        models = await self._models(opts, plan)  # an invalid model spec fails before anything is stored
        media_id = new_id("m")
        stored = await self.store.import_file(media_id, path, move=move)
        await self.store.save_thumb(media_id, info.poster)
        await self.db.insert_media(
            id=media_id, kind="video", name=name, path=str(stored), thumb_path=str(self.store.thumb_path(media_id)), width=info.width, height=info.height,
            duration=info.duration, fps=info.fps, frame_count=info.frame_count, query=opts.query, plan=plan.to_dict(), options=opts.to_dict(), status="processing",
        )
        emit("media_created", media=serialize.media(await self.db.get_media(media_id)))
        try:
            await self._video_entities(media_id, stored, info.fps, info.frame_count, models, opts, plan, emit, cancel, existing=None)
            await self.db.update_media(media_id, status="done")
        except Cancelled:
            await self.db.update_media(media_id, status="cancelled")
            raise
        except BaseException:
            await self.db.update_media(media_id, status="error")
            raise
        return media_id

    async def _video_entities(
        self, media_id: str, path: Path, fps: float, n: int, models: dict[str, Any], opts: AnalyzeOptions, plan: QueryPlan, emit: Emit,
        cancel: Optional[threading.Event], existing: Optional[list[Any]],
    ) -> Counts:
        from ..runtime import get_runtime

        det = models["detector"]
        prompts = self._detector_supports(det, plan)
        if not prompts:
            emit("detections", media_id=media_id, count=0, labels={}, frames=0)
            return {"accepted": 0, "rejected": 0, "updated": 0}
        sample_fps = min(fps, opts.video_fps or self.s.video_fps)
        max_frames = max(1, opts.max_frames or self.s.max_frames)
        step = max(1, round(fps / max(0.1, sample_fps)))
        if n and math.ceil(n / step) > max_frames:
            step = math.ceil(n / max_frames)
        total = min(math.ceil(n / step), max_frames) if n else None
        tracker = _ByteTracker(self._conf(opts, plan), fps / step)
        det_prompts = prompts if (det.open_vocab or plan.source != "default") else None
        ratio = self.s.expand_ratio
        tracks: dict[int, _Track] = {}
        sampled = 0
        last_emit = 0.0

        emit("stage", media_id=media_id, stage="track", message=f"Tracking {len(prompts)} prompt(s) at ~{fps / step:.1f} fps with {det.spec}")
        t0 = time.perf_counter()
        reader = _FrameReader(path, fps, step, max_frames)
        frames = iter(reader)
        try:
            while (item := await get_runtime().run_cpu(next, frames, None)) is not None:
                if cancel is not None and cancel.is_set():
                    raise Cancelled()
                idx, t, frame = item
                # one device call per frame (tracker state lives here), so interactive model calls never wait for a whole video
                dets = await det.detect(_bgr_to_image(frame), det_prompts, tracker.low)
                for d in tracker.update(cross_label_nms(dets, self.s.nms_iou, protect=_is_person), frame.shape[:2]):
                    tracks.setdefault(d.track_id, _Track(d.track_id)).add(idx, t, frame, d, ratio)  # type: ignore[arg-type]
                sampled += 1
                now = time.monotonic()
                if now - last_emit > 0.5:
                    last_emit = now
                    msg = f"Tracking frame {sampled}" + (f"/{total}" if total else "") + f" · {len(tracks)} track(s)"
                    emit("stage", media_id=media_id, stage="track", message=msg, frames=sampled, total=total)
        finally:
            reader.close()
        if not n and reader.reached_end:  # the container did not state its length; now we know it
            await self.db.update_media(media_id, frame_count=reader.frame_count, duration=round(reader.duration, 3))
        min_hits = self.s.min_track_hits if sampled >= 6 else 1
        good = [t for t in tracks.values() if len(t.hits) >= min_hits and t.best_crop is not None]

        cands: list[Candidate] = []
        for tr in good:
            d = Detection(box=tr.best_box, score=float(np.mean([hh[2] for hh in tr.hits])), label=tr.label, polygons=tr.best_polys if opts.segment else None, track_id=tr.tid)  # type: ignore[arg-type]
            c = Candidate(det=d, crop=tr.best_crop)  # type: ignore[arg-type]
            c.t_start, c.t_end, c.frame_index = tr.t_start, tr.t_end, tr.best_frame
            c.trajectory = _thin_traj([[hh[0], *hh[1]] for hh in tr.hits], MAX_TRAJECTORY_POINTS)
            c.reid_crops = [im for _, im in tr.reid]
            cands.append(c)
        cands, matched = _split_existing(cands, existing, _same_track)
        emit("detections", media_id=media_id, count=len(cands), labels=_count(c.det.label for c in cands), frames=sampled, elapsed=round(time.perf_counter() - t0, 2))
        cands = await self._stitch(cands, models)
        return await self._process(media_id, cands, opts, plan, models, emit, cancel, matched=matched)

    async def _stitch(self, cands: list[Candidate], models: dict[str, Any]) -> list[Candidate]:
        """Merge person tracks that are the same person re-entering (never visible at the same time).

        The person candidates keep the Re-ID embeddings computed here; the main stage reuses them.
        """
        reid = models["reid"]
        people = [c for c in cands if c.is_person and c.reid_crops]
        if reid is None or len(people) < 2:
            return cands
        vecs = await reid.embed([im for c in people for im in c.reid_crops])
        k = 0
        for c in people:
            c.reid_embedding = l2n(vecs[k : k + len(c.reid_crops)].mean(0))
            k += len(c.reid_crops)
        th = self.identity.thresholds(reid.space).match + 0.03  # stricter than cross-media linking: merging is not reviewable
        people.sort(key=lambda c: c.t_start or 0.0)
        merged: set[int] = set()
        for i, a in enumerate(people):
            if id(a) in merged:
                continue
            for b in people[i + 1 :]:
                if id(b) in merged or segments_overlap(_presence(a), _presence(b)):
                    continue
                if float(a.reid_embedding @ b.reid_embedding) >= th:  # type: ignore[operator]
                    a.trajectory = _thin_traj(sorted((a.trajectory or []) + (b.trajectory or [])), MAX_TRAJECTORY_POINTS)
                    a.t_start, a.t_end = min(a.t_start, b.t_start), max(a.t_end, b.t_end)  # type: ignore[type-var]
                    if b.det.score * math.sqrt(b.det.area) > a.det.score * math.sqrt(a.det.area):
                        a.det, a.crop, a.frame_index = Detection(box=b.det.box, score=b.det.score, label=b.det.label, polygons=b.det.polygons, track_id=a.det.track_id), b.crop, b.frame_index
                    a.reid_crops = (a.reid_crops + b.reid_crops)[:12]
                    a.reid_embedding = l2n(a.reid_embedding + b.reid_embedding)  # type: ignore[operator]
                    merged.add(id(b))
        return [c for c in cands if id(c) not in merged]

    # ── library scan ──────────────────────────────────────────────────────────
    async def scan_media(self, media_id: str, opts: AnalyzeOptions, plan: QueryPlan, emit: Emit, cancel: Optional[threading.Event] = None) -> Counts:
        """Look for the plan's concepts in stored media: new detections are added, stored entities detected
        again are re-labelled / re-verified (see ``_apply_rechecks``). Scans of one media are serialised."""
        async with self._media_lock(media_id):
            m = await self.db.get_media(media_id)
            if m is None:
                raise KeyError(media_id)
            models = await self._models(opts, plan)
            existing = await self.db.media_entities(media_id)
            if m["kind"] == "video":
                return await self._video_entities(media_id, Path(m["path"]), m["fps"] or 25.0, m["frame_count"] or 0, models, opts, plan, emit, cancel, existing)
            img = await self.store.open_image(m["path"])
            return await self._image_entities(media_id, img, models, opts, plan, emit, cancel, existing)

    def _media_lock(self, media_id: str) -> asyncio.Lock:
        lock = self._media_locks.get(media_id)
        if lock is None:
            lock = self._media_locks[media_id] = asyncio.Lock()
        return lock

    # ── Q&A ───────────────────────────────────────────────────────────────────
    async def ask_entity(self, entity_id: str, question: str, scope: str = "crop", vlm_spec: Optional[str] = None) -> str:
        e = await self.db.get_entity(entity_id)
        if e is None:
            raise KeyError(entity_id)
        vlm = await self.reg.ensure("vlm", vlm_spec)
        if vlm is None:
            raise RuntimeError("no VLM configured")
        prior = e["description"] or "none"
        if scope == "scene":
            m = await self.db.get_media(e["media_id"])
            img = await self._scene_image(m, e["frame_index"])
            from PIL import ImageDraw

            img = img.copy()
            ImageDraw.Draw(img).rectangle([e["x1"], e["y1"], e["x2"], e["y2"]], outline=(255, 0, 0), width=max(2, img.width // 300))
            prompt = (
                f"The {e['label']} of interest is outlined with a red box. Prior description: {prior}.\n"
                f"Question: {question}\nAnswer from visible evidence in the image; consider the whole scene. If unsure, say so. Be concise."
            )
            parts = [img, prompt]
        else:
            crop = await self.store.open_image(e["crop_path"])
            prompt = (
                f"This crop shows a detected {e['label']}. Prior description: {prior}.\n"
                f"Question: {question}\nAnswer only from what is visible in the crop. If it is not clearly visible, say so. Be concise."
            )
            parts = [crop, prompt]
        answer = await vlm.complete([user(*parts)], max_tokens=400)
        await self.db.add_qa(e["media_id"], entity_id, scope, question, answer)
        return answer

    async def ask_media(self, media_id: str, question: str, vlm_spec: Optional[str] = None) -> str:
        m = await self.db.get_media(media_id)
        if m is None:
            raise KeyError(media_id)
        vlm = await self.reg.ensure("vlm", vlm_spec)
        if vlm is None:
            raise RuntimeError("no VLM configured")
        if m["kind"] == "video":
            frames = await self._video_frames(m, 6)
            parts: list[Any] = []
            for t, im in frames:
                parts += [f"Frame at {t:.1f}s:", im]
            parts.append(f"These are frames sampled from a {m['duration'] or 0:.0f}s video.\nQuestion: {question}\nAnswer from visible evidence; cite timestamps when useful. Be concise.")
        else:
            img = await self.store.open_image(m["path"])
            parts = [img, f"Question: {question}\nAnswer from visible evidence in the image. If unsure, say so. Be concise."]
        answer = await vlm.complete([user(*parts)], max_tokens=600)
        await self.db.add_qa(media_id, None, "media", question, answer)
        return answer

    async def _scene_image(self, m: Any, frame_index: Optional[int]) -> Image.Image:
        if m["kind"] == "video":
            from ..runtime import get_runtime

            frames = await get_runtime().run_cpu(_read_frames, Path(m["path"]), [int(frame_index or 0)])
            if not frames:
                raise RuntimeError("cannot read frame")
            return _bgr_to_image(frames[0][2])
        return await self.store.open_image(m["path"])

    async def _video_frames(self, m: Any, k: int) -> list[tuple[float, Image.Image]]:
        """``k`` frames spread evenly over the video (the first ``k`` seconds if its length is unknown)."""
        from ..runtime import get_runtime

        n, fps = m["frame_count"] or 0, m["fps"] or 25.0
        indices = [int((i + 0.5) * n / k) for i in range(k)] if n else [int(i * fps) for i in range(k)]
        out = []
        for _, t, frame in await get_runtime().run_cpu(_read_frames, Path(m["path"]), indices):
            im = _bgr_to_image(frame)
            im.thumbnail((768, 768))
            out.append((t, im))
        return out


def _count(labels: Any) -> dict[str, int]:
    out: dict[str, int] = {}
    for l in labels:
        out[l] = out.get(l, 0) + 1
    return out


def _thin_traj(rows: list[list[float]], max_points: int) -> list[list[float]]:
    if len(rows) <= max_points:
        return rows
    idx = np.linspace(0, len(rows) - 1, max_points).round().astype(int)
    return [rows[i] for i in sorted(set(idx.tolist()))]
