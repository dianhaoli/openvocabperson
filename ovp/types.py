"""Plain data types shared by models, pipeline and storage."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional

import numpy as np
from PIL import Image

Box = tuple[float, float, float, float]
Polygon = list[list[float]]  # [[x, y], ...]


@dataclass
class Detection:
    box: Box
    score: float
    label: str
    polygons: Optional[list[Polygon]] = None
    mask_area: Optional[int] = None
    track_id: Optional[int] = None

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.box
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


@dataclass
class Concept:
    """One thing the user asked to find."""

    name: str
    prompts: list[str]
    kind: str = "object"  # "object" | "person_attribute"
    verify: bool = False
    question: Optional[str] = None


@dataclass
class QueryPlan:
    query: Optional[str]
    concepts: list[Concept]
    expanded: bool = False
    source: str = "default"  # llm | lexicon | literal | default

    @property
    def prompts(self) -> list[str]:
        seen: dict[str, None] = {}
        for c in self.concepts:
            for p in c.prompts:
                seen.setdefault(p, None)
        return list(seen)

    def concepts_for_label(self, label: str) -> list[Concept]:
        return [c for c in self.concepts if label in c.prompts]

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "expanded": self.expanded,
            "source": self.source,
            "concepts": [asdict(c) for c in self.concepts],
            "prompts": self.prompts,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "QueryPlan":
        return cls(
            query=d.get("query"),
            concepts=[Concept(**c) for c in d.get("concepts", [])],
            expanded=bool(d.get("expanded")),
            source=d.get("source", "default"),
        )


@dataclass
class Candidate:
    """A detection travelling through the pipeline (image detection or video track)."""

    det: Detection
    crop: Image.Image
    concept: Optional[str] = None
    stage: str = "detected"  # detected | described | low_confidence | rejected
    description: Optional[str] = None
    attributes: Optional[dict[str, Any]] = None
    verified: Optional[bool] = None
    verify_reason: Optional[str] = None
    image_embedding: Optional[np.ndarray] = None
    text_embedding: Optional[np.ndarray] = None
    reid_embedding: Optional[np.ndarray] = None
    # video
    t_start: Optional[float] = None
    t_end: Optional[float] = None
    frame_index: Optional[int] = None
    trajectory: Optional[list[list[float]]] = None
    reid_crops: list[Image.Image] = field(default_factory=list)
    entity_id: Optional[str] = None

    @property
    def is_person(self) -> bool:
        return self.det.label == "person"
