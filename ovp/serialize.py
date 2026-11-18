"""DB rows → API JSON (shapes documented in docs/API.md)."""

from __future__ import annotations

from typing import Any, Optional


def _ts(v: Any) -> Optional[float]:
    return v.timestamp() if v is not None and hasattr(v, "timestamp") else v


def _r(v: Any, nd: int = 1) -> Any:
    return round(float(v), nd) if v is not None else None


def person_ref(pid: Optional[str], label: Optional[str], watch: bool) -> Optional[dict]:
    if not pid:
        return None
    return {"id": pid, "label": label, "is_watchlist": bool(watch)}


def entity(row: Any) -> dict[str, Any]:
    traj = row["trajectory"]
    return {
        "id": row["id"],
        "media_id": row["media_id"],
        "label": row["label"],
        "concept": row["concept"],
        "score": round(float(row["score"]), 4),
        "box": [_r(row["x1"]), _r(row["y1"]), _r(row["x2"]), _r(row["y2"])],
        "polygon": row["polygon"],
        "mask_area": row["mask_area"],
        "crop_url": f"/api/entities/{row['id']}/crop" if row["crop_path"] else None,
        "description": row["description"],
        "attributes": row["attributes"],
        "verified": row["verified"],
        "verify_reason": row["verify_reason"],
        "stage": row["stage"],
        "track_id": row["track_id"],
        "t_start": _r(row["t_start"], 2),
        "t_end": _r(row["t_end"], 2),
        "frame_index": row["frame_index"],
        "trajectory": traj,
        "person": person_ref(row["person_id"], row["person_label"], row["person_is_watchlist"]),
        "match_status": row["match_status"],
        "match_score": round(float(row["match_score"]), 4) if row["match_score"] is not None else None,
        "suggested_person": person_ref(row["suggested_person_id"], row["suggested_label"], row["suggested_is_watchlist"]),
        "created_at": _ts(row["created_at"]),
    }


def media(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "kind": row["kind"],
        "name": row["name"],
        "width": row["width"],
        "height": row["height"],
        "duration": _r(row["duration"], 2),
        "fps": _r(row["fps"], 2),
        "url": f"/api/media/{row['id']}/file",
        "thumb_url": f"/api/media/{row['id']}/thumb",
        "query": row["query"],
        "status": row["status"],
        "entity_count": int(row["entity_count"] or 0),
        "person_count": int(row["person_count"] or 0),
        "created_at": _ts(row["created_at"]),
    }


def media_detail(row: Any, entities: list[Any]) -> dict[str, Any]:
    return {**media(row), "plan": row["plan"], "entities": [entity(e) for e in entities]}


def person(row: Any) -> dict[str, Any]:
    rep = row["representative_entity_id"]
    return {
        "id": row["id"],
        "label": row["label"],
        "is_watchlist": bool(row["is_watchlist"]),
        "notes": row["notes"],
        "sighting_count": int(row["sighting_count"] or 0),
        "media_count": int(row["media_count"] or 0),
        "representative_entity_id": rep,
        "thumb_url": f"/api/entities/{rep}/crop" if rep else None,
        "first_seen": _ts(row["first_seen"]),
        "last_seen": _ts(row["last_seen"]),
        "created_at": _ts(row["created_at"]),
        "updated_at": _ts(row["updated_at"]),
    }


def compact_entity(e: dict[str, Any]) -> dict[str, Any]:
    """Small version for LLM tool results (no polygons/trajectories)."""
    out = {k: e.get(k) for k in ("id", "media_id", "label", "concept", "score", "description", "verified", "t_start", "t_end")}
    if e.get("person"):
        out["person"] = e["person"]
    if e.get("attributes"):
        out["attributes"] = e["attributes"]
    return {k: v for k, v in out.items() if v is not None}
