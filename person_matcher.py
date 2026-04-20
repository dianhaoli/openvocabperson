# -*- coding: utf-8 -*-
"""
Pure threshold logic for linking a Re-ID embedding to an existing person cluster.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class MatchDecision:
    """Outcome of comparing one detection embedding to DB matches."""

    status: str  # "matched" | "new" | "pending"
    best_person_id: Optional[str] = None
    best_score: Optional[float] = None
    # person to create (only meaningful for status "new" before DB insert)
    create_new_person: bool = False


def decide_match(
    top_matches: List[Tuple[str, float]],
    match_threshold: float,
    review_threshold: float,
) -> MatchDecision:
    """
    Args:
        top_matches: [(person_id, cosine_similarity), ...] sorted best-first
        match_threshold: >= this → link to existing person
        review_threshold: below this → new auto-cluster; in-between → pending

    Returns:
        MatchDecision
    """
    if not top_matches:
        return MatchDecision(status="new", best_score=None, create_new_person=True)

    best_id, best_score = top_matches[0]

    if best_score >= match_threshold:
        return MatchDecision(
            status="matched",
            best_person_id=best_id,
            best_score=best_score,
        )
    if best_score < review_threshold:
        return MatchDecision(status="new", best_score=best_score, create_new_person=True)
    return MatchDecision(
        status="pending",
        best_person_id=None,
        best_score=best_score,
    )
