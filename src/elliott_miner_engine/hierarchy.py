from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from .models import ScanResult, WaveCandidate
from .mtf import DEFAULT_INTERVAL_WEIGHTS

DEGREE_ORDER = ["cycle", "primary", "intermediate", "minor", "minute"]
INTERVAL_TO_DEGREE = {
    "3mo": "cycle",
    "1mo": "primary",
    "1wk": "intermediate",
    "1d": "minor",
    "4h": "minute",
    "1h": "minute",
}


@dataclass(slots=True)
class DegreeView:
    degree: str
    interval: str
    symbol: str
    direction: Optional[str]
    pattern_type: Optional[str]
    score: Optional[float]
    confidence: Optional[float]
    invalidation: Optional[float]
    target_price: Optional[float]
    target_zone_low: Optional[float]
    target_zone_high: Optional[float]
    pivot_timestamps: List[object] = field(default_factory=list)
    pivot_prices: List[float] = field(default_factory=list)
    labels: List[str] = field(default_factory=list)
    inherited_direction_ok: bool = True
    inherited_pattern_ok: bool = True
    notes: List[str] = field(default_factory=list)


@dataclass(slots=True)
class DegreeHierarchySummary:
    symbol: str
    market: str
    dominant_direction: Optional[str]
    dominant_pattern: Optional[str]
    agreement_score: float
    inherited_alignment_score: float
    state: str
    degree_views: List[DegreeView] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


_ROMAN_5 = ["i", "ii", "iii", "iv", "v"]
_ROMAN_ABC = ["a", "b", "c", "d", "e"]


def degree_for_interval(interval: str) -> str:
    return INTERVAL_TO_DEGREE.get(interval, interval)


def _degree_rank(degree: str) -> int:
    try:
        return DEGREE_ORDER.index(degree)
    except ValueError:
        return len(DEGREE_ORDER)


def _candidate(result: ScanResult) -> Optional[WaveCandidate]:
    return result.best_candidate


def _label_set(candidate: Optional[WaveCandidate], degree: str) -> List[str]:
    if candidate is None:
        return []
    if candidate.pattern_type in {"impulse", "ending_diagonal"}:
        base = ["1", "2", "3", "4", "5"]
        if degree == "primary":
            return [f"({x})" for x in base]
        if degree == "intermediate":
            return base
        return _ROMAN_5
    base = ["A", "B", "C"] if candidate.pattern_type in {"zigzag", "flat"} else ["A", "B", "C", "D", "E"]
    if degree == "primary":
        return [f"({x})" for x in base]
    if degree == "intermediate":
        return base
    return _ROMAN_ABC[: len(base)]


def _extract_degree_view(result: ScanResult) -> DegreeView:
    c = _candidate(result)
    degree = degree_for_interval(result.interval)
    labels = _label_set(c, degree)
    pivot_ts = [] if c is None else list(c.pivot_timestamps[1 : 1 + len(labels)])
    pivot_prices = [] if c is None else list(c.pivot_prices[1 : 1 + len(labels)])
    return DegreeView(
        degree=degree,
        interval=result.interval,
        symbol=result.symbol,
        direction=None if c is None else c.direction,
        pattern_type=None if c is None else c.pattern_type,
        score=None if c is None else float(c.score),
        confidence=None if c is None else float(c.confidence),
        invalidation=None if c is None or c.invalidation is None else float(c.invalidation),
        target_price=None if c is None else c.meta.get("primary_price_target"),
        target_zone_low=None if c is None else c.meta.get("primary_price_zone_low"),
        target_zone_high=None if c is None else c.meta.get("primary_price_zone_high"),
        pivot_timestamps=pivot_ts,
        pivot_prices=[float(x) for x in pivot_prices],
        labels=labels,
        notes=[] if c is None else list(c.momentum_notes[:2]),
    )


def build_degree_hierarchy(results: List[ScanResult]) -> DegreeHierarchySummary:
    if not results:
        raise ValueError("No results supplied")
    views = [_extract_degree_view(r) for r in results]
    views = sorted(views, key=lambda v: (_degree_rank(v.degree), -DEFAULT_INTERVAL_WEIGHTS.get(v.interval, 0.0)))

    weighted = []
    for v in views:
        if v.direction and v.confidence is not None:
            weighted.append((v.direction, DEFAULT_INTERVAL_WEIGHTS.get(v.interval, 0.15) * v.confidence))
    direction_votes: Dict[str, float] = {}
    for direction, weight in weighted:
        direction_votes[direction] = direction_votes.get(direction, 0.0) + float(weight)
    dominant_direction = max(direction_votes, key=direction_votes.get) if direction_votes else None

    pattern_votes: Dict[str, float] = {}
    for v in views:
        if v.pattern_type and v.direction == dominant_direction and v.confidence is not None:
            pattern_votes[v.pattern_type] = pattern_votes.get(v.pattern_type, 0.0) + DEFAULT_INTERVAL_WEIGHTS.get(v.interval, 0.15) * v.confidence
    dominant_pattern = max(pattern_votes, key=pattern_votes.get) if pattern_votes else None

    top_direction = None
    top_pattern = None
    inheritance_hits = 0.0
    inheritance_total = 0.0
    notes: List[str] = []
    for v in views:
        weight = DEFAULT_INTERVAL_WEIGHTS.get(v.interval, 0.15)
        if top_direction is not None and v.direction is not None:
            v.inherited_direction_ok = v.direction == top_direction
            inheritance_total += weight
            inheritance_hits += weight if v.inherited_direction_ok else 0.0
            if not v.inherited_direction_ok:
                v.notes.append(f"{v.interval} direction conflicts with higher-degree {top_direction} anchor.")
        if top_pattern is not None and v.pattern_type is not None:
            v.inherited_pattern_ok = (v.pattern_type == top_pattern) or (v.direction != top_direction)
            if not v.inherited_pattern_ok:
                v.notes.append(f"{v.interval} pattern diverges from higher-degree {top_pattern} anchor.")
        if v.direction is not None and top_direction is None:
            top_direction = v.direction
        if v.pattern_type is not None and top_pattern is None:
            top_pattern = v.pattern_type

    total_w = sum(DEFAULT_INTERVAL_WEIGHTS.get(v.interval, 0.15) for v in views if v.direction is not None) or 1.0
    agree_w = sum(DEFAULT_INTERVAL_WEIGHTS.get(v.interval, 0.15) for v in views if v.direction == dominant_direction and v.direction is not None)
    agreement_score = float(agree_w / total_w)
    inherited_alignment_score = float(inheritance_hits / inheritance_total) if inheritance_total > 0 else agreement_score

    if agreement_score >= 0.8 and inherited_alignment_score >= 0.75:
        state = "hierarchical_alignment"
    elif agreement_score >= 0.6:
        state = "partial_alignment"
    else:
        state = "degree_conflict"

    if dominant_direction:
        notes.append(f"Dominant direction is {dominant_direction}, led by the highest selected degree.")
    if dominant_pattern:
        notes.append(f"Dominant pattern bias is {dominant_pattern}.")
    if state == "degree_conflict":
        notes.append("Higher and lower degrees disagree materially. Treat the lower-degree count as fragile until it realigns or invalidates.")
    elif state == "hierarchical_alignment":
        notes.append("Higher and lower degrees broadly agree. This usually produces more stable Elliott labeling than single-timeframe counting.")

    return DegreeHierarchySummary(
        symbol=results[0].symbol,
        market=results[0].market,
        dominant_direction=dominant_direction,
        dominant_pattern=dominant_pattern,
        agreement_score=agreement_score,
        inherited_alignment_score=inherited_alignment_score,
        state=state,
        degree_views=views,
        notes=notes,
    )


def hierarchy_frame(summary: DegreeHierarchySummary) -> pd.DataFrame:
    rows = []
    for v in summary.degree_views:
        rows.append(
            {
                "degree": v.degree,
                "interval": v.interval,
                "direction": v.direction,
                "pattern": v.pattern_type,
                "score": v.score,
                "confidence": v.confidence,
                "inherited_direction_ok": v.inherited_direction_ok,
                "inherited_pattern_ok": v.inherited_pattern_ok,
                "invalidation": v.invalidation,
                "target": v.target_price,
                "zone_low": v.target_zone_low,
                "zone_high": v.target_zone_high,
            }
        )
    return pd.DataFrame(rows)
