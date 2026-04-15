from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from .models import ScanResult, WaveCandidate


DEFAULT_INTERVAL_WEIGHTS: Dict[str, float] = {
    "1mo": 0.45,
    "1wk": 0.35,
    "1d": 0.20,
    "1h": 0.10,
    "4h": 0.12,
}


@dataclass(slots=True)
class TimeframeView:
    interval: str
    result: ScanResult
    weight: float
    direction: Optional[str]
    pattern_type: Optional[str]
    score: Optional[float]
    confidence: Optional[float]
    invalidation: Optional[float]
    price_target: Optional[float]
    price_zone_low: Optional[float]
    price_zone_high: Optional[float]
    time_window_start: Optional[object]
    time_window_end: Optional[object]


@dataclass(slots=True)
class MultiTimeframeSummary:
    symbol: str
    market: str
    intervals: List[str]
    consensus_direction: Optional[str]
    consensus_pattern: Optional[str]
    alignment_score: float
    confidence_score: float
    conflict_score: float
    state: str
    invalidation: Optional[float]
    target_price: Optional[float]
    target_zone_low: Optional[float]
    target_zone_high: Optional[float]
    target_time_window_start: Optional[object]
    target_time_window_end: Optional[object]
    timeframe_views: List[TimeframeView] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


def _interval_weight(interval: str, custom_weights: Optional[Dict[str, float]] = None) -> float:
    if custom_weights and interval in custom_weights:
        return float(custom_weights[interval])
    return float(DEFAULT_INTERVAL_WEIGHTS.get(interval, 0.15))


def _candidate_from_result(result: ScanResult) -> Optional[WaveCandidate]:
    return result.best_candidate if result.best_candidate is not None else None


def _safe_meta(candidate: Optional[WaveCandidate], key: str) -> Optional[float]:
    if candidate is None:
        return None
    value = candidate.meta.get(key)
    return None if value is None else float(value)


def _time_window(candidate: Optional[WaveCandidate]) -> tuple[Optional[object], Optional[object]]:
    if candidate is None or not candidate.fib_time_targets:
        return None, None
    top = sorted(candidate.fib_time_targets, key=lambda t: (-t.weight, t.projected_index))[:3]
    pts = [t.projected_timestamp for t in top if t.projected_timestamp is not None]
    if not pts:
        return None, None
    return min(pts), max(pts)


def build_timeframe_view(result: ScanResult, weight: float) -> TimeframeView:
    c = _candidate_from_result(result)
    tws, twe = _time_window(c)
    return TimeframeView(
        interval=result.interval,
        result=result,
        weight=weight,
        direction=None if c is None else c.direction,
        pattern_type=None if c is None else c.pattern_type,
        score=None if c is None else float(c.score),
        confidence=None if c is None else float(c.confidence),
        invalidation=None if c is None or c.invalidation is None else float(c.invalidation),
        price_target=_safe_meta(c, 'primary_price_target'),
        price_zone_low=_safe_meta(c, 'primary_price_zone_low'),
        price_zone_high=_safe_meta(c, 'primary_price_zone_high'),
        time_window_start=tws,
        time_window_end=twe,
    )


def _weighted_vote(items: Iterable[tuple[Optional[str], float]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key, weight in items:
        if not key or weight <= 0:
            continue
        out[key] = out.get(key, 0.0) + float(weight)
    return out


def _weighted_average(values: Iterable[tuple[Optional[float], float]]) -> Optional[float]:
    pairs = [(float(v), float(w)) for v, w in values if v is not None and w > 0]
    if not pairs:
        return None
    num = sum(v * w for v, w in pairs)
    den = sum(w for _, w in pairs)
    if den <= 0:
        return None
    return num / den


def reconcile_results(results: List[ScanResult], custom_weights: Optional[Dict[str, float]] = None) -> MultiTimeframeSummary:
    if not results:
        raise ValueError('No results supplied for reconciliation.')

    views = [build_timeframe_view(r, _interval_weight(r.interval, custom_weights)) for r in results]
    total_weight = sum(v.weight for v in views) or 1.0

    direction_votes = _weighted_vote((v.direction, v.weight * (v.confidence or 0.0)) for v in views)
    consensus_direction = max(direction_votes, key=direction_votes.get) if direction_votes else None

    pattern_votes = _weighted_vote(
        (v.pattern_type, v.weight * (v.confidence or 0.0))
        for v in views
        if consensus_direction is None or v.direction == consensus_direction
    )
    consensus_pattern = max(pattern_votes, key=pattern_votes.get) if pattern_votes else None

    aligned_views = [
        v for v in views
        if v.direction is not None and v.direction == consensus_direction
    ] if consensus_direction else []

    aligned_weight = sum(v.weight for v in aligned_views)
    alignment_score = float(aligned_weight / total_weight) if total_weight > 0 else 0.0
    confidence_score = _weighted_average((v.confidence, v.weight) for v in aligned_views) or 0.0
    conflict_score = float(max(0.0, 1.0 - alignment_score))

    if alignment_score >= 0.8 and confidence_score >= 0.6:
        state = 'high_alignment'
    elif alignment_score >= 0.6:
        state = 'moderate_alignment'
    elif alignment_score > 0.0:
        state = 'mixed'
    else:
        state = 'no_consensus'

    invalidation = None
    if aligned_views:
        invalidation = _weighted_average((v.invalidation, v.weight) for v in aligned_views)
        if invalidation is None:
            strongest = max(aligned_views, key=lambda v: (v.weight, v.confidence or 0.0))
            invalidation = strongest.invalidation

    target_price = _weighted_average((v.price_target, v.weight * (v.confidence or 0.0)) for v in aligned_views)
    target_zone_low = _weighted_average((v.price_zone_low, v.weight * (v.confidence or 0.0)) for v in aligned_views)
    target_zone_high = _weighted_average((v.price_zone_high, v.weight * (v.confidence or 0.0)) for v in aligned_views)

    starts = [pd.Timestamp(v.time_window_start) for v in aligned_views if v.time_window_start is not None]
    ends = [pd.Timestamp(v.time_window_end) for v in aligned_views if v.time_window_end is not None]
    target_time_window_start = min(starts) if starts else None
    target_time_window_end = max(ends) if ends else None

    notes: List[str] = []
    if consensus_direction:
        notes.append(f'Consensus direction is {consensus_direction} based on weighted agreement across timeframes.')
    else:
        notes.append('No clean direction consensus across the selected timeframes.')
    if consensus_pattern:
        notes.append(f'Consensus pattern leans {consensus_pattern}.')
    if state == 'mixed':
        notes.append('Timeframes disagree enough that the count should be treated as fragile; prefer invalidation-first execution.')
    if state == 'high_alignment':
        notes.append('Higher-timeframe and lower-timeframe counts are broadly aligned, which usually makes the setup more decision-useful.')

    return MultiTimeframeSummary(
        symbol=results[0].symbol,
        market=results[0].market,
        intervals=[v.interval for v in views],
        consensus_direction=consensus_direction,
        consensus_pattern=consensus_pattern,
        alignment_score=float(alignment_score),
        confidence_score=float(confidence_score),
        conflict_score=float(conflict_score),
        state=state,
        invalidation=invalidation,
        target_price=target_price,
        target_zone_low=target_zone_low,
        target_zone_high=target_zone_high,
        target_time_window_start=target_time_window_start,
        target_time_window_end=target_time_window_end,
        timeframe_views=views,
        notes=notes,
    )
