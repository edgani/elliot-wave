from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional

PatternDirection = Literal["bull", "bear"]
PatternType = Literal["impulse", "zigzag", "flat", "triangle", "ending_diagonal", "double_zigzag"]


@dataclass(slots=True)
class Pivot:
    idx: int
    ts: object
    price: float
    kind: Literal["low", "high"]


@dataclass(slots=True)
class PriceTarget:
    name: str
    value: float
    weight: float


@dataclass(slots=True)
class TimeTarget:
    name: str
    bars_from_anchor: int
    anchor_index: int
    projected_index: int
    anchor_timestamp: Optional[object] = None
    projected_timestamp: Optional[object] = None
    weight: float = 1.0


@dataclass(slots=True)
class WaveDurationProjection:
    wave_name: str
    start_index: int
    end_index: int
    start_timestamp: object
    end_timestamp: object
    actual_bars: int
    projected_bars_central: Optional[int]
    projected_bars_low: Optional[int]
    projected_bars_high: Optional[int]
    projected_end_index_central: Optional[int]
    projected_end_index_low: Optional[int]
    projected_end_index_high: Optional[int]
    projected_end_timestamp_central: Optional[object] = None
    projected_end_timestamp_low: Optional[object] = None
    projected_end_timestamp_high: Optional[object] = None
    remaining_bars_central: Optional[int] = None
    remaining_bars_low: Optional[int] = None
    remaining_bars_high: Optional[int] = None
    fit_score: Optional[float] = None
    basis: str = ""
    status: str = "completed"


@dataclass(slots=True)
class RuleCheck:
    name: str
    passed: bool
    detail: str
    weight: float = 1.0


@dataclass(slots=True)
class WaveCandidate:
    pattern_type: PatternType
    direction: PatternDirection
    pivot_indices: List[int]
    pivot_prices: List[float]
    pivot_timestamps: List[object]
    score: float
    confidence: float
    hard_rule_pass: bool
    rule_checks: List[RuleCheck] = field(default_factory=list)
    fib_price_targets: List[PriceTarget] = field(default_factory=list)
    fib_time_targets: List[TimeTarget] = field(default_factory=list)
    wave_duration_projections: List[WaveDurationProjection] = field(default_factory=list)
    momentum_notes: List[str] = field(default_factory=list)
    invalidation: Optional[float] = None
    meta: Dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class ScanResult:
    symbol: str
    market: str
    interval: str
    last_close: float
    best_candidate: Optional[WaveCandidate]
    alternate_candidates: List[WaveCandidate] = field(default_factory=list)
    error: Optional[str] = None
