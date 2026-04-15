from __future__ import annotations

from typing import Iterable, List

from .models import PriceTarget, TimeTarget

PHI = 1.618033988749895

RET_INTERNAL = [0.236, 0.382, 0.5, 0.618, 0.786]
RET_EXTERNAL = [1.272, 1.618, 2.618]
APP_PRIMARY = [0.618, 1.0, 1.618, 2.618]
APP_SECONDARY = [0.382, 0.618, 1.0]
TIME_RATIOS = [0.382, 0.618, 1.0, 1.618, 2.618]


def retracement_ratio(move_start: float, move_end: float, retrace_end: float) -> float:
    denom = abs(move_end - move_start)
    if denom == 0:
        return 0.0
    return abs(retrace_end - move_end) / denom


def external_retracement_ratio(move_start: float, move_end: float, extension_end: float) -> float:
    denom = abs(move_end - move_start)
    if denom == 0:
        return 0.0
    return abs(extension_end - move_start) / denom


def app_ratio(wave_a_start: float, wave_a_end: float, projection_from: float, projection_to: float) -> float:
    wave_a = abs(wave_a_end - wave_a_start)
    projected = abs(projection_to - projection_from)
    if wave_a == 0:
        return 0.0
    return projected / wave_a


def price_projection(anchor: float, base_move: float, ratio: float, direction: int) -> float:
    return anchor + direction * abs(base_move) * ratio


def time_projection(anchor_index: int, base_bars: int, ratio: float) -> int:
    return anchor_index + max(1, round(abs(base_bars) * ratio))


def closeness_to_set(value: float, levels: Iterable[float], tol: float = 0.12) -> float:
    levels = list(levels)
    if not levels:
        return 0.0
    distances = [abs(value - x) for x in levels]
    best = min(distances)
    score = max(0.0, 1.0 - best / tol)
    return min(1.0, score)


def build_price_targets(names: List[str], values: List[float], weights: List[float]) -> List[PriceTarget]:
    return [PriceTarget(name=n, value=v, weight=w) for n, v, w in zip(names, values, weights)]


def build_time_targets(names: List[str], anchor_index: int, projected_indices: List[int], weights: List[float]) -> List[TimeTarget]:
    return [
        TimeTarget(
            name=n,
            bars_from_anchor=max(0, p - anchor_index),
            anchor_index=anchor_index,
            projected_index=p,
            weight=w,
        )
        for n, p, w in zip(names, projected_indices, weights)
    ]
