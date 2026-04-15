from __future__ import annotations

from typing import Iterable, List, Optional

import pandas as pd

from .models import Pivot, WaveDurationProjection

TIME_SET_CORRECTIVE = [0.382, 0.618, 1.0, 1.618]
TIME_SET_EXTENDED = [0.618, 1.0, 1.618, 2.618]
TIME_SET_SHALLOW = [0.382, 0.618, 1.0]


def _closeness_to_set(value: float, levels: List[int], tol: float = 2.0) -> float:
    if not levels:
        return 0.0
    best = min(abs(value - x) for x in levels)
    score = max(0.0, 1.0 - best / tol)
    return min(1.0, score)


def infer_index_step(index: Optional[Iterable[object]]) -> Optional[pd.Timedelta]:
    if index is None:
        return None
    idx = pd.Index(index)
    if len(idx) < 2:
        return None
    try:
        if isinstance(idx, pd.DatetimeIndex):
            diffs = idx.to_series().diff().dropna()
            if diffs.empty:
                return None
            return diffs.median()
    except Exception:
        return None
    return None


def project_index_to_timestamp(index: Optional[Iterable[object]], projected_index: Optional[int]) -> Optional[object]:
    if index is None or projected_index is None:
        return None
    idx = pd.Index(index)
    if len(idx) == 0:
        return None
    if 0 <= projected_index < len(idx):
        return idx[projected_index]
    step = infer_index_step(idx)
    if step is None or not isinstance(idx, pd.DatetimeIndex):
        return None
    last = idx[-1]
    extra = projected_index - (len(idx) - 1)
    return last + extra * step


def _weighted_center(values: List[int], weights: List[float]) -> Optional[int]:
    if not values:
        return None
    total_w = sum(weights)
    if total_w <= 0:
        return int(round(sum(values) / len(values)))
    return int(round(sum(v * w for v, w in zip(values, weights)) / total_w))


def _projected_lengths(base_duration: int, ratios: List[float]) -> List[int]:
    base_duration = max(1, int(base_duration))
    return [max(1, int(round(base_duration * r))) for r in ratios]


def _make_projection(
    wave_name: str,
    start_pivot: Pivot,
    end_pivot: Pivot,
    projected_lengths: List[int],
    weights: List[float],
    basis: str,
    status: str = 'completed',
    index: Optional[Iterable[object]] = None,
) -> WaveDurationProjection:
    actual = max(1, end_pivot.idx - start_pivot.idx)
    central = _weighted_center(projected_lengths, weights) if projected_lengths else None
    low = min(projected_lengths) if projected_lengths else None
    high = max(projected_lengths) if projected_lengths else None
    fit = _closeness_to_set(actual, projected_lengths, tol=max(2.0, 0.35 * max(projected_lengths or [actual]))) if projected_lengths else None
    proj_c = None if central is None else start_pivot.idx + central
    proj_l = None if low is None else start_pivot.idx + low
    proj_h = None if high is None else start_pivot.idx + high
    return WaveDurationProjection(
        wave_name=wave_name,
        start_index=start_pivot.idx,
        end_index=end_pivot.idx,
        start_timestamp=start_pivot.ts,
        end_timestamp=end_pivot.ts,
        actual_bars=actual,
        projected_bars_central=central,
        projected_bars_low=low,
        projected_bars_high=high,
        projected_end_index_central=proj_c,
        projected_end_index_low=proj_l,
        projected_end_index_high=proj_h,
        projected_end_timestamp_central=project_index_to_timestamp(index, proj_c),
        projected_end_timestamp_low=project_index_to_timestamp(index, proj_l),
        projected_end_timestamp_high=project_index_to_timestamp(index, proj_h),
        remaining_bars_central=None if central is None else central - actual,
        remaining_bars_low=None if low is None else low - actual,
        remaining_bars_high=None if high is None else high - actual,
        fit_score=fit,
        basis=basis,
        status=status,
    )


def impulse_wave_duration_projections(pivots: List[Pivot], index: Optional[Iterable[object]] = None) -> List[WaveDurationProjection]:
    if len(pivots) != 6:
        return []
    d1 = max(1, pivots[1].idx - pivots[0].idx)
    d2 = max(1, pivots[2].idx - pivots[1].idx)
    d3 = max(1, pivots[3].idx - pivots[2].idx)

    out = [
        WaveDurationProjection(
            wave_name='W1',
            start_index=pivots[0].idx,
            end_index=pivots[1].idx,
            start_timestamp=pivots[0].ts,
            end_timestamp=pivots[1].ts,
            actual_bars=d1,
            projected_bars_central=None,
            projected_bars_low=None,
            projected_bars_high=None,
            projected_end_index_central=None,
            projected_end_index_low=None,
            projected_end_index_high=None,
            projected_end_timestamp_central=None,
            projected_end_timestamp_low=None,
            projected_end_timestamp_high=None,
            remaining_bars_central=None,
            remaining_bars_low=None,
            remaining_bars_high=None,
            fit_score=None,
            basis='Seed motive leg. No same-pattern causal prior wave is available for a clean duration projection.',
            status='completed_seed',
        ),
        _make_projection(
            'W2', pivots[1], pivots[2], _projected_lengths(d1, TIME_SET_CORRECTIVE), [0.6, 1.0, 0.8, 0.5],
            'Projected from W1 duration using Miner-style corrective time ratios 0.382/0.618/1.0/1.618 of W1.', index=index,
        ),
        _make_projection(
            'W3', pivots[2], pivots[3], _projected_lengths(d1, TIME_SET_EXTENDED), [0.5, 0.9, 1.0, 0.6],
            'Projected from W1 duration using motive-extension time ratios 0.618/1.0/1.618/2.618 of W1.', index=index,
        ),
    ]
    w4_candidates = _projected_lengths(d2, [0.618, 1.0, 1.618]) + _projected_lengths(d1, TIME_SET_SHALLOW)
    w4_weights = [0.9, 1.0, 0.6, 0.5, 0.8, 0.6]
    out.append(
        _make_projection(
            'W4', pivots[3], pivots[4], w4_candidates, w4_weights,
            'Projected from W2 and W1 durations. Corrective Wave 4 often clusters around 0.618/1.0/1.618 of W2 and 0.382/0.618/1.0 of W1.', index=index,
        )
    )
    w5_candidates = _projected_lengths(d1, [0.618, 1.0, 1.618]) + _projected_lengths(d3, TIME_SET_SHALLOW)
    w5_weights = [0.8, 1.0, 0.7, 0.6, 0.9, 0.7]
    out.append(
        _make_projection(
            'W5', pivots[4], pivots[5], w5_candidates, w5_weights,
            'Projected from W1 and W3 durations. Wave 5 often clusters around 0.618/1.0/1.618 of W1 and 0.382/0.618/1.0 of W3.', index=index,
        )
    )
    return out


def zigzag_wave_duration_projections(pivots: List[Pivot], index: Optional[Iterable[object]] = None) -> List[WaveDurationProjection]:
    if len(pivots) != 4:
        return []
    d_a = max(1, pivots[1].idx - pivots[0].idx)
    out = [
        WaveDurationProjection(
            wave_name='A',
            start_index=pivots[0].idx,
            end_index=pivots[1].idx,
            start_timestamp=pivots[0].ts,
            end_timestamp=pivots[1].ts,
            actual_bars=d_a,
            projected_bars_central=None,
            projected_bars_low=None,
            projected_bars_high=None,
            projected_end_index_central=None,
            projected_end_index_low=None,
            projected_end_index_high=None,
            projected_end_timestamp_central=None,
            projected_end_timestamp_low=None,
            projected_end_timestamp_high=None,
            remaining_bars_central=None,
            remaining_bars_low=None,
            remaining_bars_high=None,
            fit_score=None,
            basis='Seed corrective leg. No same-pattern causal prior wave is available for a clean duration projection.',
            status='completed_seed',
        ),
        _make_projection(
            'B', pivots[1], pivots[2], _projected_lengths(d_a, TIME_SET_CORRECTIVE), [0.6, 1.0, 0.8, 0.5],
            'Projected from A duration using corrective time ratios 0.382/0.618/1.0/1.618 of A.', index=index,
        ),
        _make_projection(
            'C', pivots[2], pivots[3], _projected_lengths(d_a, TIME_SET_EXTENDED), [0.6, 0.9, 1.0, 0.7],
            'Projected from A duration using motive-like time ratios 0.618/1.0/1.618/2.618 of A.', index=index,
        ),
    ]
    return out


def triangle_wave_duration_projections(pivots: List[Pivot], index: Optional[Iterable[object]] = None) -> List[WaveDurationProjection]:
    if len(pivots) != 6:
        return []
    names = ['A', 'B', 'C', 'D', 'E']
    out: List[WaveDurationProjection] = []
    base_duration = max(1, pivots[1].idx - pivots[0].idx)
    out.append(
        WaveDurationProjection(
            wave_name='A',
            start_index=pivots[0].idx,
            end_index=pivots[1].idx,
            start_timestamp=pivots[0].ts,
            end_timestamp=pivots[1].ts,
            actual_bars=base_duration,
            projected_bars_central=None,
            projected_bars_low=None,
            projected_bars_high=None,
            projected_end_index_central=None,
            projected_end_index_low=None,
            projected_end_index_high=None,
            projected_end_timestamp_central=None,
            projected_end_timestamp_low=None,
            projected_end_timestamp_high=None,
            remaining_bars_central=None,
            remaining_bars_low=None,
            remaining_bars_high=None,
            fit_score=None,
            basis='Seed triangle leg. No same-pattern causal prior leg is available for a clean duration projection.',
            status='completed_seed',
        )
    )
    for i, name in enumerate(names[1:], start=1):
        start_pivot = pivots[i]
        end_pivot = pivots[i + 1]
        prev_duration = max(1, pivots[i].idx - pivots[i - 1].idx)
        out.append(
            _make_projection(
                name,
                start_pivot,
                end_pivot,
                _projected_lengths(prev_duration, [0.618, 1.0, 1.618]),
                [0.8, 1.0, 0.6],
                f'Projected from prior triangle leg duration using 0.618/1.0/1.618 of prior leg ({names[i-1]}).',
                index=index,
            )
        )
    return out
