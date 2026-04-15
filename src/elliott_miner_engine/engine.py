from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

from .fib import build_price_targets, build_time_targets, price_projection, time_projection
from .models import Pivot, RuleCheck, ScanResult, WaveCandidate
from .momentum import add_core_indicators, wave_momentum_strength
from .pivots import adaptive_zigzag
from .timing import double_zigzag_duration_projections, impulse_wave_duration_projections, triangle_wave_duration_projections, zigzag_wave_duration_projections
from .wave_rules import (
    evaluate_double_zigzag_window,
    evaluate_flat_window,
    evaluate_impulse_window,
    evaluate_triangle_window,
    evaluate_zigzag_window,
)


@dataclass(slots=True)
class ElliottWaveEngine:
    min_reversal_pct: float = 0.03
    atr_mult: float = 1.5
    max_pivots: int = 80
    top_n: int = 5
    allow_triangle: bool = True
    allow_ending_diagonal: bool = True
    allow_flat: bool = True
    allow_double_zigzag: bool = True
    candidate_lookback_pivots: int = 25
    recency_halflife_bars: int = 20
    enable_stability_filter: bool = True
    stability_trials: int = 4
    stability_top_pool: int = 6

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        out = add_core_indicators(df)
        return out.dropna(subset=['Open', 'High', 'Low', 'Close']).copy()

    def extract_pivots(self, df: pd.DataFrame, min_reversal_pct: Optional[float] = None, atr_mult: Optional[float] = None) -> List[Pivot]:
        return adaptive_zigzag(
            df,
            min_reversal_pct=self.min_reversal_pct if min_reversal_pct is None else min_reversal_pct,
            atr_mult=self.atr_mult if atr_mult is None else atr_mult,
            max_pivots=self.max_pivots,
        )

    def analyze(self, df: pd.DataFrame, symbol: str = '', market: str = '', interval: str = '1d') -> ScanResult:
        prepared = self.prepare(df)
        pivots = self.extract_pivots(prepared)
        candidates = self._enumerate_candidates(prepared, pivots)
        candidates = sorted(candidates, key=lambda x: x.score, reverse=True)
        best = candidates[0] if candidates else None
        return ScanResult(
            symbol=symbol,
            market=market,
            interval=interval,
            last_close=float(prepared['Close'].iloc[-1]),
            best_candidate=best,
            alternate_candidates=candidates[1 : self.top_n],
            error=None,
        )

    def _recency_score(self, df: pd.DataFrame, last_pivot_idx: int) -> tuple[float, int]:
        bars_since = max(0, len(df) - 1 - last_pivot_idx)
        if self.recency_halflife_bars <= 0:
            return 1.0, bars_since
        score = float(0.5 ** (bars_since / self.recency_halflife_bars))
        return score, bars_since

    def _finalize_candidate(self, candidate: WaveCandidate, df: pd.DataFrame) -> WaveCandidate:
        recency_score, bars_since = self._recency_score(df, candidate.pivot_indices[-1])
        hard_rule_score = float(candidate.meta.get('hard_rule_score', 0.0) or 0.0)
        fib_confluence_score = float(candidate.meta.get('fib_confluence_score', 0.0) or 0.0)
        structure_quality = candidate.meta.get('structure_quality')
        if structure_quality is None:
            structure_quality = float(0.55 * hard_rule_score + 0.45 * fib_confluence_score)
        structure_quality = float(np.clip(structure_quality, 0.0, 1.0))
        candidate.meta['base_score'] = float(candidate.score)
        candidate.meta['recency_score'] = recency_score
        candidate.meta['bars_since_last_pivot'] = float(bars_since)
        candidate.meta['hard_rule_score'] = hard_rule_score
        candidate.meta['fib_confluence_score'] = fib_confluence_score
        candidate.meta['structure_quality'] = structure_quality
        candidate.score = float(0.72 * candidate.score + 0.18 * structure_quality + 0.10 * recency_score)
        candidate.confidence = float(np.clip(0.62 * candidate.confidence + 0.23 * structure_quality + 0.15 * recency_score, 0.0, 1.0))
        price_targets = sorted(candidate.fib_price_targets, key=lambda x: (-x.weight, abs(x.value - df['Close'].iloc[-1])))
        if price_targets:
            core = price_targets[: min(3, len(price_targets))]
            candidate.meta['primary_price_zone_low'] = float(min(t.value for t in core))
            candidate.meta['primary_price_zone_high'] = float(max(t.value for t in core))
            candidate.meta['primary_price_target'] = float(np.average([t.value for t in core], weights=[t.weight for t in core]))
        time_targets = sorted(candidate.fib_time_targets, key=lambda x: (-x.weight, x.bars_from_anchor))
        if time_targets:
            core_t = time_targets[: min(3, len(time_targets))]
            candidate.meta['primary_time_index_low'] = float(min(t.projected_index for t in core_t))
            candidate.meta['primary_time_index_high'] = float(max(t.projected_index for t in core_t))
        return candidate

    def _enumerate_candidates(self, df: pd.DataFrame, pivots: List[Pivot]) -> List[WaveCandidate]:
        candidates = self._enumerate_candidates_raw(df, pivots)
        if not candidates:
            return []
        candidates = sorted(candidates, key=lambda x: x.score, reverse=True)
        if self.enable_stability_filter:
            candidates = self._apply_stability_filter(df, candidates)
        dedup = {}
        for c in candidates:
            key = (c.pattern_type, c.direction, tuple(c.pivot_indices))
            if key not in dedup or c.score > dedup[key].score:
                dedup[key] = c
        return list(dedup.values())

    def _enumerate_candidates_raw(self, df: pd.DataFrame, pivots: List[Pivot]) -> List[WaveCandidate]:
        out: List[WaveCandidate] = []
        n = len(pivots)
        if n < 4:
            return out
        start = max(0, n - self.candidate_lookback_pivots)

        for i in range(start, n - 5):
            window = pivots[i : i + 6]
            try:
                hard_pass, checks, soft = evaluate_impulse_window(window)
            except Exception:
                continue
            candidate = self._build_impulse_candidate(df, window, hard_pass, checks, soft)
            if candidate is not None:
                out.append(self._finalize_candidate(candidate, df))
                if self.allow_ending_diagonal:
                    diag = self._try_ending_diagonal(df, window, checks, soft)
                    if diag is not None:
                        out.append(self._finalize_candidate(diag, df))
                if self.allow_double_zigzag:
                    try:
                        hard_pass_dz, checks_dz, soft_dz = evaluate_double_zigzag_window(window)
                    except Exception:
                        hard_pass_dz, checks_dz, soft_dz = False, [], {}
                    dz = self._build_double_zigzag_candidate(df, window, hard_pass_dz, checks_dz, soft_dz)
                    if dz is not None:
                        out.append(self._finalize_candidate(dz, df))

        for i in range(start, n - 3):
            window = pivots[i : i + 4]
            try:
                hard_pass, checks, soft = evaluate_zigzag_window(window)
            except Exception:
                continue
            candidate = self._build_zigzag_candidate(df, window, hard_pass, checks, soft)
            if candidate is not None:
                out.append(self._finalize_candidate(candidate, df))
            if self.allow_flat:
                try:
                    hard_pass_f, checks_f, soft_f = evaluate_flat_window(window)
                except Exception:
                    hard_pass_f, checks_f, soft_f = False, [], {}
                flat = self._build_flat_candidate(df, window, hard_pass_f, checks_f, soft_f)
                if flat is not None:
                    out.append(self._finalize_candidate(flat, df))

        if self.allow_triangle:
            for i in range(start, n - 5):
                window = pivots[i : i + 6]
                try:
                    hard_pass, checks, soft = evaluate_triangle_window(window)
                except Exception:
                    continue
                candidate = self._build_triangle_candidate(df, window, hard_pass, checks, soft)
                if candidate is not None:
                    out.append(self._finalize_candidate(candidate, df))
        return out

    def _stability_variants(self) -> List[tuple[float, float]]:
        if self.stability_trials <= 0:
            return []
        variants = [
            (0.90, 1.00),
            (1.10, 1.00),
            (1.00, 0.90),
            (1.00, 1.10),
            (0.95, 0.95),
            (1.05, 1.05),
        ]
        return variants[: self.stability_trials]

    def _candidate_match_score(self, ref: WaveCandidate, other: WaveCandidate) -> float:
        if ref.pattern_type != other.pattern_type or ref.direction != other.direction:
            return 0.0
        ref_idx = np.asarray(ref.pivot_indices, dtype=float)
        other_idx = np.asarray(other.pivot_indices, dtype=float)
        m = min(len(ref_idx), len(other_idx))
        if m < 3:
            return 0.0
        ref_idx = ref_idx[-m:]
        other_idx = other_idx[-m:]
        scale = max(5.0, float(max(ref_idx[-1], other_idx[-1]) - min(ref_idx[0], other_idx[0])))
        terminal = max(0.0, 1.0 - abs(ref_idx[-1] - other_idx[-1]) / max(3.0, 0.12 * scale))
        path = max(0.0, 1.0 - float(np.mean(np.abs(ref_idx - other_idx))) / max(4.0, 0.18 * scale))
        inv_ref = ref.invalidation
        inv_other = other.invalidation
        inv_score = 0.5
        if inv_ref is not None and inv_other is not None:
            base = max(abs(inv_ref), abs(inv_other), 1e-9)
            inv_score = max(0.0, 1.0 - abs(inv_ref - inv_other) / (0.10 * base))
        subtype_ref = str(ref.meta.get('subtype', ''))
        subtype_other = str(other.meta.get('subtype', ''))
        subtype_score = 1.0 if subtype_ref and subtype_ref == subtype_other else 0.5
        return float(0.45 * terminal + 0.35 * path + 0.15 * inv_score + 0.05 * subtype_score)

    def _apply_stability_filter(self, df: pd.DataFrame, candidates: List[WaveCandidate]) -> List[WaveCandidate]:
        if not candidates:
            return candidates
        pool_n = min(self.stability_top_pool, len(candidates))
        pool = candidates[:pool_n]
        variants = self._stability_variants()
        if not variants:
            return candidates
        for cand in pool:
            matches = []
            survival = 0
            for rev_mult, atr_mult_scale in variants:
                try:
                    pivots = self.extract_pivots(
                        df,
                        min_reversal_pct=max(0.0025, self.min_reversal_pct * rev_mult),
                        atr_mult=max(0.25, self.atr_mult * atr_mult_scale),
                    )
                    alt_candidates = self._enumerate_candidates_raw(df, pivots)
                except Exception:
                    alt_candidates = []
                if not alt_candidates:
                    matches.append(0.0)
                    continue
                best_match = max((self._candidate_match_score(cand, x) for x in alt_candidates), default=0.0)
                if best_match >= 0.45:
                    survival += 1
                matches.append(best_match)
            stability = float(np.mean(matches)) if matches else 0.0
            survival_ratio = float(survival / len(variants)) if variants else 0.0
            cand.meta['stability_score'] = stability
            cand.meta['stability_survival_ratio'] = survival_ratio
            cand.score = float(0.78 * cand.score + 0.16 * stability + 0.06 * survival_ratio)
            cand.confidence = float(np.clip(0.76 * cand.confidence + 0.16 * stability + 0.08 * survival_ratio, 0.0, 1.0))
            if stability >= 0.72 and survival_ratio >= 0.75:
                cand.momentum_notes.append('Count stability is high across small pivot-parameter perturbations.')
            elif stability <= 0.45:
                cand.momentum_notes.append('Count is fragile: small pivot-parameter changes produce materially different structures.')
        return sorted(candidates, key=lambda x: x.score, reverse=True)

    def _momentum_notes(self, df: pd.DataFrame, pivots: List[Pivot], direction: str) -> List[str]:
        notes: List[str] = []
        if len(pivots) < 6:
            return notes
        s1 = wave_momentum_strength(df, pivots[0].idx, pivots[1].idx)
        s3 = wave_momentum_strength(df, pivots[2].idx, pivots[3].idx)
        s5 = wave_momentum_strength(df, pivots[4].idx, pivots[5].idx)
        if s3 > s1 and s3 > s5:
            notes.append('Wave 3 has the strongest normalized momentum, which supports a standard impulse interpretation.')
        else:
            notes.append('Wave 3 is not clearly the strongest momentum leg; reduce confidence in a clean impulse count.')
        rsi3 = float(df['rsi14'].iloc[pivots[3].idx]) if 'rsi14' in df else np.nan
        rsi5 = float(df['rsi14'].iloc[pivots[5].idx]) if 'rsi14' in df else np.nan
        if direction == 'bull' and np.isfinite(rsi3) and np.isfinite(rsi5) and rsi5 < rsi3 and pivots[5].price > pivots[3].price:
            notes.append('Possible bearish momentum divergence into Wave 5.')
        if direction == 'bear' and np.isfinite(rsi3) and np.isfinite(rsi5) and rsi5 > rsi3 and pivots[5].price < pivots[3].price:
            notes.append('Possible bullish momentum divergence into Wave 5.')
        return notes

    def _build_impulse_candidate(self, df: pd.DataFrame, pivots: List[Pivot], hard_pass: bool, checks: List[RuleCheck], soft: dict) -> Optional[WaveCandidate]:
        direction = 'bull' if pivots[-1].price > pivots[0].price else 'bear'
        hard_score = sum(c.weight for c in checks if c.passed) / max(sum(c.weight for c in checks), 1e-9)
        fib_score = np.mean([soft['fib_score_w2'], soft['fib_score_w3'], soft['fib_score_w4'], soft['fib_score_w5']])
        score = 0.6 * hard_score + 0.4 * fib_score
        if soft.get('truncation_flag', 0.0) > 0:
            score *= 0.93
        if score < 0.35:
            return None
        prices = [p.price for p in pivots]
        idxs = [p.idx for p in pivots]
        tss = [p.ts for p in pivots]
        sign = 1 if direction == 'bull' else -1
        w1 = abs(prices[1] - prices[0])
        w3 = abs(prices[3] - prices[2])
        anchor_price = prices[5]
        last_idx = idxs[5]
        dur1 = idxs[1] - idxs[0]
        dur3 = idxs[3] - idxs[2]
        next_w2_targets = [
            price_projection(anchor_price, w1 if w1 else w3, 0.382, -sign),
            price_projection(anchor_price, w1 if w1 else w3, 0.5, -sign),
            price_projection(anchor_price, w1 if w1 else w3, 0.618, -sign),
        ]
        next_w3_targets = [
            price_projection(anchor_price, w1 if w1 else w3, 0.618, sign),
            price_projection(anchor_price, w1 if w1 else w3, 1.0, sign),
            price_projection(anchor_price, w1 if w1 else w3, 1.618, sign),
        ]
        price_targets = build_price_targets(
            ['next correction 38.2%', 'next correction 50%', 'next correction 61.8%', 'next motive 61.8%', 'next motive 100%', 'next motive 161.8%'],
            next_w2_targets + next_w3_targets,
            [0.6, 1.0, 0.8, 0.5, 0.8, 1.0],
        )
        time_targets = build_time_targets(
            ['next timing 61.8% of W1', 'next timing 100% of W1', 'next timing 161.8% of W1', 'next timing 100% of W3'],
            last_idx,
            [
                time_projection(last_idx, max(1, dur1), 0.618),
                time_projection(last_idx, max(1, dur1), 1.0),
                time_projection(last_idx, max(1, dur1), 1.618),
                time_projection(last_idx, max(1, dur3), 1.0),
            ],
            [0.6, 0.8, 1.0, 0.7],
            index=df.index,
        )
        invalidation = prices[4]
        confidence = float(np.clip(score, 0, 1))
        meta = dict(soft)
        meta['hard_rule_score'] = float(hard_score)
        meta['fib_confluence_score'] = float(fib_score)
        meta['structure_quality'] = float(0.55 * hard_score + 0.45 * fib_score)
        meta['subtype'] = 'truncated_impulse' if soft.get('truncation_flag', 0.0) > 0 else 'standard_impulse'
        return WaveCandidate(
            pattern_type='impulse', direction=direction, pivot_indices=idxs, pivot_prices=prices, pivot_timestamps=tss,
            score=float(score), confidence=confidence, hard_rule_pass=hard_pass, rule_checks=checks,
            fib_price_targets=price_targets, fib_time_targets=time_targets,
            wave_duration_projections=impulse_wave_duration_projections(pivots, index=df.index),
            momentum_notes=self._momentum_notes(df, pivots, direction), invalidation=invalidation, meta=meta,
        )

    def _try_ending_diagonal(self, df: pd.DataFrame, pivots: List[Pivot], checks: List[RuleCheck], soft: dict) -> Optional[WaveCandidate]:
        direction = 'bull' if pivots[-1].price > pivots[0].price else 'bear'
        prices = [p.price for p in pivots]
        overlap = prices[4] <= prices[1] if direction == 'bull' else prices[4] >= prices[1]
        if not overlap:
            return None
        ext3 = soft.get('w3_app', 0.0)
        ext5 = soft.get('w5_ext_w4', 0.0)
        if not (1.1 <= ext3 <= 2.0 and 1.1 <= ext5 <= 2.0):
            return None
        score = 0.45 + 0.25 * soft.get('fib_score_w3', 0.0) + 0.30 * soft.get('fib_score_w5', 0.0)
        meta = dict(soft)
        meta['hard_rule_score'] = 0.85
        meta['fib_confluence_score'] = float(0.5 * soft.get('fib_score_w3', 0.0) + 0.5 * soft.get('fib_score_w5', 0.0))
        meta['structure_quality'] = float(score)
        meta['subtype'] = 'ending_diagonal'
        return WaveCandidate(
            pattern_type='ending_diagonal', direction=direction, pivot_indices=[p.idx for p in pivots], pivot_prices=prices,
            pivot_timestamps=[p.ts for p in pivots], score=float(score), confidence=float(np.clip(score, 0, 1)),
            hard_rule_pass=True, rule_checks=checks + [RuleCheck('ending diagonal overlap present', True, 'wave 4 overlaps wave 1 range', 1.0)],
            fib_price_targets=[], fib_time_targets=[], wave_duration_projections=impulse_wave_duration_projections(pivots, index=df.index),
            momentum_notes=['Ending diagonal proxy: overlap is present and both waves 3 and 5 extend in a diagonal-like range.'],
            invalidation=prices[4], meta=meta,
        )

    def _build_zigzag_candidate(self, df: pd.DataFrame, pivots: List[Pivot], hard_pass: bool, checks: List[RuleCheck], soft: dict) -> Optional[WaveCandidate]:
        direction = 'bull' if pivots[-1].price > pivots[0].price else 'bear'
        hard_score = sum(c.weight for c in checks if c.passed) / max(sum(c.weight for c in checks), 1e-9)
        fib_score = np.mean([soft['fib_score_b'], soft['fib_score_c']])
        score = 0.55 * hard_score + 0.45 * fib_score
        if score < 0.32:
            return None
        prices = [p.price for p in pivots]
        idxs = [p.idx for p in pivots]
        tss = [p.ts for p in pivots]
        a = abs(prices[1] - prices[0])
        sign = 1 if direction == 'bull' else -1
        price_targets = build_price_targets(
            ['C ext 100% of A', 'C ext 127.2% of A', 'C ext 161.8% of A'],
            [price_projection(prices[2], a, 1.0, sign), price_projection(prices[2], a, 1.272, sign), price_projection(prices[2], a, 1.618, sign)],
            [0.8, 0.9, 1.0],
        )
        dur_a = idxs[1] - idxs[0]
        time_targets = build_time_targets(
            ['C time 61.8% of A', 'C time 100% of A', 'C time 161.8% of A'], idxs[2],
            [time_projection(idxs[2], max(1, dur_a), 0.618), time_projection(idxs[2], max(1, dur_a), 1.0), time_projection(idxs[2], max(1, dur_a), 1.618)],
            [0.7, 0.9, 1.0], index=df.index,
        )
        meta = dict(soft)
        meta['hard_rule_score'] = float(hard_score)
        meta['fib_confluence_score'] = float(fib_score)
        meta['structure_quality'] = float(0.55 * hard_score + 0.45 * fib_score)
        meta['subtype'] = 'sharp_zigzag' if soft.get('zigzag_variant_sharp', 0.0) >= soft.get('zigzag_variant_deep', 0.0) else 'deep_zigzag'
        return WaveCandidate(
            pattern_type='zigzag', direction=direction, pivot_indices=idxs, pivot_prices=prices, pivot_timestamps=tss,
            score=float(score), confidence=float(np.clip(score, 0, 1)), hard_rule_pass=hard_pass, rule_checks=checks,
            fib_price_targets=price_targets, fib_time_targets=time_targets, wave_duration_projections=zigzag_wave_duration_projections(pivots, index=df.index),
            momentum_notes=[], invalidation=prices[2], meta=meta,
        )

    def _build_double_zigzag_candidate(self, df: pd.DataFrame, pivots: List[Pivot], hard_pass: bool, checks: List[RuleCheck], soft: dict) -> Optional[WaveCandidate]:
        if not soft:
            return None
        hard_score = sum(c.weight for c in checks if c.passed) / max(sum(c.weight for c in checks), 1e-9)
        fib_score = np.mean([soft['fib_score_x1'], soft['fib_score_y'], soft['fib_score_x2'], soft['fib_score_z']])
        score = 0.58 * hard_score + 0.42 * fib_score
        if score < 0.37:
            return None
        direction = 'bull' if pivots[-1].price > pivots[0].price else 'bear'
        prices = [p.price for p in pivots]
        idxs = [p.idx for p in pivots]
        tss = [p.ts for p in pivots]
        first = abs(prices[1] - prices[0])
        second = abs(prices[3] - prices[2])
        sign = 1 if direction == 'bull' else -1
        anchor = prices[4]
        z_targets = build_price_targets(
            ['Z ext 61.8% of W', 'Z ext 100% of W', 'Z ext 127.2% of W', 'Z ext 161.8% of W'],
            [
                price_projection(anchor, first if first else second, 0.618, sign),
                price_projection(anchor, first if first else second, 1.0, sign),
                price_projection(anchor, first if first else second, 1.272, sign),
                price_projection(anchor, first if first else second, 1.618, sign),
            ],
            [0.6, 0.9, 1.0, 0.8],
        )
        dur_w = idxs[1] - idxs[0]
        time_targets = build_time_targets(
            ['Z time 61.8% of W', 'Z time 100% of W', 'Z time 161.8% of W'],
            idxs[4],
            [
                time_projection(idxs[4], max(1, dur_w), 0.618),
                time_projection(idxs[4], max(1, dur_w), 1.0),
                time_projection(idxs[4], max(1, dur_w), 1.618),
            ],
            [0.7, 1.0, 0.8],
            index=df.index,
        )
        meta = dict(soft)
        meta['hard_rule_score'] = float(hard_score)
        meta['fib_confluence_score'] = float(fib_score)
        meta['structure_quality'] = float(0.58 * hard_score + 0.42 * fib_score)
        meta['subtype'] = 'double_zigzag'
        return WaveCandidate(
            pattern_type='double_zigzag', direction=direction, pivot_indices=idxs, pivot_prices=prices, pivot_timestamps=tss,
            score=float(score), confidence=float(np.clip(score, 0, 1)), hard_rule_pass=hard_pass, rule_checks=checks,
            fib_price_targets=z_targets, fib_time_targets=time_targets,
            wave_duration_projections=double_zigzag_duration_projections(pivots, index=df.index),
            momentum_notes=['Complex correction proxy: W-X-Y / double-zigzag style structure detected.'], invalidation=prices[4], meta=meta,
        )

    def _build_flat_candidate(self, df: pd.DataFrame, pivots: List[Pivot], hard_pass: bool, checks: List[RuleCheck], soft: dict) -> Optional[WaveCandidate]:
        if not soft:
            return None
        direction = 'bull' if pivots[-1].price > pivots[0].price else 'bear'
        hard_score = sum(c.weight for c in checks if c.passed) / max(sum(c.weight for c in checks), 1e-9)
        fib_score = np.mean([soft['fib_score_b'], soft['fib_score_c']])
        variant_boost = max(
            float(soft.get('flat_variant_regular_score', 0.0)),
            float(soft.get('flat_variant_expanded_score', 0.0)),
            float(soft.get('flat_variant_running_score', 0.0)),
        )
        score = 0.52 * hard_score + 0.34 * fib_score + 0.14 * variant_boost
        if score < 0.34:
            return None
        prices = [p.price for p in pivots]
        idxs = [p.idx for p in pivots]
        tss = [p.ts for p in pivots]
        a = abs(prices[1] - prices[0])
        sign = 1 if direction == 'bull' else -1
        price_targets = build_price_targets(
            ['flat C 61.8% of A', 'flat C 100% of A', 'flat C 123.6% of A', 'flat C 161.8% of A'],
            [
                price_projection(prices[2], a, 0.618, sign),
                price_projection(prices[2], a, 1.0, sign),
                price_projection(prices[2], a, 1.236, sign),
                price_projection(prices[2], a, 1.618, sign),
            ],
            [0.6, 0.9, 1.0, 0.8],
        )
        dur_a = idxs[1] - idxs[0]
        dur_b = idxs[2] - idxs[1]
        time_targets = build_time_targets(
            ['flat C time 61.8% of A', 'flat C time 100% of A', 'flat C time 161.8% of B'],
            idxs[2],
            [time_projection(idxs[2], max(1, dur_a), 0.618), time_projection(idxs[2], max(1, dur_a), 1.0), time_projection(idxs[2], max(1, dur_b), 1.618)],
            [0.6, 1.0, 0.8], index=df.index,
        )
        variant = soft.get('flat_variant', 'flat')
        meta = dict(soft)
        meta['hard_rule_score'] = float(hard_score)
        meta['fib_confluence_score'] = float(fib_score)
        meta['structure_quality'] = float(0.52 * hard_score + 0.34 * fib_score + 0.14 * variant_boost)
        meta['subtype'] = variant
        return WaveCandidate(
            pattern_type='flat', direction=direction, pivot_indices=idxs, pivot_prices=prices, pivot_timestamps=tss,
            score=float(score), confidence=float(np.clip(score, 0, 1)), hard_rule_pass=hard_pass, rule_checks=checks,
            fib_price_targets=price_targets, fib_time_targets=time_targets, wave_duration_projections=zigzag_wave_duration_projections(pivots, index=df.index),
            momentum_notes=[f'Flat correction candidate: best subtype proxy is {variant.replace("_", " ")}.'], invalidation=prices[2], meta=meta,
        )

    def _build_triangle_candidate(self, df: pd.DataFrame, pivots: List[Pivot], hard_pass: bool, checks: List[RuleCheck], soft: dict) -> Optional[WaveCandidate]:
        score = 0.7 * (1.0 if hard_pass else 0.0) + 0.3 * soft.get('fib_score', 0.0)
        if score < 0.45:
            return None
        direction = 'bull' if pivots[-1].price > pivots[0].price else 'bear'
        meta = dict(soft)
        meta['hard_rule_score'] = float(1.0 if hard_pass else 0.0)
        meta['fib_confluence_score'] = float(soft.get('fib_score', 0.0))
        meta['structure_quality'] = float(0.7 * (1.0 if hard_pass else 0.0) + 0.3 * soft.get('fib_score', 0.0))
        meta['subtype'] = 'contracting_triangle'
        return WaveCandidate(
            pattern_type='triangle', direction=direction, pivot_indices=[p.idx for p in pivots], pivot_prices=[p.price for p in pivots],
            pivot_timestamps=[p.ts for p in pivots], score=float(score), confidence=float(np.clip(score, 0, 1)), hard_rule_pass=hard_pass,
            rule_checks=checks, fib_price_targets=[], fib_time_targets=[], wave_duration_projections=double_zigzag_duration_projections(pivots, index=df.index),
            momentum_notes=['Triangle proxy detected from contracting highs and rising lows.'], invalidation=None, meta=meta,
        )
