from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

from .fib import build_price_targets, build_time_targets, price_projection, time_projection
from .models import Pivot, PriceTarget, RuleCheck, ScanResult, TimeTarget, WaveCandidate
from .momentum import add_core_indicators, wave_momentum_strength
from .pivots import adaptive_zigzag
from .timing import impulse_wave_duration_projections, triangle_wave_duration_projections, zigzag_wave_duration_projections
from .wave_rules import evaluate_impulse_window, evaluate_triangle_window, evaluate_zigzag_window


@dataclass(slots=True)
class ElliottWaveEngine:
    min_reversal_pct: float = 0.03
    atr_mult: float = 1.5
    max_pivots: int = 80
    top_n: int = 5
    allow_triangle: bool = True
    allow_ending_diagonal: bool = True

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        out = add_core_indicators(df)
        return out.dropna(subset=['Open', 'High', 'Low', 'Close']).copy()

    def extract_pivots(self, df: pd.DataFrame) -> List[Pivot]:
        return adaptive_zigzag(
            df,
            min_reversal_pct=self.min_reversal_pct,
            atr_mult=self.atr_mult,
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

    def _enumerate_candidates(self, df: pd.DataFrame, pivots: List[Pivot]) -> List[WaveCandidate]:
        out: List[WaveCandidate] = []
        n = len(pivots)
        if n < 4:
            return out

        for i in range(max(0, n - 25), n - 5):
            window = pivots[i : i + 6]
            try:
                hard_pass, checks, soft = evaluate_impulse_window(window)
            except Exception:
                continue
            candidate = self._build_impulse_candidate(df, window, hard_pass, checks, soft)
            if candidate is not None:
                out.append(candidate)
                if self.allow_ending_diagonal:
                    diag = self._try_ending_diagonal(df, window, checks, soft)
                    if diag is not None:
                        out.append(diag)

        for i in range(max(0, n - 25), n - 3):
            window = pivots[i : i + 4]
            try:
                hard_pass, checks, soft = evaluate_zigzag_window(window)
            except Exception:
                continue
            candidate = self._build_zigzag_candidate(df, window, hard_pass, checks, soft)
            if candidate is not None:
                out.append(candidate)

        if self.allow_triangle:
            for i in range(max(0, n - 25), n - 5):
                window = pivots[i : i + 6]
                try:
                    hard_pass, checks, soft = evaluate_triangle_window(window)
                except Exception:
                    continue
                candidate = self._build_triangle_candidate(df, window, hard_pass, checks, soft)
                if candidate is not None:
                    out.append(candidate)

        dedup = {}
        for c in out:
            key = (c.pattern_type, c.direction, tuple(c.pivot_indices))
            if key not in dedup or c.score > dedup[key].score:
                dedup[key] = c
        return list(dedup.values())

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

        # Forward projections for a possible next correction and continuation.
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
        )

        invalidation = prices[4] if direction == 'bull' else prices[4]
        confidence = float(np.clip(score, 0, 1))
        return WaveCandidate(
            pattern_type='impulse',
            direction=direction,
            pivot_indices=idxs,
            pivot_prices=prices,
            pivot_timestamps=tss,
            score=float(score),
            confidence=confidence,
            hard_rule_pass=hard_pass,
            rule_checks=checks,
            fib_price_targets=price_targets,
            fib_time_targets=time_targets,
            wave_duration_projections=impulse_wave_duration_projections(pivots),
            momentum_notes=self._momentum_notes(df, pivots, direction),
            invalidation=invalidation,
            meta=soft,
        )

    def _try_ending_diagonal(self, df: pd.DataFrame, pivots: List[Pivot], checks: List[RuleCheck], soft: dict) -> Optional[WaveCandidate]:
        direction = 'bull' if pivots[-1].price > pivots[0].price else 'bear'
        prices = [p.price for p in pivots]
        # Ending diagonal allows wave 4 overlap into wave 1 range.
        overlap = prices[4] <= prices[1] if direction == 'bull' else prices[4] >= prices[1]
        if not overlap:
            return None
        ext3 = soft.get('w3_app', 0.0)
        ext5 = soft.get('w5_ext_w4', 0.0)
        if not (1.1 <= ext3 <= 2.0 and 1.1 <= ext5 <= 2.0):
            return None
        score = 0.45 + 0.25 * soft.get('fib_score_w3', 0.0) + 0.30 * soft.get('fib_score_w5', 0.0)
        return WaveCandidate(
            pattern_type='ending_diagonal',
            direction=direction,
            pivot_indices=[p.idx for p in pivots],
            pivot_prices=prices,
            pivot_timestamps=[p.ts for p in pivots],
            score=float(score),
            confidence=float(np.clip(score, 0, 1)),
            hard_rule_pass=True,
            rule_checks=checks + [RuleCheck('ending diagonal overlap present', True, 'wave 4 overlaps wave 1 range', 1.0)],
            fib_price_targets=[],
            fib_time_targets=[],
            wave_duration_projections=impulse_wave_duration_projections(pivots),
            momentum_notes=['Ending diagonal proxy: overlap is present and both waves 3 and 5 extend in a diagonal-like range.'],
            invalidation=prices[4],
            meta=soft,
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
            [
                price_projection(prices[2], a, 1.0, sign),
                price_projection(prices[2], a, 1.272, sign),
                price_projection(prices[2], a, 1.618, sign),
            ],
            [0.8, 0.9, 1.0],
        )
        dur_a = idxs[1] - idxs[0]
        time_targets = build_time_targets(
            ['C time 61.8% of A', 'C time 100% of A', 'C time 161.8% of A'],
            idxs[2],
            [
                time_projection(idxs[2], max(1, dur_a), 0.618),
                time_projection(idxs[2], max(1, dur_a), 1.0),
                time_projection(idxs[2], max(1, dur_a), 1.618),
            ],
            [0.7, 0.9, 1.0],
        )
        return WaveCandidate(
            pattern_type='zigzag',
            direction=direction,
            pivot_indices=idxs,
            pivot_prices=prices,
            pivot_timestamps=tss,
            score=float(score),
            confidence=float(np.clip(score, 0, 1)),
            hard_rule_pass=hard_pass,
            rule_checks=checks,
            fib_price_targets=price_targets,
            fib_time_targets=time_targets,
            wave_duration_projections=zigzag_wave_duration_projections(pivots),
            momentum_notes=[],
            invalidation=prices[2],
            meta=soft,
        )

    def _build_triangle_candidate(self, df: pd.DataFrame, pivots: List[Pivot], hard_pass: bool, checks: List[RuleCheck], soft: dict) -> Optional[WaveCandidate]:
        score = 0.7 * (1.0 if hard_pass else 0.0) + 0.3 * soft.get('fib_score', 0.0)
        if score < 0.45:
            return None
        direction = 'bull' if pivots[-1].price > pivots[0].price else 'bear'
        return WaveCandidate(
            pattern_type='triangle',
            direction=direction,
            pivot_indices=[p.idx for p in pivots],
            pivot_prices=[p.price for p in pivots],
            pivot_timestamps=[p.ts for p in pivots],
            score=float(score),
            confidence=float(np.clip(score, 0, 1)),
            hard_rule_pass=hard_pass,
            rule_checks=checks,
            fib_price_targets=[],
            fib_time_targets=[],
            wave_duration_projections=triangle_wave_duration_projections(pivots),
            momentum_notes=['Triangle proxy detected from contracting highs and rising lows.'],
            invalidation=None,
            meta=soft,
        )
