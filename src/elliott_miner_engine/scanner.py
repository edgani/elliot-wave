from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Iterable, List, Optional

import pandas as pd

from .data_sources import YahooMarketData
from .engine import ElliottWaveEngine
from .models import ScanResult


@dataclass(slots=True)
class MarketScanner:
    engine: ElliottWaveEngine
    data: YahooMarketData
    max_workers: int = 8

    def scan_symbols(self, symbols: Iterable[str], market: str, interval: str = '1d', period: str = 'max', limit: Optional[int] = None) -> List[ScanResult]:
        symbols = list(symbols)
        if limit is not None:
            symbols = symbols[:limit]
        out: List[ScanResult] = []

        def job(symbol: str) -> ScanResult:
            try:
                df = self.data.fetch(symbol, interval=interval, period=period)
                return self.engine.analyze(df, symbol=symbol, market=market, interval=interval)
            except Exception as e:
                return ScanResult(symbol=symbol, market=market, interval=interval, last_close=float('nan'), best_candidate=None, alternate_candidates=[], error=str(e))

        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futures = {ex.submit(job, s): s for s in symbols}
            for fut in as_completed(futures):
                out.append(fut.result())
        out.sort(key=lambda x: (-1 if x.best_candidate is None else -x.best_candidate.score, x.symbol))
        return out

    @staticmethod
    def to_frame(results: List[ScanResult]) -> pd.DataFrame:
        rows = []
        for r in results:
            c = r.best_candidate
            primary_time_low = None
            primary_time_high = None
            if c is not None and c.fib_time_targets:
                top_time = sorted(c.fib_time_targets, key=lambda t: (-t.weight, t.projected_index))[:3]
                pts = [t.projected_timestamp for t in top_time if t.projected_timestamp is not None]
                if pts:
                    primary_time_low = min(pts)
                    primary_time_high = max(pts)
            rows.append(
                {
                    'symbol': r.symbol,
                    'market': r.market,
                    'interval': r.interval,
                    'last_close': r.last_close,
                    'pattern': None if c is None else c.pattern_type,
                    'direction': None if c is None else c.direction,
                    'subtype': None if c is None else c.meta.get('subtype'),
                    'score': None if c is None else c.score,
                    'confidence': None if c is None else c.confidence,
                    'stability_score': None if c is None else c.meta.get('stability_score'),
                    'stability_survival_ratio': None if c is None else c.meta.get('stability_survival_ratio'),
                    'invalidation': None if c is None else c.invalidation,
                    'bars_since_last_pivot': None if c is None else c.meta.get('bars_since_last_pivot'),
                    'recency_score': None if c is None else c.meta.get('recency_score'),
                    'primary_price_target': None if c is None else c.meta.get('primary_price_target'),
                    'primary_price_zone_low': None if c is None else c.meta.get('primary_price_zone_low'),
                    'primary_price_zone_high': None if c is None else c.meta.get('primary_price_zone_high'),
                    'primary_time_window_start': primary_time_low,
                    'primary_time_window_end': primary_time_high,
                    'error': r.error,
                }
            )
        return pd.DataFrame(rows)
