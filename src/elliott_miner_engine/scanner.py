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
            rows.append(
                {
                    'symbol': r.symbol,
                    'market': r.market,
                    'interval': r.interval,
                    'last_close': r.last_close,
                    'pattern': None if c is None else c.pattern_type,
                    'direction': None if c is None else c.direction,
                    'score': None if c is None else c.score,
                    'confidence': None if c is None else c.confidence,
                    'invalidation': None if c is None else c.invalidation,
                    'error': r.error,
                }
            )
        return pd.DataFrame(rows)
