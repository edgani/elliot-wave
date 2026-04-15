from __future__ import annotations

import numpy as np
import pandas as pd

from elliott_miner_engine.engine import ElliottWaveEngine
from elliott_miner_engine.models import Pivot
from elliott_miner_engine.wave_rules import evaluate_impulse_window


def synthetic_series() -> pd.DataFrame:
    close = np.array([
        100, 102, 105, 109, 113,
        110, 107, 109, 112,
        116, 121, 127,
        124, 121, 123, 126,
        130, 135, 141,
        139, 137, 138, 141,
    ], dtype=float)
    idx = pd.date_range('2023-01-01', periods=len(close), freq='D')
    df = pd.DataFrame(index=idx)
    df['Open'] = close
    df['High'] = close + 0.8
    df['Low'] = close - 0.8
    df['Close'] = close
    df['Volume'] = 1_000
    return df


def main() -> None:
    # Rule-level sanity check with hand-crafted pivots.
    pivots = [
        Pivot(0, None, 100.0, 'low'),
        Pivot(4, None, 113.0, 'high'),
        Pivot(6, None, 107.0, 'low'),
        Pivot(11, None, 127.0, 'high'),
        Pivot(13, None, 121.0, 'low'),
        Pivot(18, None, 141.0, 'high'),
    ]
    hard_pass, checks, soft = evaluate_impulse_window(pivots)
    assert hard_pass, 'Impulse hard rules failed'
    assert soft['fib_score_w2'] > 0
    assert soft['fib_score_w3'] > 0

    # End-to-end engine sanity check.
    engine = ElliottWaveEngine(min_reversal_pct=0.015, atr_mult=0.8)
    result = engine.analyze(synthetic_series(), symbol='SYN', market='test')
    assert result.best_candidate is not None, 'No candidate found'
    assert result.best_candidate.wave_duration_projections, 'No wave-duration projections found'
    names = [x.wave_name for x in result.best_candidate.wave_duration_projections]
    assert any(n in names for n in ['W3', 'C', 'D']), 'Expected wave duration labels missing'
    print('OK:', result.best_candidate.pattern_type, result.best_candidate.score, names)


if __name__ == '__main__':
    main()
