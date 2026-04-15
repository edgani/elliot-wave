from __future__ import annotations

from typing import List

import pandas as pd

from .models import Pivot


def adaptive_zigzag(
    df: pd.DataFrame,
    atr_col: str = 'atr14',
    min_reversal_pct: float = 0.03,
    atr_mult: float = 1.5,
    max_pivots: int = 80,
) -> List[Pivot]:
    """
    Causal adaptive zigzag.
    A pivot is only confirmed after price reverses enough from the current extreme.
    """
    if df.empty:
        return []

    close = df['Close'].astype(float)
    high = df['High'].astype(float)
    low = df['Low'].astype(float)
    atr = df[atr_col].bfill().ffill() if atr_col in df else pd.Series(0.0, index=df.index)

    pivots: List[Pivot] = []

    start_price = float(close.iloc[0])
    trend = 0  # 1 up, -1 down, 0 undecided
    extreme_idx = 0
    extreme_price_high = float(high.iloc[0])
    extreme_price_low = float(low.iloc[0])
    last_confirmed_idx = 0
    last_confirmed_price = start_price

    def threshold(i: int) -> float:
        atr_part = atr_mult * float(atr.iloc[i] / max(close.iloc[i], 1e-9)) if not pd.isna(atr.iloc[i]) else 0.0
        return max(min_reversal_pct, atr_part)

    for i in range(1, len(df)):
        th = threshold(i)
        h = float(high.iloc[i])
        l = float(low.iloc[i])

        if trend >= 0:
            if h >= extreme_price_high:
                extreme_price_high = h
                extreme_idx = i
            reversal = (extreme_price_high - l) / max(extreme_price_high, 1e-9)
            if reversal >= th:
                if not pivots or pivots[-1].idx != last_confirmed_idx:
                    pivots.append(Pivot(last_confirmed_idx, df.index[last_confirmed_idx], last_confirmed_price, 'low'))
                pivots.append(Pivot(extreme_idx, df.index[extreme_idx], extreme_price_high, 'high'))
                trend = -1
                last_confirmed_idx = extreme_idx
                last_confirmed_price = extreme_price_high
                extreme_idx = i
                extreme_price_low = l
                extreme_price_high = h
        if trend <= 0:
            if l <= extreme_price_low:
                extreme_price_low = l
                extreme_idx = i
            reversal = (h - extreme_price_low) / max(extreme_price_low, 1e-9)
            if reversal >= th:
                if not pivots or pivots[-1].idx != last_confirmed_idx:
                    pivots.append(Pivot(last_confirmed_idx, df.index[last_confirmed_idx], last_confirmed_price, 'high'))
                pivots.append(Pivot(extreme_idx, df.index[extreme_idx], extreme_price_low, 'low'))
                trend = 1
                last_confirmed_idx = extreme_idx
                last_confirmed_price = extreme_price_low
                extreme_idx = i
                extreme_price_high = h
                extreme_price_low = l

    # Deduplicate and keep order.
    cleaned: List[Pivot] = []
    for p in pivots:
        if cleaned and cleaned[-1].idx == p.idx and cleaned[-1].kind == p.kind:
            cleaned[-1] = p
        elif cleaned and cleaned[-1].kind == p.kind:
            better = p if (p.price > cleaned[-1].price if p.kind == 'high' else p.price < cleaned[-1].price) else cleaned[-1]
            cleaned[-1] = better
        else:
            cleaned.append(p)

    return cleaned[-max_pivots:]
