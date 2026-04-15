from __future__ import annotations

import numpy as np
import pandas as pd


def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    avg_up = up.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    avg_down = down.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    rs = avg_up / avg_down.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50.0)


def roc(series: pd.Series, length: int = 5) -> pd.Series:
    prev = series.shift(length)
    return 100.0 * (series / prev - 1.0)


def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    high = df['High']
    low = df['Low']
    close = df['Close']
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()


def add_core_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out['rsi14'] = rsi(out['Close'], 14)
    out['roc5'] = roc(out['Close'], 5)
    out['roc13'] = roc(out['Close'], 13)
    out['atr14'] = atr(out, 14)
    out['atr_pct'] = out['atr14'] / out['Close'].replace(0, np.nan)
    return out


def wave_momentum_strength(df: pd.DataFrame, start_idx: int, end_idx: int) -> float:
    if end_idx <= start_idx:
        return 0.0
    window = df.iloc[start_idx : end_idx + 1]
    if window.empty:
        return 0.0
    price_move = abs(window['Close'].iloc[-1] - window['Close'].iloc[0])
    avg_atr = float(window['atr14'].mean()) if 'atr14' in window else 0.0
    denom = avg_atr if avg_atr and avg_atr > 0 else max(abs(window['Close'].iloc[0]) * 0.01, 1e-9)
    return float(price_move / denom)
