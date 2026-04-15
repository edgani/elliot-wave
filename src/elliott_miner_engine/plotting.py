from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd

from .models import Pivot, ScanResult


def plot_scan_result(df: pd.DataFrame, result: ScanResult, output_path: Optional[str] = None) -> Optional[str]:
    plt.figure(figsize=(14, 7))
    plt.plot(df.index, df['Close'], linewidth=1.2, label='Close')
    if result.best_candidate is not None:
        c = result.best_candidate
        xs = [df.index[i] for i in c.pivot_indices]
        ys = c.pivot_prices
        plt.plot(xs, ys, marker='o', linewidth=1.6, label=f'{c.pattern_type} / {c.direction} / {c.score:.2f}')
        for lbl, x, y in zip(['1','2','3','4','5','6','7','8'], xs, ys):
            plt.annotate(lbl, (x, y), textcoords='offset points', xytext=(0, 6), ha='center')
        for t in c.fib_price_targets[:6]:
            plt.axhline(t.value, linestyle='--', linewidth=0.8, alpha=0.5)
    plt.title(f'{result.symbol} - Elliott Wave Miner Engine')
    plt.legend()
    plt.tight_layout()
    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=140)
        plt.close()
        return str(out)
    plt.show()
    return None
