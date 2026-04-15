from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from elliott_miner_engine import ElliottWaveEngine, YahooMarketData, reconcile_results
from elliott_miner_engine.data_sources import format_symbol_for_market, load_market_universe_safe


st.set_page_config(page_title="Elliott Wave Miner Engine", layout="wide")


@st.cache_data(show_spinner=False)
def fetch_data(symbol: str, interval: str, period: str) -> pd.DataFrame:
    data = YahooMarketData()
    return data.fetch(symbol, interval=interval, period=period)


@st.cache_data(show_spinner=False)
def load_market_universe(market: str, us_common_only: bool) -> dict:
    result = load_market_universe_safe(market, us_common_only=us_common_only)
    return {
        'market': result.market,
        'df': result.df,
        'source': result.source,
        'is_live': result.is_live,
        'is_complete': result.is_complete,
        'notes': result.notes,
    }


def candidate_overview(candidate) -> pd.DataFrame:
    if candidate is None:
        return pd.DataFrame()
    rows = []
    for w in candidate.wave_duration_projections:
        rows.append(
            {
                'wave': w.wave_name,
                'actual_bars': w.actual_bars,
                'proj_bars_central': w.projected_bars_central,
                'proj_bars_low': w.projected_bars_low,
                'proj_bars_high': w.projected_bars_high,
                'remaining_bars_central': w.remaining_bars_central,
                'projected_end_central': w.projected_end_timestamp_central,
                'fit_score': w.fit_score,
                'status': w.status,
                'basis': w.basis,
            }
        )
    return pd.DataFrame(rows)


def target_table(candidate) -> pd.DataFrame:
    if candidate is None:
        return pd.DataFrame()
    rows = []
    for t in candidate.fib_price_targets[:12]:
        rows.append({'type': 'price', 'name': t.name, 'value': t.value, 'weight': t.weight})
    for t in candidate.fib_time_targets[:12]:
        rows.append({'type': 'time', 'name': t.name, 'value': t.projected_timestamp or t.projected_index, 'weight': t.weight})
    return pd.DataFrame(rows)


def plot_result(df: pd.DataFrame, result, interval: str):
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df.index, df['Close'], linewidth=1.2, label='Close')
    c = result.best_candidate
    if c is not None:
        xs = [df.index[i] for i in c.pivot_indices if 0 <= i < len(df)]
        ys = c.pivot_prices[:len(xs)]
        if xs and ys:
            ax.plot(xs, ys, marker='o', linewidth=1.5, label=f'{c.pattern_type} / {c.direction} / {c.score:.2f}')
            for idx, (x, y) in enumerate(zip(xs, ys), start=1):
                ax.annotate(str(idx), (x, y), textcoords='offset points', xytext=(0, 6), ha='center')
        for t in c.fib_price_targets[:8]:
            ax.axhline(t.value, linestyle='--', linewidth=0.8, alpha=0.4)
        inv = c.invalidation
        if inv is not None:
            ax.axhline(inv, linestyle=':', linewidth=1.0, alpha=0.8)
    ax.set_title(f'{result.symbol} | {interval}')
    ax.legend(loc='upper left')
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


def _default_symbol(market: str) -> str:
    return {
        'ihsg': 'BBCA.JK',
        'us_stocks': 'AAPL',
        'forex': 'EURUSD=X',
        'commodities': 'GC=F',
        'crypto': 'BTC-USD',
    }.get(market, '')


st.title('Elliott Wave Miner Engine')
st.caption('Multi-timeframe practical Elliott Wave analysis with price/time targets, invalidation, duration projections, and safer universe loading.')

with st.sidebar:
    st.header('Controls')
    market = st.selectbox('Market', ['ihsg', 'us_stocks', 'forex', 'commodities', 'crypto'], index=0)
    us_common_only = st.checkbox('US: common-only filter', value=False, help='Turn on to reduce many special share classes, warrants, and rights lines.') if market == 'us_stocks' else False

    universe_payload = load_market_universe(market, us_common_only)
    universe = universe_payload['df']

    st.caption(f"Universe source: {universe_payload['source']}")
    source_flags = []
    source_flags.append('live' if universe_payload['is_live'] else 'fallback')
    if universe_payload['is_complete'] is True:
        source_flags.append('full/near-full')
    elif universe_payload['is_complete'] is False:
        source_flags.append('partial/curated')
    st.caption(' | '.join(source_flags))

    if universe_payload['notes']:
        with st.expander('Universe notes', expanded=False):
            for note in universe_payload['notes']:
                st.write(f'- {note}')

    if universe.empty:
        st.warning('Universe list unavailable. Manual symbol entry still works.')
        selected_symbol = _default_symbol(market)
    else:
        st.caption(f'Universe rows loaded: {len(universe):,}')
        filter_text = st.text_input('Filter universe list', value='', help='Filter by symbol or company/coin name before picking from the dropdown.')
        filtered = universe
        if filter_text.strip():
            q = filter_text.strip().upper()
            filtered = universe[
                universe['symbol'].astype(str).str.upper().str.contains(q, na=False)
                | universe['name'].astype(str).str.upper().str.contains(q, na=False)
            ]
        max_dropdown_rows = st.slider('Max dropdown rows', 50, 1000, 300, 50)
        filtered = filtered.head(max_dropdown_rows)
        symbol_options: List[str] = filtered['symbol'].tolist()
        default_symbol = _default_symbol(market)
        if default_symbol in symbol_options:
            selected_index = symbol_options.index(default_symbol)
        else:
            selected_index = 0 if symbol_options else None
        selected_symbol = (
            st.selectbox('Symbol from loaded universe', options=symbol_options, index=selected_index)
            if symbol_options
            else default_symbol
        )

    custom_symbol = st.text_input('Symbol to analyze', value=selected_symbol, help='Manual input overrides the dropdown. For IHSG, plain BBCA will auto-convert to BBCA.JK.')
    symbol = format_symbol_for_market(custom_symbol or selected_symbol, market)

    intervals = st.multiselect('Intervals', ['1d', '1wk', '1mo'], default=['1d', '1wk', '1mo'])
    period = st.selectbox('Period', ['2y', '5y', '10y', 'max'], index=2)
    min_reversal_pct = st.slider('Min reversal %', 0.005, 0.15, 0.03, 0.005)
    atr_mult = st.slider('ATR multiplier', 0.5, 4.0, 1.5, 0.1)
    max_pivots = st.slider('Max pivots', 20, 150, 80, 5)
    candidate_lookback_pivots = st.slider('Candidate lookback pivots', 8, 50, 25, 1)
    top_n = st.slider('Top alternate counts', 1, 10, 5, 1)
    run = st.button('Run analysis', type='primary')

if run:
    if not intervals:
        st.error('Pick at least one interval.')
        st.stop()
    if not symbol:
        st.error('Enter a symbol first.')
        st.stop()

    engine = ElliottWaveEngine(
        min_reversal_pct=min_reversal_pct,
        atr_mult=atr_mult,
        max_pivots=max_pivots,
        top_n=top_n,
        candidate_lookback_pivots=candidate_lookback_pivots,
    )

    results = []
    data_map = {}
    with st.spinner('Running multi-timeframe analysis...'):
        for interval in intervals:
            df = fetch_data(symbol, interval, period)
            data_map[interval] = df
            results.append(engine.analyze(df, symbol=symbol, market=market, interval=interval))

    mtf = reconcile_results(results)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Consensus direction', mtf.consensus_direction or 'n/a')
    c2.metric('Consensus pattern', mtf.consensus_pattern or 'n/a')
    c3.metric('Alignment score', f'{mtf.alignment_score:.2f}')
    c4.metric('State', mtf.state)

    c5, c6, c7 = st.columns(3)
    c5.metric('Consensus invalidation', '-' if mtf.invalidation is None else f'{mtf.invalidation:,.4f}')
    c6.metric('Consensus target', '-' if mtf.target_price is None else f'{mtf.target_price:,.4f}')
    c7.metric('Target zone', '-' if mtf.target_zone_low is None or mtf.target_zone_high is None else f'{mtf.target_zone_low:,.4f} → {mtf.target_zone_high:,.4f}')

    if mtf.target_time_window_start is not None or mtf.target_time_window_end is not None:
        st.info(f"Consensus time window: {mtf.target_time_window_start} → {mtf.target_time_window_end}")

    with st.expander('Multi-timeframe reconciliation notes', expanded=True):
        for note in mtf.notes:
            st.write(f'- {note}')
        tf_rows = []
        for v in mtf.timeframe_views:
            tf_rows.append(
                {
                    'interval': v.interval,
                    'weight': v.weight,
                    'direction': v.direction,
                    'pattern': v.pattern_type,
                    'score': v.score,
                    'confidence': v.confidence,
                    'invalidation': v.invalidation,
                    'target_price': v.price_target,
                    'target_zone_low': v.price_zone_low,
                    'target_zone_high': v.price_zone_high,
                    'time_window_start': v.time_window_start,
                    'time_window_end': v.time_window_end,
                }
            )
        st.dataframe(pd.DataFrame(tf_rows), use_container_width=True)

    tabs = st.tabs(intervals)
    for tab, interval, result in zip(tabs, intervals, results):
        with tab:
            candidate = result.best_candidate
            if candidate is None:
                st.warning('No valid candidate found for this interval.')
                continue

            st.pyplot(plot_result(data_map[interval], result, interval), use_container_width=True)

            a, b, c = st.columns(3)
            a.metric('Pattern', candidate.pattern_type)
            b.metric('Direction', candidate.direction)
            c.metric('Score / Confidence', f'{candidate.score:.2f} / {candidate.confidence:.2f}')

            d, e, f = st.columns(3)
            d.metric('Invalidation', '-' if candidate.invalidation is None else f'{candidate.invalidation:,.4f}')
            d2 = candidate.meta.get('primary_price_target')
            e.metric('Primary target', '-' if d2 is None else f'{d2:,.4f}')
            zone_low = candidate.meta.get('primary_price_zone_low')
            zone_high = candidate.meta.get('primary_price_zone_high')
            f.metric('Primary zone', '-' if zone_low is None or zone_high is None else f'{zone_low:,.4f} → {zone_high:,.4f}')

            st.subheader('Wave duration projections')
            st.dataframe(candidate_overview(candidate), use_container_width=True)

            st.subheader('Targets')
            st.dataframe(target_table(candidate), use_container_width=True)

            if candidate.momentum_notes:
                st.subheader('Momentum / structure notes')
                for note in candidate.momentum_notes:
                    st.write(f'- {note}')

            alt_rows = []
            for alt in result.alternate_candidates:
                alt_rows.append(
                    {
                        'pattern': alt.pattern_type,
                        'direction': alt.direction,
                        'score': alt.score,
                        'confidence': alt.confidence,
                        'invalidation': alt.invalidation,
                        'primary_price_target': alt.meta.get('primary_price_target'),
                    }
                )
            if alt_rows:
                st.subheader('Alternate counts')
                st.dataframe(pd.DataFrame(alt_rows), use_container_width=True)
else:
    st.info('Set market, symbol, intervals, and parameters, then click Run analysis.')
