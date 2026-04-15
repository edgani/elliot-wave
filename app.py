
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd
import plotly.graph_objects as go
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
def load_market_universe(market: str, us_common_only: bool, allow_unverified_third_party: bool) -> dict:
    try:
        result = load_market_universe_safe(
            market,
            us_common_only=us_common_only,
            allow_unverified_third_party=allow_unverified_third_party,
        )
        return {
            'market': result.market,
            'df': result.df,
            'source': result.source,
            'is_live': result.is_live,
            'is_complete': result.is_complete,
            'notes': result.notes,
            'error': None,
        }
    except Exception as e:
        return {
            'market': market,
            'df': pd.DataFrame(columns=['symbol', 'name']),
            'source': 'unavailable',
            'is_live': False,
            'is_complete': False,
            'notes': [f'Universe load failed safely: {e}'],
            'error': str(e),
        }


def _default_symbol(market: str) -> str:
    return {
        'ihsg': 'BBCA.JK',
        'us_stocks': 'AAPL',
        'forex': 'EURUSD=X',
        'commodities': 'GC=F',
        'crypto': 'BTC-USD',
    }.get(market, '')


def _normalize_uploaded_universe(uploaded_file, market: str) -> pd.DataFrame:
    if uploaded_file is None:
        return pd.DataFrame(columns=['symbol', 'name'])
    name = uploaded_file.name.lower()
    if name.endswith('.csv'):
        raw = pd.read_csv(uploaded_file)
    elif name.endswith('.xlsx') or name.endswith('.xls'):
        raw = pd.read_excel(uploaded_file)
    else:
        raise ValueError('Unsupported file type. Use CSV or XLSX.')
    cols = {str(c).lower(): c for c in raw.columns}
    symbol_col = cols.get('symbol') or cols.get('ticker') or cols.get('code') or cols.get('kode') or raw.columns[0]
    name_col = cols.get('name') or cols.get('company') or cols.get('company name') or cols.get('nama')
    df = pd.DataFrame({
        'symbol': raw[symbol_col].astype(str).str.strip(),
        'name': raw[name_col].astype(str).str.strip() if name_col else raw[symbol_col].astype(str).str.strip(),
    })
    if market == 'ihsg':
        df['symbol'] = df['symbol'].str.upper().str.replace('.JK', '', regex=False) + '.JK'
    else:
        df['symbol'] = df['symbol'].str.upper()
    return df.drop_duplicates('symbol').reset_index(drop=True)


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
                'remaining_bars_central': w.remaining_bars_central,
                'projected_end_central': w.projected_end_timestamp_central,
                'fit_score': w.fit_score,
                'status': w.status,
                'basis': w.basis,
            }
        )
    return pd.DataFrame(rows)


def _target_rows(candidate) -> pd.DataFrame:
    if candidate is None:
        return pd.DataFrame()
    rows = []
    if candidate.invalidation is not None:
        rows.append({'kind': 'risk', 'name': 'Invalidation', 'value': candidate.invalidation, 'weight': 1.0})
    ppt = candidate.meta.get('primary_price_target')
    pzl = candidate.meta.get('primary_price_zone_low')
    pzh = candidate.meta.get('primary_price_zone_high')
    if ppt is not None:
        rows.append({'kind': 'target', 'name': 'Primary target', 'value': ppt, 'weight': 1.0})
    if pzl is not None:
        rows.append({'kind': 'zone', 'name': 'Primary zone low', 'value': pzl, 'weight': 0.9})
    if pzh is not None:
        rows.append({'kind': 'zone', 'name': 'Primary zone high', 'value': pzh, 'weight': 0.9})
    seen = set()
    for t in sorted(candidate.fib_price_targets, key=lambda x: (-x.weight, abs(x.value - (ppt or x.value)))):
        key = round(float(t.value), 6)
        if key in seen:
            continue
        rows.append({'kind': 'price', 'name': t.name, 'value': t.value, 'weight': t.weight})
        seen.add(key)
        if len(rows) >= 8:
            break
    return pd.DataFrame(rows)


def _data_audit(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame([{'rows': 0, 'start': None, 'end': None, 'stale_days': None}])
    start = pd.Timestamp(df.index.min())
    end = pd.Timestamp(df.index.max())
    stale_days = (pd.Timestamp.utcnow().tz_localize(None).normalize() - end.tz_localize(None).normalize() if getattr(end, 'tzinfo', None) else pd.Timestamp.utcnow().normalize() - end.normalize()).days
    return pd.DataFrame([{
        'rows': int(len(df)),
        'start': start,
        'end': end,
        'stale_days': int(stale_days),
        'last_close': float(df['Close'].iloc[-1]),
    }])


def plot_simple_chart(df: pd.DataFrame, result, interval: str, lookback_bars: int = 250):
    c = result.best_candidate
    view = df.tail(lookback_bars).copy()
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=view.index,
            open=view['Open'],
            high=view['High'],
            low=view['Low'],
            close=view['Close'],
            name='Price',
            increasing_line_color='#00cc96',
            decreasing_line_color='#ef553b',
            showlegend=False,
        )
    )

    last_x = view.index[-1]
    levels = []
    if c is not None:
        if c.invalidation is not None:
            levels.append(('Invalidation', float(c.invalidation), '#FFA15A', 'dot'))
        ppt = c.meta.get('primary_price_target')
        pzl = c.meta.get('primary_price_zone_low')
        pzh = c.meta.get('primary_price_zone_high')
        if pzl is not None:
            levels.append(('Zone low', float(pzl), '#19D3F3', 'dash'))
        if pzh is not None:
            levels.append(('Zone high', float(pzh), '#19D3F3', 'dash'))
        if ppt is not None:
            levels.append(('Primary target', float(ppt), '#FECB52', 'solid'))
        if c.pivot_indices and c.pivot_prices:
            xs, ys = [], []
            for idx, price in zip(c.pivot_indices, c.pivot_prices):
                if idx < len(df):
                    ts = df.index[idx]
                    if ts >= view.index[0]:
                        xs.append(ts)
                        ys.append(price)
            if xs and ys:
                fig.add_trace(go.Scatter(x=xs, y=ys, mode='lines+markers+text', text=[str(i) for i in range(1, len(xs)+1)], textposition='top center', name='Best count', line=dict(width=1.5, color='#ab63fa'), marker=dict(size=6)))

    levels.append(('Last close', float(view['Close'].iloc[-1]), '#FFFFFF', 'dot'))

    for name, level, color, dash in levels:
        fig.add_hline(y=level, line_width=1, line_dash=dash, line_color=color, annotation_text=f'{name}: {level:,.4f}', annotation_position='right')

    fig.update_layout(
        title=f'{result.symbol} | {interval}',
        template='plotly_dark',
        height=620,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
    )
    fig.update_yaxes(fixedrange=False)
    return fig


st.title('Elliott Wave Miner Engine')
st.caption('Simple decision chart: candlesticks, invalidation, primary target/zone, best count, and timeframe reconciliation.')

with st.sidebar:
    st.header('Controls')
    market = st.selectbox('Market', ['ihsg', 'us_stocks', 'forex', 'commodities', 'crypto'], index=0)
    us_common_only = st.checkbox('US common-only filter', value=False) if market == 'us_stocks' else False
    allow_unverified_third_party = st.checkbox('Allow unverified third-party universe fallback', value=False, help='Useful when official IDX blocks hosted requests. Keep off if you want stricter source hygiene.') if market == 'ihsg' else False
    uploaded_universe = st.file_uploader('Optional universe CSV/XLSX override', type=['csv', 'xlsx', 'xls'], help='Upload an official/exported universe file if you want full control over the dropdown list.')

    universe_payload = load_market_universe(market, us_common_only, allow_unverified_third_party)
    universe = universe_payload['df']

    if uploaded_universe is not None:
        try:
            universe = _normalize_uploaded_universe(uploaded_universe, market)
            st.caption('Universe source: uploaded override file')
            st.caption(f'Rows loaded: {len(universe):,}')
        except Exception as e:
            st.error(f'Uploaded universe file could not be parsed: {e}')
    else:
        st.caption(f"Universe source: {universe_payload['source']}")
        flags = ['live' if universe_payload['is_live'] else 'fallback']
        if universe_payload['is_complete'] is True:
            flags.append('full/near-full')
        elif universe_payload['is_complete'] is False:
            flags.append('partial/curated')
        st.caption(' | '.join(flags))
        with st.expander('Universe notes', expanded=False):
            for note in universe_payload['notes']:
                st.write(f'- {note}')

    default_symbol = _default_symbol(market)
    filter_text = st.text_input('Filter universe list', value='')
    if not universe.empty and filter_text.strip():
        q = filter_text.strip().upper()
        universe = universe[
            universe['symbol'].astype(str).str.upper().str.contains(q, na=False)
            | universe['name'].astype(str).str.upper().str.contains(q, na=False)
        ]
    max_dropdown_rows = st.slider('Max dropdown rows', 100, 5000, 1000, 100)
    universe = universe.head(max_dropdown_rows) if not universe.empty else universe
    symbol_options: List[str] = universe['symbol'].tolist() if not universe.empty else []
    selected_index = symbol_options.index(default_symbol) if default_symbol in symbol_options else (0 if symbol_options else None)
    selected_symbol = st.selectbox('Symbol from universe', symbol_options, index=selected_index) if symbol_options else default_symbol
    custom_symbol = st.text_input('Symbol to analyze', value=selected_symbol)
    symbol = format_symbol_for_market(custom_symbol or selected_symbol, market)

    intervals = st.multiselect('Intervals', ['1d', '1wk', '1mo'], default=['1d', '1wk', '1mo'])
    period = st.selectbox('Period', ['2y', '5y', '10y', 'max'], index=2)
    lookback_bars = st.slider('Chart lookback bars', 80, 600, 250, 10)
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
    fetch_errors = []
    with st.spinner('Running multi-timeframe analysis...'):
        for interval in intervals:
            try:
                df = fetch_data(symbol, interval, period)
                data_map[interval] = df
                results.append(engine.analyze(df, symbol=symbol, market=market, interval=interval))
            except Exception as e:
                fetch_errors.append(f'{interval}: {e}')

    if fetch_errors:
        st.warning('Some intervals failed: ' + ' | '.join(fetch_errors))
    if not results:
        st.error('No intervals could be analyzed. Check the symbol/provider format.')
        st.stop()

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
        st.info(f'Consensus time window: {mtf.target_time_window_start} → {mtf.target_time_window_end}')

    st.subheader('Timeframe summary')
    tf_rows = []
    for v in mtf.timeframe_views:
        tf_rows.append({
            'interval': v.interval,
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
        })
    st.dataframe(pd.DataFrame(tf_rows), use_container_width=True)

    tabs = st.tabs([r.interval for r in results])
    for tab, result in zip(tabs, results):
        with tab:
            candidate = result.best_candidate
            df = data_map[result.interval]
            st.plotly_chart(plot_simple_chart(df, result, result.interval, lookback_bars=lookback_bars), use_container_width=True)
            st.caption('Fetched data audit')
            st.dataframe(_data_audit(df), use_container_width=True, hide_index=True)

            if candidate is None:
                st.warning('No valid candidate found for this interval.')
                continue

            a, b, c = st.columns(3)
            a.metric('Pattern', candidate.pattern_type)
            b.metric('Direction', candidate.direction)
            c.metric('Score / Confidence', f'{candidate.score:.2f} / {candidate.confidence:.2f}')

            d, e, f = st.columns(3)
            d.metric('Invalidation', '-' if candidate.invalidation is None else f'{candidate.invalidation:,.4f}')
            ppt = candidate.meta.get('primary_price_target')
            e.metric('Primary target', '-' if ppt is None else f'{ppt:,.4f}')
            pzl = candidate.meta.get('primary_price_zone_low')
            pzh = candidate.meta.get('primary_price_zone_high')
            f.metric('Primary zone', '-' if pzl is None or pzh is None else f'{pzl:,.4f} → {pzh:,.4f}')

            st.subheader('Key targets')
            st.dataframe(_target_rows(candidate), use_container_width=True, hide_index=True)

            st.subheader('Wave duration projections')
            st.dataframe(candidate_overview(candidate), use_container_width=True, hide_index=True)

            if candidate.momentum_notes:
                st.subheader('Momentum / structure notes')
                for note in candidate.momentum_notes:
                    st.write(f'- {note}')

            alt_rows = []
            for alt in result.alternate_candidates:
                alt_rows.append({
                    'pattern': alt.pattern_type,
                    'direction': alt.direction,
                    'score': alt.score,
                    'confidence': alt.confidence,
                    'invalidation': alt.invalidation,
                    'primary_price_target': alt.meta.get('primary_price_target'),
                })
            if alt_rows:
                st.subheader('Alternate counts')
                st.dataframe(pd.DataFrame(alt_rows), use_container_width=True, hide_index=True)
else:
    st.info('Set market, symbol, intervals, and parameters, then click Run analysis.')
