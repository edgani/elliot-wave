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

from elliott_miner_engine import ElliottWaveEngine, MarketScanner, YahooMarketData, reconcile_results, build_degree_hierarchy, hierarchy_frame
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
            "market": result.market,
            "df": result.df,
            "source": result.source,
            "is_live": result.is_live,
            "is_complete": result.is_complete,
            "notes": result.notes,
            "error": None,
        }
    except Exception as e:
        return {
            "market": market,
            "df": pd.DataFrame(columns=["symbol", "name"]),
            "source": "unavailable",
            "is_live": False,
            "is_complete": False,
            "notes": [f"Universe load failed safely: {e}"],
            "error": str(e),
        }


@st.cache_data(show_spinner=False)
def scan_universe_cached(
    symbols: tuple[str, ...],
    market: str,
    interval: str,
    period: str,
    min_reversal_pct: float,
    atr_mult: float,
    max_pivots: int,
    top_n: int,
    candidate_lookback_pivots: int,
    max_workers: int,
) -> pd.DataFrame:
    engine = ElliottWaveEngine(
        min_reversal_pct=min_reversal_pct,
        atr_mult=atr_mult,
        max_pivots=max_pivots,
        top_n=top_n,
        candidate_lookback_pivots=candidate_lookback_pivots,
    )
    scanner = MarketScanner(engine=engine, data=YahooMarketData(), max_workers=max_workers)
    results = scanner.scan_symbols(list(symbols), market=market, interval=interval, period=period, limit=None)
    return scanner.to_frame(results)



def _default_symbol(market: str) -> str:
    return {
        "ihsg": "BBCA.JK",
        "us_stocks": "AAPL",
        "forex": "EURUSD=X",
        "commodities": "GC=F",
        "crypto": "BTC-USD",
    }.get(market, "")



def _normalize_uploaded_universe(uploaded_file, market: str) -> pd.DataFrame:
    if uploaded_file is None:
        return pd.DataFrame(columns=["symbol", "name"])
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        raw = pd.read_csv(uploaded_file)
    elif name.endswith(".xlsx") or name.endswith(".xls"):
        raw = pd.read_excel(uploaded_file)
    else:
        raise ValueError("Unsupported file type. Use CSV or XLSX.")
    cols = {str(c).lower(): c for c in raw.columns}
    symbol_col = cols.get("symbol") or cols.get("ticker") or cols.get("code") or cols.get("kode") or raw.columns[0]
    name_col = cols.get("name") or cols.get("company") or cols.get("company name") or cols.get("nama")
    df = pd.DataFrame(
        {
            "symbol": raw[symbol_col].astype(str).str.strip(),
            "name": raw[name_col].astype(str).str.strip() if name_col else raw[symbol_col].astype(str).str.strip(),
        }
    )
    if market == "ihsg":
        df["symbol"] = df["symbol"].str.upper().str.replace(".JK", "", regex=False) + ".JK"
    else:
        df["symbol"] = df["symbol"].str.upper()
    return df.drop_duplicates("symbol").reset_index(drop=True)



def _to_naive_utc_day(ts) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts.normalize()



def _data_audit(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(
            [{"rows": 0, "start": None, "end": None, "stale_days": None, "last_close": None}]
        )
    start = pd.Timestamp(df.index.min())
    end = pd.Timestamp(df.index.max())
    now_day = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    end_day = _to_naive_utc_day(end)
    stale_days = int((now_day - end_day).days)
    last_close = float(df["Close"].iloc[-1]) if "Close" in df.columns and len(df) else None
    return pd.DataFrame(
        [
            {
                "rows": int(len(df)),
                "start": start,
                "end": end,
                "stale_days": stale_days,
                "last_close": last_close,
            }
        ]
    )



def _candidate_overview(candidate) -> pd.DataFrame:
    if candidate is None:
        return pd.DataFrame()
    rows = []
    for w in candidate.wave_duration_projections:
        rows.append(
            {
                "wave": w.wave_name,
                "actual_bars": w.actual_bars,
                "proj_bars_central": w.projected_bars_central,
                "remaining_bars_central": w.remaining_bars_central,
                "projected_end_central": w.projected_end_timestamp_central,
                "fit_score": w.fit_score,
                "status": w.status,
                "basis": w.basis,
            }
        )
    return pd.DataFrame(rows)



def _target_rows(candidate) -> pd.DataFrame:
    if candidate is None:
        return pd.DataFrame()
    rows = []
    if candidate.invalidation is not None:
        rows.append({"kind": "risk", "name": "Invalidation", "value": candidate.invalidation, "weight": 1.0})
    ppt = candidate.meta.get("primary_price_target")
    pzl = candidate.meta.get("primary_price_zone_low")
    pzh = candidate.meta.get("primary_price_zone_high")
    if ppt is not None:
        rows.append({"kind": "target", "name": "Primary target", "value": ppt, "weight": 1.0})
    if pzl is not None:
        rows.append({"kind": "zone", "name": "Primary zone low", "value": pzl, "weight": 0.95})
    if pzh is not None:
        rows.append({"kind": "zone", "name": "Primary zone high", "value": pzh, "weight": 0.95})
    seen = set()
    for t in sorted(candidate.fib_price_targets, key=lambda x: (-x.weight, abs(x.value - (ppt or x.value)))):
        key = round(float(t.value), 6)
        if key in seen:
            continue
        rows.append({"kind": "price", "name": t.name, "value": t.value, "weight": t.weight})
        seen.add(key)
        if len(rows) >= 10:
            break
    return pd.DataFrame(rows)



def _interval_offset(interval: str, steps: int):
    if interval == "1mo":
        return pd.DateOffset(months=steps)
    if interval == "1wk":
        return pd.Timedelta(days=7 * steps)
    return pd.Timedelta(days=steps)



def _project_right_x(index: pd.DatetimeIndex, interval: str, steps: int = 12):
    last_x = pd.Timestamp(index[-1])
    return last_x + _interval_offset(interval, steps)



def _label_points(candidate, degree: str = "minor"):
    if candidate is None:
        return []
    idxs = candidate.pivot_indices
    prices = candidate.pivot_prices
    out = []
    if candidate.pattern_type in {"impulse", "ending_diagonal"} and len(idxs) >= 6:
        base = ["1", "2", "3", "4", "5"]
        if degree == "primary":
            labels = [f"({x})" for x in base]
            color = "#2d66ff"
        elif degree == "intermediate":
            labels = base
            color = "#111111"
        else:
            labels = ["i", "ii", "iii", "iv", "v"]
            color = "#0b8f3a"
        for idx, price, label in zip(idxs[1:6], prices[1:6], labels):
            out.append((idx, price, label, color))
    elif candidate.pattern_type in {"zigzag", "flat"} and len(idxs) >= 4:
        base = ["A", "B", "C"]
        if degree == "primary":
            labels = [f"({x})" for x in base]
            color = "#2d66ff"
        elif degree == "intermediate":
            labels = base
            color = "#ff3b7f"
        else:
            labels = ["a", "b", "c"]
            color = "#ff3b7f"
        for idx, price, label in zip(idxs[1:4], prices[1:4], labels):
            out.append((idx, price, label, color))
    elif candidate.pattern_type == "triangle" and len(idxs) >= 6:
        base = ["A", "B", "C", "D", "E"]
        if degree == "primary":
            labels = [f"({x})" for x in base]
            color = "#2d66ff"
        elif degree == "intermediate":
            labels = base
            color = "#ff3b7f"
        else:
            labels = ["a", "b", "c", "d", "e"]
            color = "#ff3b7f"
        for idx, price, label in zip(idxs[1:6], prices[1:6], labels):
            out.append((idx, price, label, color))
    return out


_DEGREE_STYLE = {
    "primary": {"color": "#2d66ff", "size": 20, "yshift": 20},
    "intermediate": {"color": "#111111", "size": 16, "yshift": 6},
    "minor": {"color": "#0b8f3a", "size": 13, "yshift": -10},
    "minute": {"color": "#0b8f3a", "size": 11, "yshift": -18},
    "cycle": {"color": "#2d66ff", "size": 22, "yshift": 26},
}


def _candidate_time_window(candidate):
    if candidate is None or not candidate.fib_time_targets:
        return None, None
    top = sorted(candidate.fib_time_targets, key=lambda t: (-t.weight, t.projected_index))[:3]
    pts = [pd.Timestamp(t.projected_timestamp) for t in top if t.projected_timestamp is not None]
    if not pts:
        return None, None
    return min(pts), max(pts)


def plot_reference_chart(df: pd.DataFrame, result, interval: str, lookback_bars: int = 250, degree_summary=None, degree_results: dict | None = None):
    candidate = result.best_candidate
    view = df.tail(lookback_bars).copy()
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=view.index,
            open=view["Open"],
            high=view["High"],
            low=view["Low"],
            close=view["Close"],
            name="Price",
            increasing_line_color="#6b6b6b",
            decreasing_line_color="#6b6b6b",
            increasing_fillcolor="rgba(0,0,0,0)",
            decreasing_fillcolor="rgba(0,0,0,0)",
            whiskerwidth=0.2,
            showlegend=False,
        )
    )

    last_close = float(view["Close"].iloc[-1])
    x0 = view.index[0]
    x1 = _project_right_x(view.index, interval, steps=22)
    fig.add_hline(y=last_close, line_width=1, line_dash="dot", line_color="#7d7d7d")

    if candidate is not None:
        vis_x = []
        vis_y = []
        for idx, price in zip(candidate.pivot_indices, candidate.pivot_prices):
            if 0 <= idx < len(df):
                ts = df.index[idx]
                if ts >= x0:
                    vis_x.append(ts)
                    vis_y.append(price)
        if len(vis_x) >= 2:
            fig.add_trace(
                go.Scatter(
                    x=vis_x,
                    y=vis_y,
                    mode="lines",
                    line=dict(color="#7d7d7d", width=1.0),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

    if degree_summary is not None and degree_results:
        for dv in degree_summary.degree_views:
            src_result = degree_results.get(dv.interval)
            src_candidate = None if src_result is None else src_result.best_candidate
            if src_candidate is None:
                continue
            style = _DEGREE_STYLE.get(dv.degree, _DEGREE_STYLE["minor"])
            label_points = _label_points(src_candidate, degree=dv.degree)
            pivot_ts = list(src_candidate.pivot_timestamps[1 : 1 + len(label_points)])
            for (idx, price, label, _), ts in zip(label_points, pivot_ts):
                ts = pd.Timestamp(ts)
                if ts >= x0:
                    fig.add_annotation(
                        x=ts,
                        y=price,
                        text=label,
                        font=dict(color=style["color"], size=style["size"]),
                        xanchor="center",
                        yanchor="bottom",
                        yshift=style["yshift"],
                        showarrow=False,
                        bgcolor="rgba(255,255,255,0)",
                    )

    target_rows = _target_rows(candidate)
    core_targets = target_rows[target_rows["kind"].isin(["target", "zone", "risk", "price"])].copy()
    if not core_targets.empty:
        core_targets["priority"] = core_targets["value"].apply(lambda x: abs(float(x) - last_close))
        core_targets = core_targets.sort_values(["kind", "weight", "priority"], ascending=[True, False, True]).head(5)
        for _, row in core_targets.iterrows():
            value = float(row["value"])
            name = str(row["name"])
            color = "#2d66ff" if value >= last_close else "#ff3b7f"
            if name == "Invalidation":
                color = "#111111"
            fig.add_shape(
                type="line",
                x0=view.index[-1],
                x1=x1,
                y0=value,
                y1=value,
                line=dict(color=color, width=1.4, dash="dot"),
            )
            fig.add_annotation(
                x=x1,
                y=value,
                text=f"({value:,.0f})",
                font=dict(color=color, size=16),
                xanchor="left",
                yanchor="middle",
                showarrow=False,
            )

    if candidate is not None:
        tws, twe = _candidate_time_window(candidate)
        if tws is not None and twe is not None:
            fig.add_vrect(x0=tws, x1=twe, fillcolor="#bdbdbd", opacity=0.08, line_width=0)
        if candidate.invalidation is not None:
            fig.add_annotation(
                x=x1,
                y=float(candidate.invalidation),
                text=f"invalid {float(candidate.invalidation):,.0f}",
                font=dict(color="#111111", size=12),
                xanchor="left",
                yanchor="bottom",
                showarrow=False,
            )

    title = f"{result.symbol} | {interval}"
    if degree_summary is not None:
        title += f" | {degree_summary.state}"
    fig.update_layout(
        title=title,
        height=680,
        margin=dict(l=10, r=120, t=30, b=10),
        paper_bgcolor="#e9e9e9",
        plot_bgcolor="#e9e9e9",
        xaxis_rangeslider_visible=False,
        showlegend=False,
        font=dict(color="#222222"),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, showline=False, ticks="", range=[view.index[0], x1])
    fig.update_yaxes(showgrid=False, zeroline=False, showline=False, side="right")
    return fig


def _build_engine(min_reversal_pct, atr_mult, max_pivots, top_n, candidate_lookback_pivots):
    return ElliottWaveEngine(
        min_reversal_pct=min_reversal_pct,
        atr_mult=atr_mult,
        max_pivots=max_pivots,
        top_n=top_n,
        candidate_lookback_pivots=candidate_lookback_pivots,
    )


def _candidate_pool(result):
    pool = []
    if result.best_candidate is not None:
        pool.append(result.best_candidate)
    for alt in result.alternate_candidates:
        if alt is not None:
            pool.append(alt)
    uniq = []
    seen = set()
    for c in pool:
        key = (c.pattern_type, c.direction, tuple(c.pivot_indices))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(c)
    return uniq


def _anchored_candidate_score(candidate, anchor_direction=None, anchor_pattern=None):
    score = float(candidate.score) * 0.75 + float(candidate.confidence) * 0.25
    if anchor_direction is not None:
        score += 0.10 if candidate.direction == anchor_direction else -0.08
    if anchor_pattern is not None:
        score += 0.04 if candidate.pattern_type == anchor_pattern else 0.0
    if candidate.hard_rule_pass:
        score += 0.03
    if candidate.meta.get("recency_score") is not None:
        score += 0.04 * float(candidate.meta.get("recency_score"))
    if candidate.meta.get("anchor_aligned_promoted"):
        score += 0.01
    return score


def _promote_anchor_aligned_candidates(results):
    if not results:
        return results
    ordered = sorted(results, key=lambda r: {"1mo": 0, "1wk": 1, "1d": 2, "4h": 3, "1h": 4}.get(r.interval, 99))
    anchor_direction = None
    anchor_pattern = None
    for result in ordered:
        pool = _candidate_pool(result)
        if not pool:
            continue
        current = result.best_candidate
        if anchor_direction is None and current is not None:
            anchor_direction = current.direction
            anchor_pattern = current.pattern_type
            continue
        scored = sorted(pool, key=lambda c: _anchored_candidate_score(c, anchor_direction, anchor_pattern), reverse=True)
        chosen = scored[0]
        if current is None or chosen is not current:
            chosen.meta["anchor_aligned_promoted"] = 1.0
            result.best_candidate = chosen
            result.alternate_candidates = [c for c in pool if c is not chosen][: max(0, len(result.alternate_candidates))]
        if chosen is not None:
            if anchor_direction is None:
                anchor_direction = chosen.direction
            elif chosen.direction == anchor_direction and chosen.confidence >= 0.45:
                anchor_direction = chosen.direction
            if anchor_pattern is None and chosen is not None:
                anchor_pattern = chosen.pattern_type
    return results


st.title("Elliott Wave Miner Engine")
st.caption(
    "Scanner + single-chart workflow with hierarchical multi-degree counting. Visual leans closer to analyst-style Elliott charts: clean candles, degree labels, invalidation, and target lines."
)

with st.sidebar:
    st.header("Controls")
    mode = st.radio("Mode", ["Single chart", "Scanner"], index=0)
    market = st.selectbox("Market", ["ihsg", "us_stocks", "forex", "commodities", "crypto"], index=0)
    us_common_only = st.checkbox("US common-only filter", value=False) if market == "us_stocks" else False
    allow_unverified_third_party = (
        st.checkbox(
            "Allow unverified third-party universe fallback",
            value=False,
            help="Useful when official IDX blocks hosted requests. Keep off if you want stricter source hygiene.",
        )
        if market == "ihsg"
        else False
    )
    uploaded_universe = st.file_uploader(
        "Optional universe CSV/XLSX override",
        type=["csv", "xlsx", "xls"],
        help="Upload an official/exported universe file if you want full control over the dropdown list.",
    )

    universe_payload = load_market_universe(market, us_common_only, allow_unverified_third_party)
    raw_universe = universe_payload["df"].copy()

    if uploaded_universe is not None:
        try:
            raw_universe = _normalize_uploaded_universe(uploaded_universe, market)
            st.caption("Universe source: uploaded override file")
            st.caption(f"Rows loaded: {len(raw_universe):,}")
        except Exception as e:
            st.error(f"Uploaded universe file could not be parsed: {e}")
    else:
        st.caption(f"Universe source: {universe_payload['source']}")
        flags = ["live" if universe_payload["is_live"] else "fallback"]
        if universe_payload["is_complete"] is True:
            flags.append("full/near-full")
        elif universe_payload["is_complete"] is False:
            flags.append("partial/curated")
        st.caption(" | ".join(flags))
        with st.expander("Universe notes", expanded=False):
            for note in universe_payload["notes"]:
                st.write(f"- {note}")

    filter_text = st.text_input("Filter universe list", value="")
    filtered_universe = raw_universe.copy()
    if not filtered_universe.empty and filter_text.strip():
        q = filter_text.strip().upper()
        filtered_universe = filtered_universe[
            filtered_universe["symbol"].astype(str).str.upper().str.contains(q, na=False)
            | filtered_universe["name"].astype(str).str.upper().str.contains(q, na=False)
        ]
    st.caption(f"Filtered universe rows: {len(filtered_universe):,}")

    default_symbol = _default_symbol(market)
    max_dropdown_rows = st.slider("Max dropdown rows", 100, 5000, 1000, 100)
    display_universe = filtered_universe.head(max_dropdown_rows) if not filtered_universe.empty else filtered_universe
    symbol_options: List[str] = display_universe["symbol"].tolist() if not display_universe.empty else []
    selected_index = symbol_options.index(default_symbol) if default_symbol in symbol_options else (0 if symbol_options else None)
    selected_symbol = st.selectbox("Symbol from universe", symbol_options, index=selected_index) if symbol_options else default_symbol
    custom_symbol = st.text_input("Symbol to analyze", value=selected_symbol)
    symbol = format_symbol_for_market(custom_symbol or selected_symbol, market)

    if mode == "Single chart":
        intervals = st.multiselect("Intervals", ["1d", "1wk", "1mo"], default=["1d", "1wk", "1mo"])
        period = st.selectbox("Period", ["2y", "5y", "10y", "max"], index=2)
        lookback_bars = st.slider("Chart lookback bars", 80, 600, 250, 10)
    else:
        scan_interval = st.selectbox("Scanner interval", ["1d", "1wk", "1mo"], index=0)
        period = st.selectbox("Scanner period", ["2y", "5y", "10y", "max"], index=2)
        lookback_bars = st.slider("Preview chart lookback bars", 80, 600, 220, 10)
        scan_max_workers = st.slider("Scanner max workers", 2, 32, 8, 1)
        scan_limit = st.number_input(
            "How many filtered symbols to scan",
            min_value=1,
            max_value=max(1, int(len(filtered_universe) if not filtered_universe.empty else 1)),
            value=max(1, int(len(filtered_universe) if not filtered_universe.empty else 1)),
            step=1,
        )

    min_reversal_pct = st.slider("Min reversal %", 0.005, 0.15, 0.03, 0.005)
    atr_mult = st.slider("ATR multiplier", 0.5, 4.0, 1.5, 0.1)
    max_pivots = st.slider("Max pivots", 20, 150, 80, 5)
    candidate_lookback_pivots = st.slider("Candidate lookback pivots", 8, 50, 25, 1)
    top_n = st.slider("Top alternate counts", 1, 10, 5, 1)
    run = st.button("Run", type="primary")


if run:
    if mode == "Single chart":
        if not intervals:
            st.error("Pick at least one interval.")
            st.stop()
        if not symbol:
            st.error("Enter a symbol first.")
            st.stop()

        engine = _build_engine(min_reversal_pct, atr_mult, max_pivots, top_n, candidate_lookback_pivots)
        results = []
        data_map = {}
        fetch_errors = []

        with st.spinner("Running multi-timeframe analysis..."):
            for interval in intervals:
                try:
                    df = fetch_data(symbol, interval, period)
                    data_map[interval] = df
                    results.append(engine.analyze(df, symbol=symbol, market=market, interval=interval))
                except Exception as e:
                    fetch_errors.append(f"{interval}: {e}")

        if fetch_errors:
            st.warning("Some intervals failed: " + " | ".join(fetch_errors))
        if not results:
            st.error("No intervals could be analyzed. Check the symbol/provider format or try another period.")
            st.stop()

        results = _promote_anchor_aligned_candidates(results)
        mtf = reconcile_results(results)
        degree_summary = build_degree_hierarchy(results)
        degree_results = {r.interval: r for r in results}

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Consensus direction", mtf.consensus_direction or "n/a")
        c2.metric("Consensus pattern", mtf.consensus_pattern or "n/a")
        c3.metric("Alignment score", f"{mtf.alignment_score:.2f}")
        c4.metric("State", mtf.state)

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Hierarchy state", degree_summary.state)
        c6.metric("Hierarchy agreement", f"{degree_summary.agreement_score:.2f}")
        c7.metric("Inherited alignment", f"{degree_summary.inherited_alignment_score:.2f}")
        c8.metric("Consensus invalidation", "-" if mtf.invalidation is None else f"{mtf.invalidation:,.4f}")

        c9, c10 = st.columns(2)
        c9.metric("Consensus target", "-" if mtf.target_price is None else f"{mtf.target_price:,.4f}")
        c10.metric(
            "Target zone",
            "-"
            if mtf.target_zone_low is None or mtf.target_zone_high is None
            else f"{mtf.target_zone_low:,.4f} → {mtf.target_zone_high:,.4f}",
        )

        if mtf.target_time_window_start is not None or mtf.target_time_window_end is not None:
            st.info(f"Consensus time window: {mtf.target_time_window_start} → {mtf.target_time_window_end}")

        st.subheader("Hierarchical degree summary")
        st.dataframe(hierarchy_frame(degree_summary), use_container_width=True, hide_index=True)
        for note in degree_summary.notes:
            st.caption(note)

        st.subheader("Timeframe summary")
        tf_rows = []
        for v in mtf.timeframe_views:
            tf_rows.append(
                {
                    "interval": v.interval,
                    "direction": v.direction,
                    "pattern": v.pattern_type,
                    "score": v.score,
                    "confidence": v.confidence,
                    "invalidation": v.invalidation,
                    "target_price": v.price_target,
                    "target_zone_low": v.price_zone_low,
                    "target_zone_high": v.price_zone_high,
                    "time_window_start": v.time_window_start,
                    "time_window_end": v.time_window_end,
                }
            )
        st.dataframe(pd.DataFrame(tf_rows), use_container_width=True, hide_index=True)

        base_interval = "1d" if "1d" in data_map else results[0].interval
        st.subheader(f"Reference chart: {base_interval} with multi-degree overlay")
        st.plotly_chart(
            plot_reference_chart(data_map[base_interval], degree_results[base_interval], base_interval, lookback_bars=lookback_bars, degree_summary=degree_summary, degree_results=degree_results),
            use_container_width=True,
            key=f"ref-{symbol}-{base_interval}-{len(results)}",
        )

        tabs = st.tabs([r.interval for r in results])
        for tab, result in zip(tabs, results):
            with tab:
                candidate = result.best_candidate
                df = data_map[result.interval]
                if candidate is not None and candidate.meta.get("anchor_aligned_promoted"):
                    st.caption("Best count was upgraded from alternates because it aligns better with the higher-degree anchor.")
                st.plotly_chart(
                    plot_reference_chart(df, result, result.interval, lookback_bars=lookback_bars, degree_summary=degree_summary, degree_results=degree_results),
                    use_container_width=True,
                    key=f"tab-{symbol}-{result.interval}",
                )
                st.caption("Fetched data audit")
                st.dataframe(_data_audit(df), use_container_width=True, hide_index=True)

                if candidate is None:
                    st.warning("No valid candidate found for this interval.")
                    continue

                a, b, c = st.columns(3)
                a.metric("Pattern", candidate.pattern_type)
                b.metric("Direction", candidate.direction)
                c.metric("Score / Confidence", f"{candidate.score:.2f} / {candidate.confidence:.2f}")

                d, e, f = st.columns(3)
                d.metric("Invalidation", "-" if candidate.invalidation is None else f"{candidate.invalidation:,.4f}")
                ppt = candidate.meta.get("primary_price_target")
                e.metric("Primary target", "-" if ppt is None else f"{ppt:,.4f}")
                pzl = candidate.meta.get("primary_price_zone_low")
                pzh = candidate.meta.get("primary_price_zone_high")
                f.metric(
                    "Primary zone",
                    "-" if pzl is None or pzh is None else f"{pzl:,.4f} → {pzh:,.4f}",
                )

                st.subheader("Key targets")
                st.dataframe(_target_rows(candidate), use_container_width=True, hide_index=True)

                st.subheader("Wave duration projections")
                st.dataframe(_candidate_overview(candidate), use_container_width=True, hide_index=True)

                if candidate.momentum_notes:
                    st.subheader("Momentum / structure notes")
                    for note in candidate.momentum_notes:
                        st.write(f"- {note}")

                alt_rows = []
                for alt in result.alternate_candidates:
                    alt_rows.append(
                        {
                            "pattern": alt.pattern_type,
                            "direction": alt.direction,
                            "score": alt.score,
                            "confidence": alt.confidence,
                            "invalidation": alt.invalidation,
                            "primary_price_target": alt.meta.get("primary_price_target"),
                        }
                    )
                if alt_rows:
                    st.subheader("Alternate counts")
                    st.dataframe(pd.DataFrame(alt_rows), use_container_width=True, hide_index=True)

    else:
        if filtered_universe.empty:
            st.error("Universe is empty. Upload a CSV/XLSX or change the filters/source.")
            st.stop()
        scan_symbols = tuple(
            format_symbol_for_market(s, market)
            for s in filtered_universe["symbol"].astype(str).head(int(scan_limit)).tolist()
        )
        with st.spinner(f"Scanning {len(scan_symbols):,} symbols..."):
            scan_df = scan_universe_cached(
                scan_symbols,
                market,
                scan_interval,
                period,
                min_reversal_pct,
                atr_mult,
                max_pivots,
                top_n,
                candidate_lookback_pivots,
                scan_max_workers,
            )

        if scan_df.empty:
            st.error("No scan output returned.")
            st.stop()

        ranked = scan_df[scan_df["error"].isna()].copy()
        ranked = ranked.sort_values(["score", "confidence"], ascending=[False, False])
        errors = scan_df[scan_df["error"].notna()].copy()

        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Scanned symbols", f"{len(scan_df):,}")
        s2.metric("Valid candidates", f"{len(ranked):,}")
        s3.metric("Failed fetch/analyze", f"{len(errors):,}")
        s4.metric("Top score", "-" if ranked.empty else f"{ranked['score'].iloc[0]:.2f}")

        st.subheader("Scanner ranking")
        display_cols = [
            "symbol",
            "pattern",
            "direction",
            "score",
            "confidence",
            "invalidation",
            "primary_price_target",
            "primary_price_zone_low",
            "primary_price_zone_high",
            "primary_time_window_start",
            "primary_time_window_end",
            "bars_since_last_pivot",
            "recency_score",
        ]
        st.dataframe(ranked[display_cols], use_container_width=True, hide_index=True)

        if not errors.empty:
            with st.expander("Scanner failures", expanded=False):
                st.dataframe(errors[["symbol", "error"]], use_container_width=True, hide_index=True)

        preview_default = ranked["symbol"].iloc[0] if not ranked.empty else symbol
        preview_symbol = st.selectbox(
            "Preview scanned symbol",
            ranked["symbol"].tolist() if not ranked.empty else [symbol],
            index=0,
        ) if not ranked.empty else preview_default

        try:
            preview_df = fetch_data(preview_symbol, scan_interval, period)
            engine = _build_engine(min_reversal_pct, atr_mult, max_pivots, top_n, candidate_lookback_pivots)
            preview_result = engine.analyze(preview_df, symbol=preview_symbol, market=market, interval=scan_interval)
            st.subheader(f"Preview: {preview_symbol}")
            st.plotly_chart(
                plot_reference_chart(preview_df, preview_result, scan_interval, lookback_bars=lookback_bars),
                use_container_width=True,
                key=f"preview-{preview_symbol}-{scan_interval}",
            )
            st.caption("Fetched data audit")
            st.dataframe(_data_audit(preview_df), use_container_width=True, hide_index=True)
        except Exception as e:
            st.warning(f"Preview fetch/analyze failed for {preview_symbol}: {e}")
else:
    st.info("Set your controls, then click Run.")
