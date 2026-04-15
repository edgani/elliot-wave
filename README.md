# Elliott Wave Miner-Style Python Engine — Ultra Hardening Pass

This build upgrades the prior version in the ways that matter most for real trading use:
- all-wave duration projections remain available
- time projections now also carry **projected timestamps/date windows**, not just bar counts
- score ranking is now **recency-aware**, so stale counts are penalized
- correction taxonomy is broader: **flat** support was added on top of impulse / zigzag / triangle / ending diagonal proxy
- scanner output now includes **primary price zone** and **primary time window** columns for quicker decision review

## What changed in this pass

### 1) Explicit date-window timing
Both `fib_time_targets` and `wave_duration_projections` now try to map projected bar counts into real projected timestamps when the input index is datetime-like.

### 2) Recency-aware ranking
A real scanner should not rank an excellent but stale count above a slightly weaker count that finished much more recently. This build now adds:
- `bars_since_last_pivot`
- `recency_score`
- final score blending base structure quality with recency

### 3) Flat correction support
The engine now evaluates a simple flat family:
- deep B retrace
- C termination projected from A
- duration logic reused from corrective ABC timing

### 4) Scanner-ready summary fields
Best-candidate metadata now includes:
- `primary_price_target`
- `primary_price_zone_low`
- `primary_price_zone_high`
- scanner table time window columns

## Important honesty note
This is still not an honest claim of perfect Elliott labeling or exact Dynamic Traders proprietary replication. It is a more decision-useful open implementation that moves closer to actual trading workflow.


## Main paths
- Core engine: `src/elliott_miner_engine/engine.py`
- Multi-timeframe reconciliation: `src/elliott_miner_engine/mtf.py`
- CLI entry: `src/elliott_miner_engine/cli.py`
- Streamlit app entry: `app.py`

## Run with Streamlit
From the project root:

```bash
streamlit run app.py
```

## What multi-timeframe reconciliation does
The new reconciliation layer combines `1d`, `1wk`, and `1mo` outputs into a weighted consensus:
- direction vote
- pattern vote
- alignment score
- conflict score
- consensus invalidation
- consensus price target / zone
- consensus time window

This does **not** pretend there is one perfect count. It explicitly measures whether the lower and higher timeframes agree enough to trust the setup.


## Universe/data source behavior in the Streamlit app
- The app no longer crashes if a live universe source fails.
- IHSG tries the official IDX stock-list page first. If IDX blocks the request, it falls back to a GitHub mirror snapshot, then finally to a bundled local sample.
- US stocks try the live official Nasdaq Trader symbol directories first, then fall back to a bundled local sample.
- Crypto tries the live CoinGecko coin list for discovery, but analysis still uses the symbol you type and Yahoo-compatible symbols work best in the current build.
- Forex and commodities are curated Yahoo-compatible lists, not complete global master universes.

## Honesty on completeness
- US can be near-full when the live Nasdaq Trader directory is reachable.
- IHSG can be near-full only when the official IDX page is reachable or the mirror snapshot is available; otherwise the bundled local fallback is only a sample.
- Crypto discovery can be very broad through CoinGecko, but price-history execution in this build is still most stable with Yahoo-style symbols like `BTC-USD`.
