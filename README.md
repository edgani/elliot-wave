# Elliott Wave Miner-Style Python Engine

This project is a from-scratch Python implementation of a **practical Elliott Wave engine** inspired by public material from Robert Miner / Dynamic Traders:
- simplified Elliott Wave pattern-position logic
- alternate counts rather than one forced count
- Fibonacci price projection zones
- Fibonacci time projection zones
- explicit per-wave duration projection layer
- momentum confirmation layer
- multi-market scanning architecture

## What it does well

- Detects **candidate impulse waves**, **zigzags**, **triangles**, and a simple **ending diagonal proxy**.
- Uses a **causal ATR-adaptive zigzag pivot extractor** rather than hard-coded fixed swings.
- Scores each candidate with:
  - hard Elliott rules
  - Miner-style Fib confluence
  - momentum behavior consistency
- Produces:
  - best count
  - alternate counts
  - invalidation level
  - projected price zones
  - projected time windows
  - projected duration profile for each wave (W1-W5 / ABC / triangle legs)
- Scales to scanners across:
  - IHSG
  - US stocks
  - forex
  - commodities
  - crypto

## What it does **not** guarantee

Automatic Elliott Wave labeling is inherently uncertain. There is no honest way to promise “perfect” or fully objective wave labeling on every chart. This engine is therefore built as a **probabilistic ranking system**, not a fake deterministic oracle.

That matters because the real problem is not “can we label every chart?” but:
1. what is the **highest-probability current pattern-position**,
2. what invalidates it,
3. what are the likely price and time target zones,
4. what are the strongest alternates.

## Miner-style public logic implemented here

### 1. Pattern first
The engine starts with the most practical question:
- is price making a **trend** or a **correction**?
- is the move more consistent with a **5-wave** motive structure or a **3-wave** correction?

### 2. Fibonacci price logic
Public Dynamic Traders material emphasizes:
- **internal retracements**
- **external retracements**
- **alternate price projections (APP)**
- **end-of-wave target zones**

Implemented examples:
- Wave 2 / B retracement emphasis: `50% / 61.8% / 78.6%`
- Wave 3 / C APP emphasis: `62% / 100% / 162% / 262%`
- Wave 4 retracement emphasis: `38.2% / 50% / 61.8%`
- Wave 5 emphasis:
  - ext ret of Wave 4: `127% / 162% / 262%`
  - APP of Wave 1: `61.8% / 100% / 162%`
  - APP of Waves 1-3: `38.2% / 61.8% / 100%`

### 3. Fibonacci time logic
Robert Miner’s public material clearly emphasizes Fib time targets, time bands, and dynamic time targets. The exact proprietary internals are not fully public, so this code implements an **explicit approximation**:
- project future time windows from prior swing durations using:
  - `0.382`
  - `0.618`
  - `1.0`
  - `1.618`
  - `2.618`
- rank windows by confluence rather than pretending we know the proprietary weighting exactly.

### 4. Per-wave duration projection
This build now separates two different time outputs:
- **generic Fib time targets** for the next likely timing window
- **explicit wave-duration projections** for each identified wave

Examples:
- impulse: `W2, W3, W4, W5` duration estimates from prior-wave time ratios
- zigzag: `B, C` duration estimates from Wave A
- triangle: `B, C, D, E` duration estimates from the prior triangle leg

Important honesty note: these are still **explicit approximations of public Miner-style time logic**, not a claim of exact replication of proprietary DT timing internals.

### 5. Momentum confirmation
Public Dynamic Traders material repeatedly stresses that wave, price, and time are not enough by themselves. This build adds a lightweight momentum layer:
- RSI
- ROC
- ATR-normalized wave strength
- Wave 3 vs Wave 1 vs Wave 5 momentum comparison
- simple Wave 5 divergence notes

## Multi-market universe support

### IHSG
`ExchangeUniverseLoader.load_idx()` pulls the listed stock table from IDX and appends `.JK`.

### US stocks
`ExchangeUniverseLoader.load_us_equities()` uses Nasdaq Trader symbol-directory files.
This covers Nasdaq-listed issues and “other listed” issues distributed through Nasdaq’s official symbol directory.

### Forex and commodities
A default universe is included, but unlike listed equities, there is no single canonical “all symbols” source here. Treat these universes as **configurable provider universes**.

### Crypto
`CoinGeckoUniverseLoader.load()` can load the full CoinGecko coin map.
The included CLI uses major Yahoo-format crypto pairs by default for OHLC convenience, but the architecture is ready for a dedicated CoinGecko/CCXT OHLC adapter if you want full-universe crypto scanning.

## Install

```bash
pip install -r requirements.txt
pip install -e .
```

## Single-symbol example

```bash
python -m elliott_miner_engine.cli analyze --symbol BTC-USD --market crypto --period 5y --plot btc.png
```

## Scanner example

```bash
python -m elliott_miner_engine.cli scan --market ihsg --limit 200 --out ihsg_scan.csv
python -m elliott_miner_engine.cli scan --market us_stocks --limit 1000 --out us_scan.csv
```

## CLI output note

`analyze` JSON output now includes `wave_duration_projections` in addition to generic `time_targets`.

## Where to take it next

If the goal is a serious production-grade engine, the next upgrades should be:
1. Multi-timeframe top-down count reconciliation
2. Hidden Markov / particle filtering over competing wave states
3. Bayesian alternate-count ranking
4. Dedicated crypto OHLC adapter
5. Better correction taxonomy: flats, double zigzags, combinations
6. Dynamic time-band calibration by market and timeframe
7. Walk-forward validation by pattern family and regime
8. Dashboard layer for scanner + chart review

## Honesty note

This code is a solid foundation, but it is **not** a claim of exact replication of Robert Miner’s proprietary DT software. It is a practical open implementation based on public material plus explicit engineering assumptions where the public material is incomplete.
