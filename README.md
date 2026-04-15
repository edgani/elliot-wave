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
