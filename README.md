# pumpfun-top5-forecast-01 entry

Forecasts the 5 largest pump.fun coins by market cap at each answer read.

## How it works

`main.py` (Python 3.13, standard library only, no network) reads
`/input/task.json` and `/input/history.ndjson` and writes
`/output/forecast.json` (`$MAKEFUN_INPUT_DIR` / `$MAKEFUN_OUTPUT_DIR`
override the paths outside the grader).

For every coin in the history it tracks the trajectory of log10 market
cap (SOL) across the 10-minute marks, fits a shrunk recent drift and a
per-coin volatility, then Monte-Carlo forecasts (fixed seed) each coin's
log-cap at every answer read in `task.json`, with:

- drift that decays with horizon (half-life ~1.5 h),
- uncertainty that widens with horizon,
- a per-venue "death" mixture (pool 0.10/h, curve 0.30/h, migrating
  0.20/h) — most pump.fun coins rug or crash to a few SOL,
- a common market shift shared by all coins (widens the size
  distribution without changing the order).

Place probabilities and size histograms are the Monte-Carlo frequencies
averaged over the reads, scaled down to reserve mass for newcomers
(coins born after the cut, which the forecast cannot name but which take
roughly a third of top-5 slots even at a 10-minute horizon, and over
half at a 7–13 hour horizon). The unnamed share of each place is what
the scorer gives those newcomers and the un-named tail of known coins.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Deterministic: one fixed seed, sorted iteration order, no wall clock.
Runs in ~15–45 s on the practice input, far inside the 300 s limit.

## Local checks (no local grade exists for this competition)

- `python3 /tmp/pumpfun-test/validate.py out/forecast.json input/task.json`
  — strict format check (keys, row shapes, sums ≤ 1, bins, top5).
- Backtest on truncated history (marks to 20:30, pseudo-reads 20:40 and
  20:50): log score ≈ −30.2/−30.3 vs house baseline −33.5 on the same
  two reads (order ≈ −16.3 vs −18.6, sizes ≈ −14.0 vs −14.9).
  The backtest horizon is only 10–20 minutes; the final horizon is
  7–13 hours, where the model's newcomer reserve and widening
  uncertainty matter most.
