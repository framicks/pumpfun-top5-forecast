# pumpfun-top5-forecast-01 entry (Python 3.13, standard library only).
#
# Model: for every coin seen in the history, track its log10 market cap
# (SOL) trajectory across the 10-minute marks.  Fit a shrunk recent drift
# and a per-coin volatility, then Monte-Carlo (fixed seed) each coin's
# log-cap at every answer read in task.json, with a drift that decays with
# horizon, uncertainty that widens with horizon, and a per-venue "death"
# mixture (most pump.fun coins rug/crash to a few SOL).  Place
# probabilities and size histograms are the Monte-Carlo frequencies
# averaged over the reads, scaled down to reserve mass for newcomers
# (coins born after the cut, which the forecast cannot name but which
# take a large share of top-5 slots at long horizons).  The unnamed share
# of each place is what the scorer gives those newcomers (and the
# un-named tail of known coins).
#
# Reads task.json for places / size edges / reads / output name and the
# horizon; never hardcodes times.  Deterministic: one fixed seed, sorted
# iteration order, no wall-clock use.

import json
import math
import os
import random
import sys
from datetime import datetime, timezone

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

SEED = 20261112
MC_SAMPLES = 4000          # Monte-Carlo draws per answer read
MAX_COINS = 400            # coins modelled (top by last log-cap)
LOOKBACK_MARKS = 12        # marks of history used for the coin universe
DRIFT_WINDOW = 6           # marks used for the recent-drift fit
DRIFT_SHRINK = 0.4         # shrink the fitted drift toward 0
DRIFT_TAU_H = 1.5          # drift decays: mean += drift*tau*(1-exp(-h/tau))
DRIFT_CLAMP = 0.30         # dex per hour, max |drift|
VOL_FLOOR = 0.06           # per-10-min log-return sd floor
VOL_CAP = 0.35             # per-10-min log-return sd cap
VOL_DEFAULT = 0.12         # when a coin has too few observations
BASE_SD = 0.15             # extra sd on every forecast (model error)
SD_CAP = 1.2               # max forecast sd (dex)
DEATH_LAMBDA = {"pool": 0.10, "curve": 0.30, "migrating": 0.20}  # per hour
DEATH_MEAN = 0.5           # log10 SOL of a dead coin (~3 SOL)
DEATH_SD = 0.6
NEWCOMER_R_MIN = {1: 0.30, 2: 0.28, 3: 0.26, 4: 0.24, 5: 0.22}  # floor: newcomers take ~30% of top-5 slots even at 10-min horizon (practice + backtest)
NEWCOMER_R_MAX = {1: 0.55, 2: 0.50, 3: 0.45, 4: 0.40, 5: 0.35}  # long-horizon ceiling
NEWCOMER_TAU_H = 4.0       # newcomer share grows with horizon
SIZE_UNNAMED = 0.10        # mass left unnamed in every place's sizes
MIN_P = 0.001              # smallest named probability (after scaling)
MAX_ROWS_PER_PLACE = 60    # named coins per place (limit is 100)
MKT_DRIFT = 0.03           # dex/hour: the top-5 caps drift up as fresh pumps arrive
MKT_SD = 0.20              # dex/sqrt(hour): common market shift, shared by all coins
MICRO = 1_000_000


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def logcap_sol(lamports):
    return math.log10(float(lamports)) - 9.0


def erfc(x):
    z = abs(x)
    t = 1.0 / (1.0 + 0.5 * z)
    r = t * math.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))))
    return r if x >= 0 else 2.0 - r


def above(x, mean, sd):
    if x == float("-inf"):
        return 1.0
    if x == float("inf"):
        return 0.0
    return 0.5 * erfc((x - mean) / (sd * math.sqrt(2.0)))


def main():
    task = json.load(open(os.path.join(INPUT_DIR, "task.json")))
    places = int(task["places"])
    edges = [float(e) for e in task["size"]["edges"]]
    reads = [parse_time(r) for r in task["reads"]]
    out_name = task.get("output", "forecast.json")

    marks = []
    with open(os.path.join(INPUT_DIR, "history.ndjson")) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = json.loads(line)
            if m.get("ok") is True and isinstance(m.get("rows"), list) and m["rows"]:
                marks.append(m)
    if not marks:
        raise SystemExit("the history holds no ok mark with rows")

    mark_times = [parse_time(m["at"]) for m in marks]
    last_t = mark_times[-1]
    horizons_h = [max(0.0, (r - last_t).total_seconds() / 3600.0) for r in reads]

    # ---- per-coin series -------------------------------------------------
    # mint -> {"pts": [(t_min, logcap)], "venue": str, "complete": bool,
    #          "mayhem": bool, "created": datetime|None, "last_rank": int}
    coins = {}
    t0 = mark_times[0]
    for mi, m in enumerate(marks):
        t_min = (mark_times[mi] - t0).total_seconds() / 60.0
        for row in m["rows"]:
            mint = row["mint"]
            c = coins.get(mint)
            if c is None:
                c = coins[mint] = {"pts": [], "venue": row.get("venue", "curve"),
                                   "complete": bool(row.get("complete")),
                                   "mayhem": bool(row.get("mayhem")),
                                   "created": None, "last_rank": 10 ** 9}
                try:
                    c["created"] = parse_time(row["created_at"])
                except Exception:
                    c["created"] = None
            c["pts"].append((t_min, logcap_sol(row["market_cap_lamports"])))
            c["venue"] = row.get("venue", c["venue"])
            c["complete"] = bool(row.get("complete"))
            c["mayhem"] = bool(row.get("mayhem"))
            c["last_rank"] = int(row.get("rank", 10 ** 9))

    # ---- coin universe: seen in the last LOOKBACK_MARKS + ever top-10 ----
    recent = set()
    for m in marks[-LOOKBACK_MARKS:]:
        for row in m["rows"]:
            recent.add(row["mint"])
    ever_top = set()
    for m in marks:
        for row in m["rows"][:10]:
            ever_top.add(row["mint"])
    universe = recent | ever_top
    # keep the MAX_COINS with the highest last log-cap
    scored = []
    for mint in universe:
        c = coins[mint]
        scored.append((c["pts"][-1][1], mint))
    scored.sort(reverse=True)
    universe = [mint for _, mint in scored[:MAX_COINS]]
    universe.sort()  # deterministic iteration order

    # ---- per-coin forecast parameters ------------------------------------
    params = {}
    for mint in universe:
        c = coins[mint]
        pts = c["pts"]
        x0 = pts[-1][1]
        # recent drift: OLS slope over the last DRIFT_WINDOW points
        wpts = pts[-DRIFT_WINDOW:]
        drift = 0.0
        if len(wpts) >= 3:
            n = len(wpts)
            mt = sum(p[0] for p in wpts) / n
            mx = sum(p[1] for p in wpts) / n
            num = sum((p[0] - mt) * (p[1] - mx) for p in wpts)
            den = sum((p[0] - mt) ** 2 for p in wpts)
            if den > 0:
                slope_per_min = num / den
                drift = max(-DRIFT_CLAMP, min(DRIFT_CLAMP, slope_per_min * 60.0))
                drift *= DRIFT_SHRINK
        # volatility: sd of 10-min log diffs over the whole series
        diffs = [pts[i + 1][1] - pts[i][1] for i in range(len(pts) - 1)]
        if len(diffs) >= 3:
            md = sum(diffs) / len(diffs)
            var = sum((d - md) ** 2 for d in diffs) / (len(diffs) - 1)
            vol = math.sqrt(max(var, 0.0))
        else:
            vol = VOL_DEFAULT
        vol = max(VOL_FLOOR, min(VOL_CAP, vol))
        lam = DEATH_LAMBDA.get(c["venue"], 0.20)
        if c["mayhem"]:
            lam = max(lam, 0.15)
        params[mint] = (x0, drift, vol, lam)

    # ---- Monte-Carlo per read --------------------------------------------
    rng = random.Random(SEED)
    n_bins = len(edges) + 1  # 82
    place_counts = {k: {} for k in range(1, places + 1)}  # place -> mint -> count
    size_counts = {k: [0] * n_bins for k in range(1, places + 1)}
    total_samples = 0
    for h in horizons_h:
        # per-coin (mean, sd, survival) at this horizon
        dist = []
        for mint in universe:
            x0, drift, vol, lam = params[mint]
            mean = x0 + drift * DRIFT_TAU_H * (1.0 - math.exp(-h / DRIFT_TAU_H))
            sd = math.sqrt((vol * math.sqrt(6.0)) ** 2 * h + BASE_SD ** 2)
            sd = min(sd, SD_CAP)
            surv = math.exp(-lam * h)
            dist.append((mint, mean, sd, surv))
        for _ in range(MC_SAMPLES):
            # common market shift: all coins move together (widens the
            # size distribution; ranking is unchanged, so order is unaffected)
            shift = rng.gauss(MKT_DRIFT * h, MKT_SD * math.sqrt(h))
            draws = []
            for mint, mean, sd, surv in dist:
                if rng.random() < surv:
                    draws.append((rng.gauss(mean, sd) + shift, mint))
                else:
                    draws.append((rng.gauss(DEATH_MEAN, DEATH_SD) + shift, mint))
            draws.sort(reverse=True)
            total_samples += 1
            for k in range(1, places + 1):
                if k - 1 < len(draws):
                    val, mint = draws[k - 1]
                    pc = place_counts[k]
                    pc[mint] = pc.get(mint, 0) + 1
                    b = bin_of(val, edges)
                    size_counts[k][b] += 1

    # ---- average over reads, reserve newcomer mass -----------------------
    # newcomer reserve per place, averaged over the reads' horizons
    reserve = {}
    for k in range(1, places + 1):
        r = sum(NEWCOMER_R_MIN[k] + (NEWCOMER_R_MAX[k] - NEWCOMER_R_MIN[k]) * (1.0 - math.exp(-h / NEWCOMER_TAU_H))
                for h in horizons_h) / max(1, len(horizons_h))
        reserve[k] = r

    probabilities = []
    for k in range(1, places + 1):
        pc = place_counts[k]
        if not pc:
            continue
        scale = (1.0 - reserve[k]) / total_samples
        ranked = sorted(pc.items(), key=lambda kv: (-kv[1], kv[0]))
        named = 0
        for mint, cnt in ranked:
            p = cnt * scale
            if p < MIN_P or named >= MAX_ROWS_PER_PLACE:
                break
            probabilities.append({"mint": mint, "place": k, "p": floor6(p)})
            named += 1

    # ---- sizes: MC histogram averaged over reads, widened ----------------
    sizes = []
    for k in range(1, places + 1):
        counts = size_counts[k]
        if total_samples == 0 or sum(counts) == 0:
            continue
        # widen: mix the histogram with a broad normal around its median
        hist = [c / total_samples for c in counts]
        med = median_of_hist(hist, edges)
        broad = []
        for b in range(n_bins):
            lo = float("-inf") if b == 0 else edges[b - 1]
            hi = float("inf") if b == n_bins - 1 else edges[b]
            broad.append(above(lo, med, 0.8) - above(hi, med, 0.8))
        tot_b = sum(broad)
        if tot_b > 0:
            broad = [x / tot_b for x in broad]
        mix = 0.75  # weight on the MC histogram; rest on the broad mix
        row = [(mix * hist[b] + (1.0 - mix) * broad[b]) for b in range(n_bins)]
        # leave SIZE_UNNAMED unnamed: scale so the named sum is 1 - SIZE_UNNAMED
        s = sum(row)
        if s > 0:
            row = [x * (1.0 - SIZE_UNNAMED) / s for x in row]
        for b in range(n_bins):
            p = floor6(row[b])
            if p > 0:
                sizes.append({"place": k, "bin": b, "p": p})

    # ---- top5: most likely coin per place (distinct) ---------------------
    top5 = []
    used = set()
    for k in range(1, places + 1):
        pc = place_counts[k]
        best = None
        for mint, cnt in sorted(pc.items(), key=lambda kv: (-kv[1], kv[0])):
            if mint not in used:
                best = mint
                break
        if best is None:  # fall back: highest last log-cap not yet used
            for _, mint in scored:
                if mint not in used:
                    best = mint
                    break
        if best is not None:
            top5.append(best)
            used.add(best)
    while len(top5) < places:  # extremely defensive; should not happen
        for _, mint in scored:
            if mint not in used:
                top5.append(mint)
                used.add(mint)
                break
        else:
            break

    forecast = {"probabilities": probabilities, "top5": top5[:places], "sizes": sizes}
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, out_name), "w") as f:
        f.write(json.dumps(forecast, separators=(",", ":")) + "\n")


def bin_of(val, edges):
    # bin 0: < edges[0]; bin i (1..len-1): [edges[i-1], edges[i]); last: >= edges[-1]
    for i, e in enumerate(edges):
        if val < e:
            return i
    return len(edges)


def median_of_hist(hist, edges):
    cum = 0.0
    for b, p in enumerate(hist):
        cum += p
        if cum >= 0.5:
            if b == 0:
                return edges[0] - 0.1
            if b == len(edges):
                return edges[-1] + 0.1
            return (edges[b - 1] + edges[b]) / 2.0
    return (edges[len(edges) // 2 - 1] + edges[len(edges) // 2]) / 2.0


def floor6(x):
    return math.floor(x * MICRO) / MICRO


if __name__ == "__main__":
    main()
