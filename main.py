"""Deterministic offline top-5 forecast for pumpfun-top5-forecast-01 (Python 3.13, stdlib only).

Reads $MAKEFUN_INPUT_DIR (default /input) and writes forecast.json to $MAKEFUN_OUTPUT_DIR (default /output).
Method: seeded Monte Carlo over the coins in the last mark. Each coin's log10 market cap is moved by a
mean-reverting drift (graduated coins hold up better) plus a shared market factor and idiosyncratic noise that
grows with the forecast horizon. Place probabilities are simulated frequencies, shrunk to leave room for coins
born after the cut. Size bins come from the simulated cap at each place, smoothed.
"""
import json
import math
import os
import random
from datetime import datetime

IN = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUT = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
SIMS = 3000
TOP_N = 60
NEWBORN_SHARE = 0.35  # share of each place left unnamed for newcomers and movers


def ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def size_of(lamports):
    return math.log10(int(lamports)) - 9


def main():
    with open(os.path.join(IN, "task.json")) as f:
        task = json.load(f)
    marks = []
    with open(os.path.join(IN, "history.ndjson")) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = json.loads(line)
            if m.get("ok") and isinstance(m.get("rows"), list):
                marks.append(m)
    if not marks:
        raise SystemExit("no ok mark")
    last = marks[-1]
    rows = sorted(last["rows"], key=lambda r: r["rank"])
    places = task["places"]
    edges = task["size"]["edges"]
    nb = len(edges) + 1
    until = ts(task["history_until"])
    reads = [ts(r) for r in task["reads"]]
    horizons = [max(0.0, (r - until) / 3600.0) for r in reads]
    hz = sum(horizons) / len(horizons)

    # momentum: log10 change over the last few marks, used as a weak signal
    prev = {}
    if len(marks) >= 4:
        for r in marks[-4]["rows"]:
            prev[r["mint"]] = size_of(r["market_cap_lamports"])

    cands = rows[:TOP_N]
    base = []
    for r in cands:
        s = size_of(r["market_cap_lamports"])
        mom = s - prev[r["mint"]] if r["mint"] in prev else 0.0
        graduated = r.get("venue") == "pool" or r.get("complete")
        drift = -0.10 * hz if not graduated else -0.03 * hz
        drift += 0.15 * max(-1.0, min(1.0, mom))
        base.append((r["mint"], s, drift))

    rng = random.Random(20261112)
    sd = 0.30 + 0.15 * math.sqrt(hz)
    counts = [dict() for _ in range(places)]
    cap_samples = [[] for _ in range(places)]
    for _ in range(SIMS):
        common = rng.gauss(0, 0.15)
        sim = []
        for mint, s, d in base:
            sim.append((s + d + common + rng.gauss(0, sd), mint))
        sim.sort(key=lambda x: (-x[0], x[1]))
        for k in range(places):
            v, mint = sim[k]
            counts[k][mint] = counts[k].get(mint, 0) + 1
            cap_samples[k].append(v)

    probabilities = []
    for k in range(places):
        items = sorted(counts[k].items(), key=lambda x: (-x[1], x[0]))[:100]
        for mint, c in items:
            p = c / SIMS * (1 - NEWBORN_SHARE)
            p = math.floor(p * 1e6) / 1e6
            if p > 0:
                probabilities.append({"mint": mint, "place": k + 1, "p": p})

    sizes = []
    for k in range(places):
        xs = cap_samples[k]
        mean = sum(xs) / len(xs)
        var = sum((x - mean) ** 2 for x in xs) / len(xs)
        s2 = math.sqrt(var + 0.15 ** 2)
        mean += 0.10  # newborn coins can take a place too, nudge mass upward

        def cdf(x, mean=mean, s2=s2):
            if x == -math.inf:
                return 0.0
            if x == math.inf:
                return 1.0
            return 0.5 * math.erfc(-(x - mean) / (s2 * math.sqrt(2)))

        for b in range(nb):
            lo = -math.inf if b == 0 else edges[b - 1]
            hi = math.inf if b == nb - 1 else edges[b]
            p = math.floor((cdf(hi) - cdf(lo)) * 0.95 * 1e6) / 1e6
            if p > 0:
                sizes.append({"place": k + 1, "bin": b, "p": p})

    top5, used = [], set()
    for k in range(places):
        for mint, _c in sorted(counts[k].items(), key=lambda x: (-x[1], x[0])):
            if mint not in used:
                top5.append(mint)
                used.add(mint)
                break
    for r in rows:
        if len(top5) >= places:
            break
        if r["mint"] not in used:
            top5.append(r["mint"])
            used.add(r["mint"])

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, task["output"]), "w") as f:
        json.dump({"probabilities": probabilities, "top5": top5, "sizes": sizes}, f, separators=(",", ":"))
        f.write("\n")


if __name__ == "__main__":
    main()
