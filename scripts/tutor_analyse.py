"""Multi-axis analysis of the tutorspike mastery surface, across a seed pair.

The point of the seed pair is the control: any ordering that does not survive a change of seed is not
a finding. This project has been here before (the GQA A/B was unresolvable because seed noise ran 1.73x
the signal), so every table below is printed per seed and then judged on agreement, never averaged into
one number that hides the disagreement.
"""
import json, sys, csv, math, statistics as st
from collections import defaultdict

RUNS = sys.argv[1:] or ["models/tutor_s1", "models/tutor_s2"]


def load(run):
    snap = json.load(open(f"{run}/tutor_surface.json"))
    events = []
    try:
        with open(f"{run}/tutor_events.csv") as f:
            for r in csv.DictReader(f):
                events.append(r)
    except FileNotFoundError:
        pass
    return snap, events


def per_pop(snap):
    out = {}
    for p, name in enumerate(snap["populations"]):
        idx = [i for i, x in enumerate(snap["doc_pop"]) if x == p]
        seen = [i for i in idx if snap["visits"][i] > 0]
        vel = [snap["velocity"][i] for i in idx if snap["velocity"][i] != 0]
        tr = [snap["transfer"][i] for i in idx if snap["transfer"][i] != 0]
        out[name] = dict(
            docs=len(idx), seen=len(seen),
            cover=len(seen) / len(idx),
            visits=st.mean(snap["visits"][i] for i in idx),
            nelbo=st.mean(snap["nelbo"][i] for i in seen) if seen else float("nan"),
            vel=st.mean(vel) if vel else float("nan"), n_vel=len(vel),
            tr=st.mean(tr) if tr else float("nan"), n_tr=len(tr),
        )
    return out


BINS = [(2, 5), (5, 10), (10, 20), (20, 50), (50, 100), (100, 1e9)]


def matched(snap):
    """Velocity binned on the entry's OWN applied learning -- the first normalising axis."""
    res = defaultdict(dict)
    for p, name in enumerate(snap["populations"]):
        for lo, hi in BINS:
            v = [snap["velocity"][i] for i in range(len(snap["velocity"]))
                 if snap["doc_pop"][i] == p and snap["velocity"][i] != 0
                 and lo <= snap["applied"][i] < hi]
            if len(v) > 3:
                res[(lo, hi)][name] = (st.mean(v), len(v))
    return res


def surface2d(events, pops):
    """Velocity on BOTH axes: own applied learning x global progress.

    The ledger cannot express this -- it keeps only an entry's current velocity -- which is exactly why
    the events exist. Binning on own-applied alone still confounds "how far this entry has come" with
    "how mature the model was when the reading was taken".
    """
    gmax = max((float(e["global_applied"]) for e in events if e["kind"] == "0"), default=0.0)
    if gmax <= 0:
        return {}
    halves = [(0.0, gmax / 3, "early"), (gmax / 3, 2 * gmax / 3, "mid"), (2 * gmax / 3, gmax * 2, "late")]
    res = defaultdict(dict)
    for e in events:
        if e["kind"] != "0":
            continue
        g, own, val = float(e["global_applied"]), float(e["own_applied"]), float(e["value"])
        for lo, hi in BINS:
            if lo <= own < hi:
                for glo, ghi, gname in halves:
                    if glo <= g < ghi:
                        res[(gname, (lo, hi))].setdefault(pops[int(e["pop"])], []).append(val)
                break
    return res


loaded = [(r, *load(r)) for r in RUNS]

print("=" * 100)
print("PER-POPULATION, PER SEED")
print("=" * 100)
for run, snap, ev in loaded:
    print(f"\n{run}  step {snap['step']}  coverage {snap['coverage']:.4f}  "
          f"drift floor {snap['drift_floor']:.2e}  events {len(ev)}")
    print(f"  {'pop':<12}{'docs':>7}{'cover':>8}{'visits':>8}{'nelbo':>9}"
          f"{'velocity':>11}{'n_vel':>8}{'transfer':>12}{'n_tr':>7}")
    for name, d in per_pop(snap).items():
        print(f"  {name:<12}{d['docs']:>7}{d['cover']:>8.3f}{d['visits']:>8.1f}{d['nelbo']:>9.3f}"
              f"{d['vel']:>11.4f}{d['n_vel']:>8}{d['tr']:>12.3e}{d['n_tr']:>7}")

print("\n" + "=" * 100)
print("VELOCITY AT MATCHED OWN-APPLIED LEARNING  (seed1 | seed2)")
print("=" * 100)
pops = loaded[0][1]["populations"]
mm = [matched(s) for _, s, _ in loaded]
print(f"{'applied':<10}" + "".join(f"{n:>30}" for n in pops))
for b in BINS:
    row = f"{f'{b[0]}-{b[1] if b[1] < 1e9 else 0}':<10}"
    for n in pops:
        cells = []
        for m in mm:
            c = m.get(b, {}).get(n)
            cells.append(f"{c[0]:.4f}" if c else "-")
        row += f"{' | '.join(cells):>30}"
    print(row)

print("\n" + "=" * 100)
print("SEED AGREEMENT -- does the ordering survive a change of seed?")
print("=" * 100)
for b in BINS:
    ranks = []
    for m in mm:
        cell = {n: m.get(b, {}).get(n) for n in pops}
        avail = {n: v[0] for n, v in cell.items() if v}
        if len(avail) < 2:
            ranks.append(None)
        else:
            ranks.append(tuple(sorted(avail, key=avail.get)))
    if all(r is not None for r in ranks):
        agree = "AGREE" if len(set(ranks)) == 1 else "DISAGREE"
        print(f"  applied {b[0]}-{b[1] if b[1] < 1e9 else 0:<6}  {agree:<9} "
              f"s1={'<'.join(ranks[0])}   s2={'<'.join(ranks[1])}")

print("\n" + "=" * 100)
print("VELOCITY ON BOTH AXES (own applied x global progress) -- seed 1")
print("=" * 100)
s2d = surface2d(loaded[0][2], pops)
print(f"{'phase':<8}{'applied':<10}" + "".join(f"{n:>22}" for n in pops))
for gname in ("early", "mid", "late"):
    for b in BINS:
        cell = s2d.get((gname, b))
        if not cell:
            continue
        row = f"{gname:<8}{f'{b[0]}-{b[1] if b[1] < 1e9 else 0}':<10}"
        for n in pops:
            v = cell.get(n, [])
            row += f"{(f'{st.mean(v):.4f} (n={len(v)})' if len(v) > 3 else '-'):>22}"
        print(row)
