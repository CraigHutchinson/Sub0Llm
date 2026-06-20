#!/usr/bin/env python3
"""M3 — correlation-decay probe (ch32 BUILD_PLAN Phase 0).

Tests the Lin & Tegmark (2017) criticality claim on OUR data: does the
mutual information I(X_i ; X_{i+d}) between two characters d apart decay as a
POWER LAW (critical / scale-free, long-range structure) or EXPONENTIALLY
(short memory)? The ch32 multi-resolution / MERA hierarchy is only justified if
real text is power-law AND a short-memory reference is not.

Method (char-level; small alphabet → reliable MI):
  * real I(d): empirical, with Miller-Madow finite-sample bias correction.
  * Markov-1 reference I(d): ANALYTIC from the estimated 1-step transition
    matrix (pi, P): I(d) = sum_ij pi_i P^d_ij log2( P^d_ij / pi_j ). Same local
    statistics as the corpus, but provably exponential decay — the control.
  * shuffle floor: MI of a shuffled copy ~ the residual finite-sample bias.
Fits log I(d) vs log d (power law, slope = -alpha) and vs d (exponential),
reports which wins by R^2.

Python lives in tools/ only (repo policy). Usage:
  python mi_decay.py CORPUS.txt [--max-d 100] [--limit-mb 8] [--label NAME]
"""
import sys, argparse, numpy as np


def load_ids(path, limit_bytes=None):
    with open(path, "rb") as f:
        data = f.read(limit_bytes) if limit_bytes else f.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    uniq, inv = np.unique(arr, return_inverse=True)
    return inv.astype(np.int64), len(uniq)


def entropy_bits(p):
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def mi_at(ids, K, d):
    """Empirical I(X_i; X_{i+d}) in bits, Miller-Madow corrected."""
    a, b = ids[:-d], ids[d:]
    N = a.size
    J = np.bincount(a * K + b, minlength=K * K).astype(np.float64).reshape(K, K)
    J /= N
    pa, pb = J.sum(1), J.sum(0)
    nz = J > 0
    mi = float((J[nz] * np.log2(J[nz] / (pa[:, None] * pb[None, :])[nz])).sum())
    # Miller-Madow: + (m_xy - m_x - m_y + 1) / (2 N ln2)
    mm = (np.count_nonzero(J) - np.count_nonzero(pa) - np.count_nonzero(pb) + 1)
    return mi + mm / (2 * N * np.log(2))


def markov1_mi(ids, K, dists):
    """Analytic I(d) for the 1-step Markov chain fit to the data (exponential ref)."""
    a, b = ids[:-1], ids[1:]
    C = np.bincount(a * K + b, minlength=K * K).astype(np.float64).reshape(K, K)
    pi = C.sum(1)
    pi /= pi.sum()
    rs = C.sum(1, keepdims=True)
    P = np.divide(C, rs, out=np.zeros_like(C), where=rs > 0)
    out = []
    Pd = np.linalg.matrix_power(P, 1)
    cache = {1: Pd}
    for d in dists:
        if d not in cache:
            cache[d] = np.linalg.matrix_power(P, d)
        Pd = cache[d]
        # I(d) = sum_i pi_i sum_j Pd_ij log2(Pd_ij / pi_j)
        mi = 0.0
        for i in range(K):
            row = Pd[i]
            nz = row > 0
            mi += pi[i] * float((row[nz] * np.log2(row[nz] / pi[nz])).sum())
        out.append(mi)
    return np.array(out)


def fit_quality(d, y):
    """Return (powerlaw_R2, exp_R2, alpha) using only positive y."""
    m = y > 0
    d, y = d[m], y[m]
    if len(d) < 4:
        return float("nan"), float("nan"), float("nan")
    ly = np.log(y)
    # power law: ly = c - alpha ln d
    A = np.polyfit(np.log(d), ly, 1)
    pl_pred = np.polyval(A, np.log(d))
    pl_r2 = 1 - ((ly - pl_pred) ** 2).sum() / ((ly - ly.mean()) ** 2).sum()
    # exponential: ly = c - beta d
    B = np.polyfit(d, ly, 1)
    ex_pred = np.polyval(B, d)
    ex_r2 = 1 - ((ly - ex_pred) ** 2).sum() / ((ly - ly.mean()) ** 2).sum()
    return pl_r2, ex_r2, -A[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus")
    ap.add_argument("--max-d", type=int, default=100)
    ap.add_argument("--limit-mb", type=float, default=8.0)
    ap.add_argument("--label", default=None)
    a = ap.parse_args()

    ids, K = load_ids(a.corpus, int(a.limit_mb * 1024 * 1024))
    label = a.label or a.corpus
    print(f"# {label}: {ids.size:,} chars, alphabet K={K}, H1={entropy_bits(np.bincount(ids).astype(float)/ids.size):.3f} bits")

    dists = np.unique(np.round(np.geomspace(1, a.max_d, 24)).astype(int))
    real = np.array([mi_at(ids, K, int(d)) for d in dists])
    rng = np.random.default_rng(0)
    sh = ids.copy(); rng.shuffle(sh)
    floor = mi_at(sh, K, 1)               # finite-sample MI bias floor
    mk = markov1_mi(ids, K, dists)

    print(f"# shuffle floor (bias): {floor:.4f} bits  (subtracted from real below)")
    print(f"#  {'d':>4}  {'I_real':>9}  {'I_markov1':>9}   real/markov")
    for d, r, m in zip(dists, real, mk):
        rc = max(r - floor, 1e-9)
        print(f"   {d:>4}  {rc:>9.4f}  {m:>9.4f}   {rc/max(m,1e-9):>6.1f}x")

    rc = np.maximum(real - floor, 1e-9)
    pr, er, alpha = fit_quality(dists.astype(float), rc)
    mpr, mer, _ = fit_quality(dists.astype(float), mk)
    print(f"\n# REAL    : power-law R2={pr:.3f} (alpha={alpha:.2f})  vs  exponential R2={er:.3f}  -> "
          f"{'POWER-LAW' if pr>er else 'exponential'}")
    print(f"# MARKOV-1: power-law R2={mpr:.3f}                vs  exponential R2={mer:.3f}  -> "
          f"{'power-law' if mpr>mer else 'EXPONENTIAL'}")
    tail = rc[dists >= a.max_d // 2].mean() / max(mk[dists >= a.max_d // 2].mean(), 1e-9)
    print(f"# VERDICT : real long-range MI is {tail:.0f}x the Markov-1 (short-memory) reference at d>={a.max_d//2} "
          f"-> {'CRITICAL/long-range structure REAL' if tail>3 else 'little beyond local structure'}")


if __name__ == "__main__":
    main()
