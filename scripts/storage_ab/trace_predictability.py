#!/usr/bin/env python3
"""Offline questions about a routed-expert access trace: how wide is the working set, what does a cache
budget buy, and could the misses have been predicted from recent tokens?

Reads the S0RT trace and S0RX extents that `sub0llm-qwen4-gen --expert-trace PREFIX` writes (formats:
Sub0TieredCache `docs/trace-replay.md`). Pure Python plus numpy, a minute or two for 2,000 tokens; it needs
no model and no sidecar.

The cache model is LRU over exact-size rows, which is what `SizeClassedTable` approximates. It reports
hit rates, not waits: use `replay_ab.py` for what a miss costs.

Usage:  python scripts/storage_ab/trace_predictability.py PREFIX [--layers 48] [--budgets 10,17,26]
Results and their reading: docs/optimization/opportunities/O13_storage_convergence.md.
"""
from __future__ import annotations

import argparse
import collections
import pathlib
import struct

import numpy as np


def load(prefix: pathlib.Path) -> tuple[list[np.ndarray], np.ndarray]:
    """Return (batches of row ids, row byte sizes)."""
    trace = prefix.with_suffix(".trace").read_bytes()
    extents = prefix.with_suffix(".extents").read_bytes()
    if trace[:4] != b"S0RT" or extents[:4] != b"S0RX":
        raise ValueError(f"{prefix}: not an S0RT/S0RX pair")
    n_batches = struct.unpack_from("<Q", trace, 8)[0]
    n_rows = struct.unpack_from("<Q", extents, 8)[0]
    size = np.frombuffer(extents, dtype="<u8", offset=16).reshape(n_rows, 2)[:, 1].astype(np.int64)
    batches, off = [], 16
    for _ in range(n_batches):
        n = struct.unpack_from("<I", trace, off)[0]
        off += 4
        batches.append(np.frombuffer(trace, dtype="<u8", count=n, offset=off).astype(np.int64))
        off += 8 * n
    return batches, size


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prefix", type=pathlib.Path, help="trace path without its .trace/.extents suffix")
    ap.add_argument("--layers", type=int, default=48, help="batches per token (one per MoE layer)")
    ap.add_argument("--budgets", default="10,14,17,20,23,26,30", help="cache budgets to replay, GiB")
    ap.add_argument("--history", type=int, default=8, help="longest recency predictor, in tokens")
    args = ap.parse_args()

    batches, size = load(args.prefix)
    layers, n_batches = args.layers, len(batches)
    tokens = n_batches // layers
    gib = float(2**30)
    sel = [[set(batches[t * layers + l].tolist()) for l in range(layers)] for t in range(tokens)]
    count = np.bincount(np.concatenate(batches), minlength=len(size))
    print(f"{n_batches} batches = {tokens} tokens x {layers} layers; {len(size)} rows of "
          f"{size.min() / 2**20:.2f}-{size.max() / 2**20:.2f} MiB, {size.sum() / gib:.2f} GiB in all")
    print(f"touched: {(count > 0).sum()} rows, {size[count > 0].sum() / gib:.2f} GiB")

    print("\nrecall of a layer's whole selection, predicted before its router runs:")
    first = args.history
    spans = sorted({1, 2, 4, args.history})
    for span in spans:
        hit = need = predicted = 0
        for t in range(first, tokens):
            for l in range(layers):
                guess = set().union(*(sel[t - j][l] for j in range(1, span + 1)))
                hit += len(guess & sel[t][l])
                need += len(sel[t][l])
                predicted += len(guess)
        n = (tokens - first) * layers
        print(f"  same layer, previous {span} token(s): {hit / need * 100:5.1f}%  ({predicted / n:.1f} rows guessed)")

    print("\nLRU over exact-size rows; steady state is the second half of the trace:")
    print("  budget   hit rate   misses/token   layers with a miss   misses a recency predictor names")
    for g in (float(x) for x in args.budgets.split(",")):
        budget, used = int(g * gib), 0
        lru: collections.OrderedDict[int, int] = collections.OrderedDict()
        hits = total = misses = miss_layers = scored_layers = 0
        named = dict.fromkeys(spans, 0)
        for bi, batch in enumerate(batches):
            t, l = divmod(bi, layers)
            steady = bi >= n_batches // 2 and t >= first
            missed_here = 0
            for r in batch.tolist():
                if r in lru:
                    lru.move_to_end(r)
                    hits += steady
                else:
                    s = int(size[r])
                    while used + s > budget:
                        used -= lru.popitem(last=False)[1]
                    lru[r] = s
                    used += s
                    if steady:
                        misses += 1
                        missed_here += 1
                        for span in spans:
                            named[span] += any(r in sel[t - j][l] for j in range(1, span + 1))
                total += steady
            if steady:
                scored_layers += 1
                miss_layers += missed_here > 0
        steady_tokens = scored_layers / layers
        named_text = " / ".join(f"{named[s] / max(misses, 1) * 100:.1f}%" for s in spans)
        print(f"  {g:4.0f} GiB   {hits / total * 100:5.1f}%   {misses / steady_tokens:10.1f}   "
              f"{miss_layers / scored_layers * 100:14.0f}%   {named_text}  (previous {'/'.join(map(str, spans))})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
