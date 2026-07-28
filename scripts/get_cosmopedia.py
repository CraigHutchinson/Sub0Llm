#!/usr/bin/env python
"""Download Cosmopedia-v2 and write a plain-text corpus for sub0-configure.

WHY THIS CORPUS. TinyStories is synthetically constrained to a ~1.5k-word vocabulary of concrete
nouns and simple actions. That makes it useless for testing any mechanism whose job is to refine
ABSTRACTIONS -- weight-shared looping (LoopSplit), depth attention, mHC. A negative result there says
"the corpus had no abstraction to contract", not "the mechanism does not work"; see the scope caveat
on the LoopSplit A/B.

Cosmopedia is the direct answer: same synthetic-and-coherent character as TinyStories (it is from the
same HuggingFace team), but generated as TEXTBOOKS, WikiHow articles, blogposts and educational
stories -- so it is dense in DEFINED CONCEPTS and abstract terms while staying clean and on-topic.
If iterative-refinement mechanisms help at small scale anywhere, expository text explaining concepts
is where.

Source: HuggingFaceTB/smollm-corpus, `cosmopedia-v2` config (the curated v2 used to train SmolLM,
recommended over v1). 104 parquet shards; this pulls shards in order until --target-gb of TEXT is
written, so the size is controllable rather than all-or-nothing (the full set is ~25B tokens).

Documents are separated by `<|endoftext|>` on its own line -- this project's required convention. The
tokenizer's document scan is EOT-only (the blank-line fallback was removed as unsound), so a corpus
without these markers is REJECTED at configure time rather than silently training across document
boundaries.

Requirements (host Python, not committed deps):  pip install huggingface_hub pyarrow

Usage:
    python scripts/get_cosmopedia.py [--out data/cosmopedia.txt] [--target-gb 10]
"""
import argparse, os, time

REPO = "HuggingFaceTB/smollm-corpus"
CONFIG = "cosmopedia-v2"
N_SHARDS = 104
TEXT_COLUMN = "text"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join("data", "cosmopedia.txt"))
    ap.add_argument("--target-gb", type=float, default=10.0,
                    help="stop once this many GB of text has been written (0 = all shards)")
    args = ap.parse_args()

    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    def log(*a):
        print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)

    target_bytes = int(args.target_gb * 1e9) if args.target_gb > 0 else None
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)

    written = 0
    docs = 0
    marker = b"\n<|endoftext|>\n"
    with open(args.out, "wb") as dst:
        for i in range(N_SHARDS):
            name = f"{CONFIG}/train-{i:05d}-of-{N_SHARDS:05d}.parquet"
            log(f"shard {i + 1}/{N_SHARDS}: {name}")
            path = hf_hub_download(REPO, name, repo_type="dataset")
            table = pq.read_table(path, columns=[TEXT_COLUMN])
            for chunk in table.column(TEXT_COLUMN).chunks:
                for value in chunk:
                    text = value.as_py()
                    if not text:
                        continue
                    dst.write(text.encode("utf-8"))
                    dst.write(marker)
                    written += len(text) + len(marker)
                    docs += 1
            log(f"  -> {written / 1e9:.2f} GB, {docs} documents")
            if target_bytes and written >= target_bytes:
                log(f"reached --target-gb {args.target_gb}; stopping at shard {i + 1}")
                break

    log(f"done -> {args.out} ({os.path.getsize(args.out) / 1e9:.2f} GB, {docs} documents, "
        f"<|endoftext|>-separated)")


if __name__ == "__main__":
    main()
