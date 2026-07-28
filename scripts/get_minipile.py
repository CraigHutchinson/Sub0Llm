#!/usr/bin/env python
"""Download MiniPile and write a plain-text corpus for sub0-configure.

WHY THIS CORPUS. MiniPile is a ~6GB subset of the deduplicated Pile, curated deliberately for
data-efficient research: every document was embedded, the embedding space k-means clustered, and
low-quality clusters dropped. The result is the most DIVERSITY per GB available -- code, prose,
academic writing, dialogue, reference text -- at a size that iterates fast.

It plays a different role from cosmopedia.txt, and both are worth having. Cosmopedia is dense but
narrow: synthetic expository text in a consistent voice. MiniPile is genuinely heterogeneous. An
architecture result that holds on BOTH is far more convincing than one that holds on either, because
they fail in different directions -- Cosmopedia could flatter a mechanism that likes clean regular
prose, MiniPile could flatter one that likes distribution shift.

Documents are separated by `<|endoftext|>` on its own line -- this project's required convention. The
tokenizer's document scan is EOT-only (the blank-line fallback was removed as unsound), so a corpus
without these markers is REJECTED at configure time rather than silently training across document
boundaries.

Requirements (host Python, not committed deps):  pip install huggingface_hub pyarrow

Usage:
    python scripts/get_minipile.py [--out data/minipile.txt] [--split train]
"""
import argparse, os, time

REPO = "JeanKaddour/minipile"
TEXT_COLUMN = "text"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join("data", "minipile.txt"))
    ap.add_argument("--split", choices=["train", "validation", "test"], default="train")
    args = ap.parse_args()

    from huggingface_hub import list_repo_files, hf_hub_download
    import pyarrow.parquet as pq

    def log(*a):
        print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)

    # Resolve the split's shards from the repo listing rather than hardcoding a filename pattern --
    # HuggingFace has reorganised parquet layouts before, and a stale hardcoded path fails obscurely.
    files = [f for f in list_repo_files(REPO, repo_type="dataset")
             if f.endswith(".parquet") and args.split in f]
    files.sort()
    if not files:
        raise SystemExit(f"no parquet shards for split '{args.split}' in {REPO}")
    log(f"{len(files)} shard(s) for split '{args.split}'")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    written = 0
    docs = 0
    marker = b"\n<|endoftext|>\n"
    with open(args.out, "wb") as dst:
        for i, name in enumerate(files):
            log(f"shard {i + 1}/{len(files)}: {name}")
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

    log(f"done -> {args.out} ({os.path.getsize(args.out) / 1e9:.2f} GB, {docs} documents, "
        f"<|endoftext|>-separated)")


if __name__ == "__main__":
    main()
