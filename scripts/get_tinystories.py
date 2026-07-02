#!/usr/bin/env python
"""Download the canonical TinyStories dataset and write a plain-text corpus for sub0-configure.

Unlike FineWeb-Edu, the raw .txt release on the HuggingFace Hub already uses the RIGHT
document-separator convention: a literal `<|endoftext|>` marker on its own line between
stories (21,989 of them in the ~19MB validation split alone) -- the standard GPT-2/3
end-of-text signal, and exactly what lets the tokenizer teach the model an explicit stop
token instead of never seeing a document actually end. This script just fetches that file
as-is; it does not need to invent or reconstruct the marker.

(This project's earlier data/tinystories.txt was NOT produced by this script -- it predates
it, and its `<|endoftext|>` markers were lost at some point, replaced with a bare blank
line that the sampler could not reliably tell apart from an ordinary paragraph break.)

Requirements (host Python, not committed deps):  pip install huggingface_hub

Usage:
    python scripts/get_tinystories.py [--out data/tinystories.txt] [--variant v2] [--split train]

--variant v2 (default) is TinyStoriesV2-GPT4: GPT-4-only generations, the dataset's own
README recommends it over the original mixed GPT-3.5/GPT-4 release ("of lesser quality").
Pass --variant v1 for the original TinyStories-*.txt used in the paper's own models.
"""
import argparse, os, time

REPO = "roneneldan/TinyStories"
FILENAMES = {
    ("v1", "train"): "TinyStories-train.txt",
    ("v1", "valid"): "TinyStories-valid.txt",
    ("v2", "train"): "TinyStoriesV2-GPT4-train.txt",
    ("v2", "valid"): "TinyStoriesV2-GPT4-valid.txt",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join("data", "tinystories.txt"))
    ap.add_argument("--variant", choices=["v1", "v2"], default="v2")
    ap.add_argument("--split", choices=["train", "valid"], default="train")
    args = ap.parse_args()

    from huggingface_hub import hf_hub_download

    def log(*a):
        print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)

    filename = FILENAMES[(args.variant, args.split)]
    log(f"downloading {filename} ({args.variant}/{args.split})")
    path = hf_hub_download(REPO, filename, repo_type="dataset")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    log(f"copying to {args.out}")
    with open(path, "rb") as src, open(args.out, "wb") as dst:
        while chunk := src.read(1 << 20):
            dst.write(chunk)

    n_docs = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip() == "<|endoftext|>":
                n_docs += 1
    log(f"done -> {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB, ~{n_docs} stories, "
        f"<|endoftext|>-separated)")


if __name__ == "__main__":
    main()
