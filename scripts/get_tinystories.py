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

    # The VALIDATION splits begin mid-story; the TRAIN splits do not.
    #
    # Both valid files are slices of a larger file and the cut lands inside a story:
    # TinyStoriesV2-GPT4-valid.txt opens on "u don't have to be scared of the loud dog", and
    # TinyStories-valid.txt on " Spot. Spot saw the shiny car". Copying verbatim would hand the
    # tokenizer a first "document" that is a sentence fragment -- exactly the document-boundary
    # unsoundness the <|endoftext|>-only scan was introduced to eliminate, so importing one
    # deliberately would be perverse. Costs one story.
    #
    # This keys off the SPLIT, not off inspecting the text, because there is no reliable structural
    # signal: v1-valid's fragment begins " Spot." which reads exactly like a legitimate sentence
    # start. Whether the file is a slice is known a priori; guessing it from the bytes is not.
    # (Learned the hard way -- a content heuristic here silently ate the train split's real first
    # story, the "little boy named Ben" one.)
    drop_partial_head = args.split == "valid"
    marker = b"<|endoftext|>"
    with open(path, "rb") as src, open(args.out, "wb") as dst:
        head = src.read(1 << 20)
        if marker not in head:
            raise SystemExit(f"no {marker.decode()} in the first 1MB of {filename} -- wrong file?")
        if drop_partial_head:
            cut = head.find(marker)
            log(f"{args.split} split is a slice: dropping {cut} leading bytes (partial first story)")
            head = head[cut + len(marker):].lstrip(b"\r\n")
        dst.write(head)
        while chunk := src.read(1 << 20):
            dst.write(chunk)

    n_docs = 0
    with open(args.out, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip() == "<|endoftext|>":
                n_docs += 1
    log(f"done -> {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB, ~{n_docs} stories, "
        f"<|endoftext|>-separated)")


if __name__ == "__main__":
    main()
