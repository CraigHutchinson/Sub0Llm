#!/usr/bin/env python
"""Download FineWeb-Edu (sample/10BT) and write a plain-text corpus for sub0-configure.

FineWeb-Edu ships as parquet on the HuggingFace Hub; this fetches the shards one at a
time, streams the `text` column out to a UTF-8 file, and deletes each parquet after
extraction -- so peak extra disk is one shard (~2-3GB) plus the growing text file, not
the whole 27GB of parquet. It is resumable: shards already extracted (recorded in
<out>.done) are skipped.

Documents are separated by a literal `<|endoftext|>` marker on its own line -- the same
convention the canonical TinyStories dataset uses (see get_tinystories.py) -- so the
tokenizer can learn an explicit stop signal instead of the model never seeing a
document actually end. A bare blank line is NOT a reliable boundary signal: it also
appears mid-document (paragraph breaks), so the sampler's old "\\n\\n" heuristic could
mistake a paragraph break for a document break, or vice versa.

Requirements (host Python, not committed deps):  pip install huggingface_hub duckdb

Usage:
    python scripts/get_fineweb.py [--out data/fineweb_edu.txt] [--shards 14] [--subset sample/10BT]

The full sample/10BT is ~46GB of text (~12M unique words); pass --shards N for a smaller
slice. Point the configurator at it with:  sub0llm-configure --corpus <abs path to out>
(--corpus-pretok defaults to AUTO, which will choose on-demand tokenization for a corpus this large).
"""
import argparse, os, time

REPO = "HuggingFaceFW/fineweb-edu"
EOS = "<|endoftext|>"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join("data", "fineweb_edu.txt"))
    ap.add_argument("--subset", default="sample/10BT")
    ap.add_argument("--shards", type=int, default=14)
    ap.add_argument("--cache", default=os.path.join(os.path.dirname(__file__) or ".", ".fw_cache"))
    args = ap.parse_args()

    from huggingface_hub import hf_hub_download
    import duckdb

    os.makedirs(args.cache, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    done_path = args.out + ".done"
    done = set(open(done_path).read().split()) if os.path.exists(done_path) else set()
    con = duckdb.connect()

    def log(*a):
        print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)

    for i in range(args.shards):
        shard = f"{args.subset}/{i:03d}_00000.parquet"
        if shard in done:
            log("skip (done)", shard)
            continue
        log("downloading", shard)
        path = hf_hub_download(REPO, shard, repo_type="dataset", local_dir=args.cache)
        log("extracting text from", os.path.basename(path))
        con.execute("SELECT text FROM read_parquet(?)", [path])
        ndoc = 0
        with open(args.out, "ab") as out:
            while True:
                rows = con.fetchmany(5000)
                if not rows:
                    break
                chunk = f"\n{EOS}\n".join(t for (t,) in rows if t)
                if chunk:
                    out.write(chunk.encode("utf-8"))
                    out.write(f"\n{EOS}\n".encode("utf-8"))   # also terminate the shard's last document
                ndoc += len(rows)
        try:
            os.remove(path)
        except OSError:
            pass
        with open(done_path, "a") as f:
            f.write(shard + "\n")
        log(f"done {shard}: {ndoc} docs | corpus now {os.path.getsize(args.out) / 1e9:.1f} GB")

    log("ALL SHARDS COMPLETE ->", args.out)


if __name__ == "__main__":
    main()
