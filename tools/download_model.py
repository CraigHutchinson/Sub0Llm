#!/usr/bin/env python3
"""Download a small GGUF model for the sub0llm episodic memory PoC.

Fetches from HuggingFace Hub using huggingface_hub (pip install huggingface_hub).
Run this from any machine with HuggingFace access; the container's network
policy blocks huggingface.co.

Usage:
  python3 tools/download_model.py [--model MODEL] [--quant QUANT] [--out DIR]

Defaults:
  --model  Qwen/Qwen2-0.5B-Instruct-GGUF      (494 MB, q4_k_m)
  --quant  qwen2-0_5b-instruct-q4_k_m.gguf
  --out    /tmp/models/

Supported presets (--preset flag):
  qwen2-0.5b    Qwen/Qwen2-0.5B-Instruct-GGUF  q4_k_m  ~494 MB
  qwen2-1.5b    Qwen/Qwen2-1.5B-Instruct-GGUF  q4_k_m  ~986 MB
  qwen3-0.6b    Qwen/Qwen3-0.6B-GGUF           q4_k_m  ~433 MB  (Qwen3, tied emb)
  qwen3-1.7b    Qwen/Qwen3-1.7B-GGUF           q4_k_m  ~1.1 GB
  qwen3-4b      Qwen/Qwen3-4B-GGUF             q4_k_m  ~2.6 GB  (needs head_dim fix)

After download, run:
  ./build/bin/sub0llm-episodic info --model /tmp/models/<filename>

  # Use a NOVEL fact the model cannot already know.
  # Bad: "capital of France" — already in training data, NLL drop is meaningless.
  # Good: project-specific names, invented terms, clearly fictional specifics.

  ./build/bin/sub0llm-episodic probe --model /tmp/models/<filename> \\
      --fact "sub0llm is a C++23 educational LLM framework by CraigHutchinson" \\
      --query "what is sub0llm used for"
"""

import argparse
import os
import sys

PRESETS = {
    "qwen2-0.5b": ("Qwen/Qwen2-0.5B-Instruct-GGUF",  "qwen2-0_5b-instruct-q4_k_m.gguf"),
    "qwen2-1.5b": ("Qwen/Qwen2-1.5B-Instruct-GGUF",  "qwen2-1_5b-instruct-q4_k_m.gguf"),
    "qwen3-0.6b": ("Qwen/Qwen3-0.6B-GGUF",            "Qwen3-0.6B-Q8_0.gguf"),
    "qwen3-1.7b": ("Qwen/Qwen3-1.7B-GGUF",            "qwen3-1.7b-q4_k_m.gguf"),
    "qwen3-4b":   ("Qwen/Qwen3-4B-GGUF",              "qwen3-4b-q4_k_m.gguf"),
}

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preset", choices=list(PRESETS), default="qwen3-0.6b",
                   help="Model preset (default: qwen2-0.5b)")
    p.add_argument("--model", help="Override HuggingFace repo ID")
    p.add_argument("--quant", help="Override filename within the repo")
    p.add_argument("--out",   default="models",
                   help="Output directory (default: models)")
    p.add_argument("--token", default=os.environ.get("HF_TOKEN"),
                   help="HuggingFace token (or set HF_TOKEN env var)")
    return p.parse_args()


def main():
    args = parse_args()

    repo_id, filename = PRESETS[args.preset]
    if args.model:
        repo_id = args.model
    if args.quant:
        filename = args.quant

    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)

    if os.path.exists(out_path):
        size_mb = os.path.getsize(out_path) / 1024 / 1024
        print(f"Already exists ({size_mb:.0f} MB): {out_path}")
        return out_path

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("Error: huggingface_hub not installed.", file=sys.stderr)
        print("  pip install huggingface_hub", file=sys.stderr)
        sys.exit(1)

    print(f"Downloading {repo_id}/{filename} → {out_path}")
    print("(This may take a few minutes depending on your connection speed)")

    local = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=out_dir,
        token=args.token,
    )
    size_mb = os.path.getsize(local) / 1024 / 1024
    print(f"Done: {local} ({size_mb:.0f} MB)")
    return local


if __name__ == "__main__":
    main()
