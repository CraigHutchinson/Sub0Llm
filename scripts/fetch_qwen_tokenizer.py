#!/usr/bin/env python3
"""scripts/fetch_qwen_tokenizer.py -- Download Qwen tokenizer files into data/qwen_tokenizer/

Fetches:
  - vocab.json
  - merges.txt
  - tokenizer_config.json
from official Hugging Face repository into data/qwen_tokenizer/.
"""

import os
import sys
import urllib.request
from pathlib import Path

BASE_URL = "https://huggingface.co/Qwen/Qwen3.8-Flash-Next/raw/main"
FILES = [
    "vocab.json",
    "merges.txt",
    "tokenizer_config.json"
]


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    target_dir = repo_root / "data" / "qwen_tokenizer"
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fetching Qwen tokenizer files into: {target_dir}")
    for fname in FILES:
        out_path = target_dir / fname
        if out_path.exists() and out_path.stat().st_size > 0:
            print(f"  ✓ {fname} already exists ({out_path.stat().st_size} bytes)")
            continue

        url = f"{BASE_URL}/{fname}"
        print(f"  Downloading {fname} from {url}...")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Sub0Llm-Tokenizer-Fetcher/1.0"})
            with urllib.request.urlopen(req, timeout=30) as resp, open(out_path, "wb") as f:
                f.write(resp.read())
            print(f"  ✓ {fname} saved ({out_path.stat().st_size} bytes)")
        except Exception as e:
            print(f"  ✗ Failed to download {fname}: {e}", file=sys.stderr)
            return 1

    print("\nAll Qwen tokenizer files are ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

