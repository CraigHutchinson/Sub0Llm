#!/usr/bin/env python
"""Download GSM8K (grade-school math word problems) and write a plain-text corpus for sub0-configure.

GSM8K is the reasoning-density corpus in docs/ROADMAP.md §D, and the natural real-data testbed for the
deterministic-mechanisms op-delegation work (docs/DETERMINISTIC_MECHANISMS.md): its worked solutions carry
inline CALCULATOR ANNOTATIONS -- `<<48/2=24>>` -- a ready-made op-frame marking exactly the arithmetic spans
to delegate to the `math` node and MASK from the loss (so the model routes instead of learning fuzzy
arithmetic). This script fetches the data as-is and KEEPS those annotations, so the conversion (via
sub0/gsm8k.hpp's build_stream) can find them later.

Each example is written as one `<|endoftext|>`-separated document:

    <question>
    <answer, with <<expr=result>> annotations and the final `#### N`>
    <|endoftext|>

ELI5 -- what is `<<expr=result>>`? It's GSM8K's OWN format, written by the dataset's original authors
(OpenAI), not something this project adds. A solution reads like:

    There are 80/100 * 10 = <<80/100*10=8>>8 more purple flowers than yellow flowers.

`<<80/100*10=8>>` is a "calculator annotation": a parenthetical aside meaning "here's the arithmetic that
was calculated, and the answer". The `8` right after `>>` is unrelated to the annotation -- it's just the
sentence continuing normally, spelling out the number in prose, which is why it reads as if doubled
(`=8>>8`). GSM8K ships these so a grader/tool can find and verify every intermediate calculation. For us
they're a ready-made, already-labelled op-frame: sub0/gsm8k.hpp's build_stream re-verifies each one via the
exact `math` node, then converts it into our own delegated `[op math ...]` frame and masks the result from
the loss (docs/DETERMINISTIC_MECHANISMS.md).

Note: `<<` / `>>` here are just this dataset's chosen delimiter TEXT inside a plain-text corpus file -- they
are not C++ shift/stream/template-bracket operators, and nothing in this repo's C++ source is parsed this
way. The parser (gsm8k.hpp's `segment()`) finds them by literal substring search over corpus bytes, same as
finding any other punctuation in text.

Two ways it feeds training:
  * As a plain base corpus (reasoning-density fluency), blended with the synthetic op-delegation curriculum
    (a blend-schedule source with `"generator": "op_curriculum"`, `sub0llm-train --blend-config <path>`),
    which is what teaches the model to emit `[op math]`.
  * (Follow-on) a train-time pass that runs each solution's <<..>> annotations through build_stream to
    delegate + mask GSM8K's OWN arithmetic -- the piece that makes GSM8K itself fully delegating.

Requirements (host Python, not committed deps):  pip install datasets

Usage:
    python scripts/get_gsm8k.py [--out data/gsm8k.txt] [--split train] [--config main]

--split train (default, 7473 examples) or test (1319). --config main (default) is the human-written
solutions; socratic is the same problems with a socratic-style rationale.
"""
import argparse, os, time


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join("data", "gsm8k.txt"))
    ap.add_argument("--split", choices=["train", "test"], default="train")
    ap.add_argument("--config", choices=["main", "socratic"], default="main")
    args = ap.parse_args()

    from datasets import load_dataset

    def log(*a):
        print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)

    repo = "openai/gsm8k"   # the bare "gsm8k" id is deprecated; hub now requires namespace/name
    log(f"loading {repo}/{args.config} [{args.split}]")
    ds = load_dataset(repo, args.config, split=args.split)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    log(f"writing {len(ds)} problems -> {args.out} (annotations kept, <|endoftext|>-separated)")
    n = 0
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        for ex in ds:
            q = ex["question"].strip()
            a = ex["answer"].strip()
            f.write(f"{q}\n{a}\n<|endoftext|>\n")
            n += 1

    size_mb = os.path.getsize(args.out) / 1e6
    log(f"done -> {args.out} ({size_mb:.1f} MB, {n} problems)")


if __name__ == "__main__":
    main()
