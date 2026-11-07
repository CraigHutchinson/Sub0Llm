#!/usr/bin/env python3
"""Bounded ChatML console proving ground over the mapped Qwen backend."""
import argparse
import json
import re
import subprocess
import tempfile
from pathlib import Path


def render(messages):
    """Verified plain text user/assistant ChatML subset; no tools, vision or thinking options."""
    return "".join(f"<|im_start|>{role}\n{content}<|im_end|>\n" for role, content in messages) + "<|im_start|>assistant\n"


class Session:
    def __init__(self, backend, model, tokenizer, maximum, threads):
        self.base = [str(backend), str(model), "--tokenizer-dir", str(tokenizer),
                     "--projection-threads", str(threads), "--continuation-steps", str(maximum - 1)]
        self.messages = []
        self.last_context = None

    def reset(self):
        self.messages.clear()
        self.last_context = None

    def turn(self, user):
        pending = self.messages + [("user", user)]
        command = self.base + ["--text", render(pending)]
        inspected = subprocess.run(command + ["--inspect-context"], capture_output=True, text=True)
        match = re.search(r"CONTEXT prompt=(\d+) reserve=(\d+) capacity=(\d+)", inspected.stdout)
        context = dict(zip(("prompt", "reserve", "capacity"), map(int, match.groups()))) if match else None
        if inspected.returncode:
            # Failed turns leave the accepted conversation intact.
            raise RuntimeError(f"prompt rejected; context={context}; {inspected.stderr.strip()}")
        if context is None:
            raise RuntimeError("backend omitted context accounting")
        print(json.dumps({"context": context}), flush=True)
        with tempfile.TemporaryDirectory(prefix="sub0-chat-") as directory:
            output = Path(directory) / "response.txt"
            completed = subprocess.run(command + ["--response-file", str(output)], capture_output=True)
            if completed.returncode:
                raise RuntimeError(completed.stderr.decode("utf-8", errors="replace"))
            # Do not silently change replayed context when a capped byte-BPE response ends mid-codepoint.
            try:
                response = output.read_bytes().decode("utf-8")
            except UnicodeDecodeError as error:
                raise RuntimeError("response cap ended inside a UTF-8 character; increase the token cap") from error
        self.messages = pending + [("assistant", response)]
        self.last_context = context
        return response


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--backend", type=Path, default=root / "out/build/qwen35-native/sub0llm-dense-token")
    parser.add_argument("--tokenizer-dir", type=Path, default=root / "out/build/qwen35-native/tokenizer")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--threads", type=int, choices=range(1, 9), default=8)
    parser.add_argument("--turn", action="append", help="Scripted turn or /reset or /context; repeatable")
    args = parser.parse_args()
    if not 1 <= args.max_new_tokens <= 128:
        parser.error("max-new-tokens must be 1..128")
    session = Session(args.backend.resolve(), args.model.resolve(), args.tokenizer_dir.resolve(),
                      args.max_new_tokens, args.threads)
    scripted = iter(args.turn) if args.turn is not None else None
    while True:
        try:
            text = next(scripted) if scripted is not None else input("You: ")
        except (StopIteration, EOFError):
            break
        if text == "/quit":
            break
        if text == "/reset":
            session.reset()
            print(json.dumps({"reset": True, "messages": 0}), flush=True)
            continue
        if text == "/context":
            print(json.dumps({"last_context": session.last_context, "messages": len(session.messages)}), flush=True)
            continue
        if not text.strip():
            continue
        try:
            response = session.turn(text)
            print(json.dumps({"assistant": response, "messages": len(session.messages)}, ensure_ascii=False), flush=True)
        except (RuntimeError, OSError) as error:
            print(json.dumps({"error": str(error)}), flush=True)
            if scripted is not None:
                return 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
