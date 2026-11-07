#!/usr/bin/env python3
"""Bounded ChatML console proving ground over the mapped Qwen backend."""
import argparse
import json
import re
import subprocess
import tempfile
from pathlib import Path


def render(messages):
    """Embedded template subset: enable_thinking=False, preserve_thinking=False, text only."""
    if not messages or messages[-1][0] != "user" or any(role not in ("user", "assistant") for role, _ in messages):
        raise ValueError("chat transcript must contain user/assistant text and end with a user")
    return "".join(f"<|im_start|>{role}\n{content.strip()}<|im_end|>\n" for role, content in messages) + "<|im_start|>assistant\n<think>\n\n</think>\n\n"


class Session:
    def __init__(self, backend, model, tokenizer, maximum, threads, iq1s_kernel=None):
        self.base = [str(backend), str(model), "--tokenizer-dir", str(tokenizer),
                     "--projection-threads", str(threads), "--continuation-steps", str(maximum - 1)]
        if iq1s_kernel is not None:
            self.base += ["--iq1s-kernel", iq1s_kernel]
        self.messages = []
        self.last_context = None
        self.last_result = None

    def reset(self):
        self.messages.clear()
        self.last_context = None
        self.last_result = None

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
        finished = re.search(rb"(?:^|\n)TEXT_RESULT generated=(\d+) finish=(eos|length)\n\Z", completed.stdout)
        if finished is None:
            raise RuntimeError("backend omitted completion diagnostics; rebuild the backend")
        self.last_result = {"generated_tokens": int(finished.group(1)), "finish_reason": finished.group(2).decode("ascii")}
        self.messages = pending + [("assistant", response)]
        self.last_context = context
        return response


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    local_model = root.parent.parent / "Models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ1_S.gguf"
    parser.add_argument("model", type=Path, nargs="?", default=local_model, help="GGUF model; defaults to the sibling Models download")
    parser.add_argument("--backend", type=Path, default=root / "out/build/qwen35-native/sub0llm-dense-token")
    parser.add_argument("--tokenizer-dir", type=Path, default=root / "out/build/qwen35-native/tokenizer")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--threads", type=int, choices=range(1, 9), default=8)
    parser.add_argument("--iq1s-kernel", choices=("baseline", "scalar", "native"), help="Opt-in kernel experiment; requires the matching rebuilt backend")
    parser.add_argument("--turn", action="append", help="Scripted turn or /reset or /context; repeatable")
    args = parser.parse_args()
    if not 1 <= args.max_new_tokens <= 128:
        parser.error("max-new-tokens must be 1..128")
    session = Session(args.backend.resolve(), args.model.resolve(), args.tokenizer_dir.resolve(),
                      args.max_new_tokens, args.threads, args.iq1s_kernel)
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
            print(json.dumps({"assistant": response, "messages": len(session.messages), **session.last_result}, ensure_ascii=False), flush=True)
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
