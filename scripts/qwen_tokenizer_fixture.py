#!/usr/bin/env python3
"""Regenerate tests/fixtures/qwen_tokenizer/* from the REAL reference tokenizer.

    python scripts/qwen_tokenizer_fixture.py <snapshot-dir> [--out tests/fixtures/qwen_tokenizer]

<snapshot-dir> is a local copy of Qwen/Qwen3.8-Flash-Next's tokenizer files (tokenizer.json and
tokenizer_config.json are read). Needs `tokenizers`; `transformers` too, for the cross-check.

THE ORACLE, and why it is not the obvious one (docs/QWEN_TOKENIZER.md sec 2.1). Primary:
`tokenizers.Tokenizer.from_file(tokenizer.json)` -- the model's OWN exported artifact. Cross-checked
case by case against `transformers.models.qwen3_5.Qwen3_5Tokenizer`, which is what transformers' own
model_type -> tokenizer map selects for this model's `qwen4_exp`. Both build the SAME pipeline, and
every case below agreed.

NOT the oracle: plain `AutoTokenizer.from_pretrained(...)`. It honours tokenizer_config.json's stale
`tokenizer_class: "Qwen2Tokenizer"` field, and transformers v5's Qwen2Tokenizer REBUILDS the pipeline
from vocab.json+merges.txt with its own hardcoded PRETOKENIZE_REGEX -- an OLDER regex that lacks
\\p{M} -- discarding the model's own. Gating against it would have baked a real, measurable
mis-tokenization of every combining mark into this engine.
"""
import argparse, sys, os, json, unicodedata
sys.stdout.reconfigure(encoding='utf-8')
from tokenizers import Tokenizer, pre_tokenizers, Regex, normalizers
from transformers.models.qwen3_5.tokenization_qwen3_5 import Qwen3_5Tokenizer

_ap = argparse.ArgumentParser()
_ap.add_argument("snapshot", help="directory holding tokenizer.json + tokenizer_config.json")
_ap.add_argument("--out", default="tests/fixtures/qwen_tokenizer")
_args = _ap.parse_args()
SNAP = _args.snapshot.rstrip("/\\")
DEST = _args.out
os.makedirs(DEST, exist_ok=True)

raw = Tokenizer.from_file(SNAP + "/tokenizer.json")
hf = Qwen3_5Tokenizer.from_pretrained(SNAP)
cfg = json.load(open(SNAP + "/tokenizer_config.json", encoding="utf-8"))
tjs = json.load(open(SNAP + "/tokenizer.json", encoding="utf-8"))
PAT = tjs["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"]
split = pre_tokenizers.Split(Regex(PAT), "isolated")
nfc = normalizers.NFC()
SPECIALS = [v["content"] for k, v in sorted(cfg["added_tokens_decoder"].items(), key=lambda kv: int(kv[0]))]

def hx(s): return s.encode("utf-8").hex()

# ------------------------------------------------------------------ the cases
C = []
def add(label, s): C.append((label, s))

add("empty", "")
for n in range(1, 9): add("spaces_%d" % n, " " * n)
for n in range(1, 5): add("tabs_%d" % n, "\t" * n)
for n in range(1, 5): add("newlines_%d" % n, "\n" * n)
add("crlf", "\r\n"); add("cr", "\r"); add("crlf_x2", "\r\n\r\n")
add("mixed_ws", "  \t  \n\n \t x")
add("ws_before_word", "   hello")
add("ws_after_word", "hello   ")
add("ws_only_trailing", "hello \n ")
add("space_newline_space", "a \n b")
add("para", "a  \n\n  b")
add("indent_block", "def f():\n    return 1\n\n\ndef g():\n\treturn 2\n")

for w in ["a", "hello", "Hello", "HELLO", "hello world", "The quick brown fox jumps over the lazy dog.",
          "antidisestablishmentarianism", "supercalifragilisticexpialidocious"]:
    add("word_" + w[:12].replace(" ", "_"), w)
    add("lead_space_" + w[:12].replace(" ", "_"), " " + w)

for c in ["don't", "DON'T", "Don'T", "they're", "THEY'RE", "we've", "I'm", "we'll", "WE'LL", "he'd",
          "'s", "'S", "'t", "'re", "'ve", "'m", "'ll", "'d", "x'sy", "'\u017f", "'\u017fx", "a'\u017fb",
          "'\u212ax", "rock'n'roll", "y'all'd've"]:
    add("contr_%d" % len(C), c)

for d in ["0", "7", "10", "123456", "3.14159", "1,000,000", "-42", "2026-09-08", "0x1F", "1e10",
          "\u0660\u0661\u0662", "\u4e00\u4e8c\u4e09", "\u2160\u2161", "\u00bd", "\u2460"]:
    add("num_%d" % len(C), d)

for t in ["\u4f60\u597d\u4e16\u754c", "\u3053\u3093\u306b\u3061\u306f\u4e16\u754c",
          "\uc548\ub155\ud558\uc138\uc694", "\u1112\u1161\u11ab\u1100\u1173\u11af",   # Hangul jamo -> composes
          "\u041f\u0440\u0438\u0432\u0435\u0442 \u043c\u0438\u0440", "\u0393\u03b5\u03b9\u03b1 \u03c3\u03bf\u03c5",
          "\u05e9\u05dc\u05d5\u05dd", "\u0645\u0631\u062d\u0628\u0627 \u0628\u0627\u0644\u0639\u0627\u0644\u0645",
          "\u0e2a\u0e27\u0e31\u0e2a\u0e14\u0e35", "\u0928\u092e\u0938\u094d\u0924\u0947"]:
    add("script_%d" % len(C), t)

# The \p{M} regression set: scripts whose marks have NO precomposed form, so NFC leaves them for the
# regex to see. These are the inputs where the OLDER, \p{M}-less regex that
# AutoTokenizer/Qwen2Tokenizer would have used produces different ids (docs/QWEN_TOKENIZER.md sec 2.1).
for t in ["\u0e01\u0e31\u0e19", "\u0915\u094d\u0937", "\u0d28\u0d4d\u0d31",
          "\u0623\u064e\u0647\u0652\u0644\u0627\u064b", "\u05d0\u05b8\u05dc\u05b6\u05e3",
          "a\u0316", "x\u0316y"]:
    add("mark_%d" % len(C), t)

for t in ["caf\u00e9", "cafe\u0301", "na\u00efve", "nai\u0308ve", "pi\u00f1ata", "\u00c5ngstr\u00f6m",
          "A\u030angstro\u0308m", "\u1ebf", "e\u0302\u0301", "e\u0301\u0302", "Ti\u1ebfng Vi\u1ec7t",
          "Tie\u0301\u0302ng Vie\u0323\u0302t", "\u00fc\u0308", "q\u0307\u0323", "q\u0323\u0307"]:
    add("nfc_%d" % len(C), t)

for t in ["\U0001F600", "a\U0001F600b", "\U0001F600\U0001F601\U0001F602",
          "\U0001F468\u200D\U0001F469\u200D\U0001F467\u200D\U0001F466", "\U0001F1EC\U0001F1E7",
          "\U0001F44D\U0001F3FD", "\U00010330\U00010331", "\U0002A6B2"]:
    add("astral_%d" % len(C), t)

for t in ["\u00a0", "a\u00a0b", "\u3000", "\u3000x", "x\u3000", "\u2028", "\u2029", "\u1680",
          "\u2000\u2001\u2002", "\u205f", "\u202f", "a\u00adb", "a\u200bb", "\ufeffabc", "a\u0000b", "\u007f"]:
    add("wsx_%d" % len(C), t)

for t in ["def f(x): return x*2", "a->b", "x == y", "#include <stdio.h>", "if (a && b) { c(); }",
          '{"k": [1, 2, 3], "b": null}', "https://example.com/a?b=c#d", "user@example.com",
          "C:\\Users\\craig\\file.txt", "/usr/bin/env python3", "SELECT * FROM t WHERE x>1;",
          "**bold** _em_ `code`\n- item\n- item\n", "$5.00 + 20% = \u20ac6.00", "a_b-c.d/e",
          "<<<>>>", "!!!???", "....", "---", "===", "\\n\\t"]:
    add("code_%d" % len(C), t)

for s in SPECIALS:
    add("spec_alone_" + s.strip("<>|/"), s)
    add("spec_mid_" + s.strip("<>|/"), "hi" + s + "there")
    add("spec_ws_" + s.strip("<>|/"), " " + s + " ")
add("spec_pair", "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n")
add("spec_partial", "<|im_end|")
add("spec_nested", "<<|im_end|>>")
add("spec_run", "<|endoftext|><|endoftext|><|im_end|>")
add("spec_think", "<think>reasoning</think>answer")
add("spec_tool", "<tool_call>{\"name\": \"f\"}</tool_call>")

add("prose_1", "The 2024 report showed a 12.5% increase in revenue, driven by strong demand.")
add("prose_2", "In 1969, Apollo 11 landed on the Moon; Neil Armstrong stepped out at 02:56 UTC.")
add("prose_3", "Don't forget: it's the tokenizer's job to be *exactly* reproducible \u2014 no \"close enough\".")
add("prose_4", "Mixed \u4e2d\u6587 and English \u0442\u0435\u043a\u0441\u0442 with num6ers and sym#bols.")
add("prose_long", ("Byte-level BPE tokenizers map every byte to a printable codepoint so that no input "
                   "is ever out-of-vocabulary. The pre-tokenization regex then splits text into chunks, "
                   "and merges never cross a chunk boundary.\n\n") * 2)

# ------------------------------------------------------------------ emit
enc_rows, mismatch = [], 0
for label, s in C:
    a = raw.encode(s).ids
    b = hf.encode(s)
    if a != b:
        mismatch += 1
        print("ORACLE DISAGREEMENT", label, repr(s), a, b)
    # skip_special_tokens=False: keep every token's text. That is transformers' own default and the
    # only behaviour under which decode(encode(x)) == NFC(x); `tokenizers`' raw default (True) drops
    # the 20 tokens flagged `special`, which is a DISPLAY policy, not a decode.
    d = raw.decode(a, skip_special_tokens=False)
    if hf.decode(a) != d:
        print("DECODE DISAGREEMENT", label, repr(s), repr(d), repr(hf.decode(a)))
    enc_rows.append((label, hx(s), a, hx(d)))
print("encode/decode cases:", len(enc_rows), "oracle disagreements:", mismatch)

# decode-only cases: id sequences a generation loop can produce that no encode() ever emits.
V = raw.get_vocab()
dec_only = [
    ("partial_utf8_1", [V[chr(0xE4)]]),
    ("partial_utf8_2", [V[chr(0xE4)], V[chr(0xBD)]]),
    ("partial_then_text", [V[chr(0xE4)], V[chr(0xBD)], V["a"]]),
    ("lone_continuation", [V[chr(0xBD)]]),
    ("byte_run", [V[chr(0xE4)], V[chr(0xBD)], V[chr(0x121)]]),        # 0xE4 0xBD 0xA0 == U+4F60
    ("pad_row_low", [248077]),                                        # the model's VOCAB axis is 248320;
    ("pad_row_high", [248319]),                                       # rows above the tokenizer are padding
    ("pad_between_text", [V["a"], 248200, V["b"]]),
    ("empty", []),
]
dec_rows = [(l, " ".join(str(i) for i in ids), hx(raw.decode(ids, skip_special_tokens=False)))
            for l, ids in dec_only]
print("decode-only cases:", len(dec_rows))

pre_rows = []
for label, s in C:
    n = nfc.normalize_str(s)
    pre_rows.append((label, hx(n), [hx(c) for c, _ in split.pre_tokenize_str(n)]))

nfc_rows = []
for label, s in C:
    nfc_rows.append((label, hx(s), hx(nfc.normalize_str(s))))
# plus every codepoint whose NFC differs from itself, in isolation and after a letter
extra = 0
for cp in range(0x110000):
    if 0xD800 <= cp < 0xE000: continue
    c = chr(cp)
    if nfc.normalize_str(c) != c:
        nfc_rows.append(("cp_%04X" % cp, hx(c), hx(nfc.normalize_str(c))))
        extra += 1
print("nfc cases:", len(nfc_rows), "(of which single-codepoint:", extra, ")")

def write(name, header, rows):
    with open(os.path.join(DEST, name), "w", encoding="utf-8", newline="\n") as f:
        f.write(header)
        for r in rows:
            f.write("\t".join(r) + "\n")

HDR = ("# GENERATED from the REAL Qwen/Qwen3.8-Flash-Next tokenizer -- do not edit by hand.\n"
       "# See tests/fixtures/qwen_tokenizer/manifest.json for provenance and\n"
       "# docs/QWEN_TOKENIZER.md for how it is used. Every text column is HEX-ENCODED UTF-8, so\n"
       "# control characters, tabs and newlines survive the file format unambiguously.\n")

write("encode_cases.tsv", HDR + "# label <TAB> input_hex <TAB> ids (space separated) <TAB> decode_hex\n",
      [(l, h, " ".join(str(i) for i in ids), dh) for l, h, ids, dh in enc_rows])
write("pretokenize_cases.tsv", HDR + "# label <TAB> nfc_input_hex <TAB> chunk_hex (space separated)\n",
      [(l, h, " ".join(cs)) for l, h, cs in pre_rows])
write("nfc_cases.tsv", HDR + "# label <TAB> input_hex <TAB> nfc_hex\n", nfc_rows)
write("decode_cases.tsv", HDR + "# label <TAB> ids (space separated) <TAB> decode_hex\n", dec_rows)

json.dump({
    "description": "Real-reference fixture for sub0::qwen_tok, this engine's reimplementation of the "
                   "Qwen3.8-Flash-Next byte-level BPE tokenizer. Every expected value was produced by "
                   "the real tokenizer, not hand-derived.",
    "source": {
        "repo": "Qwen/Qwen3.8-Flash-Next",
        "snapshot": os.path.basename(SNAP),
        "files_read": ["tokenizer.json", "tokenizer_config.json"],
        "oracle": "tokenizers.Tokenizer.from_file(tokenizer.json) -- the model's own artifact",
        "cross_check": "transformers.models.qwen3_5.Qwen3_5Tokenizer (transformers' own model_type map "
                       "sends qwen4_exp here); every case agreed",
        "NOT_the_oracle": "transformers.AutoTokenizer.from_pretrained(...), which honours the stale "
                          "tokenizer_class='Qwen2Tokenizer' field and rebuilds the pipeline with an "
                          "OLDER pre-tokenization regex that lacks \\p{M} -- see docs/QWEN_TOKENIZER.md sec 2.1",
        "tokenizers_version": __import__("tokenizers").__version__,
        "transformers_version": __import__("transformers").__version__,
        "pretokenize_regex": PAT,
    },
    "files": {
        "encode_cases.tsv": "label / hex(input) / token ids / hex(reference decode(ids))",
        "pretokenize_cases.tsv": "label / hex(NFC(input)) / hex of each pre-token chunk",
        "nfc_cases.tsv": "label / hex(input) / hex(NFC(input))",
        "decode_cases.tsv": "label / token ids / hex(reference decode(ids)) -- id sequences encode() "
                            "never produces: truncated multi-byte characters and the model's padding rows",
    },
    "decode_convention": "skip_special_tokens=False (transformers' own default). `tokenizers`' raw "
                         "Tokenizer.decode defaults to True and would drop the 20 tokens flagged "
                         "`special`; that is a display policy, not a decode, and it breaks the "
                         "decode(encode(x)) == NFC(x) round trip.",
    "counts": {"encode_cases": len(enc_rows), "pretokenize_cases": len(pre_rows),
               "nfc_cases": len(nfc_rows), "decode_cases": len(dec_rows)},
}, open(os.path.join(DEST, "manifest.json"), "w", encoding="utf-8"), indent=2)
print("wrote", DEST)
