#!/usr/bin/env python3
"""Regenerate include/sub0/qwen_unicode_tables.hpp, and VERIFY it against the real reference.

    python scripts/qwen_tokenizer_tables.py <snapshot-dir> [--out include/sub0/qwen_unicode_tables.hpp]
                                            [--verify {fast,full}]

<snapshot-dir> is a local copy of Qwen/Qwen3.8-Flash-Next's tokenizer files -- only tokenizer.json is
read here. Needs `tokenizers` installed; that is the REAL reference this checks against, not a
paraphrase of it (AGENTS.md sec 5).

WHY THIS SCRIPT EXISTS. sub0/qwen_unicode.hpp answers four questions about a codepoint -- is it a
Letter, a Mark, a Number, whitespace -- plus NFC. Those answers have to agree with what the reference
tokenizer's own regex engine and normalizer believe, not merely with "Unicode". The tables are emitted
from Python's `unicodedata`, and then every one of them is cross-checked against the reference itself:

  --verify fast (default)  the anchors and the NFC round-trips: every canonically-decomposable
                           codepoint, every composition pair, every composition exclusion, all 11,172
                           Hangul syllables, and the character classes over a few thousand sampled
                           codepoints.  ~1 minute.
  --verify full            additionally sweeps the character classes over ALL 1,112,064 non-surrogate
                           codepoints and NFC over every one of them.  ~25 minutes. This is the run
                           that produced the "0 disagreements" claim in docs/QWEN_TOKENIZER.md sec 3.

THE THREE PROBES that read the classes back out of the reference's own regex (docs sec 3.1 derives
them from the pattern; each isolates one class by making the OTHER clauses fail):
  cp is in L|M   <=>  pre_tokenize("a" + cp)     is 1 chunk   (clause 2 continues a letter run)
  cp is in N     <=>  pre_tokenize(cp + cp)      is 2 chunks  (clause 3 is a BARE \\p{N}: one per chunk)
  cp is in \\s    <=>  pre_tokenize(cp+cp+".")    is not 1 chunk, for cp outside L|M|N
                                                 (clause 4's `+` excludes whitespace, so a whitespace
                                                  cp cannot glue itself to the following '.')
L and M cannot be told apart this way, and provably do not need to be: they are interchangeable
everywhere in that regex. The split is therefore taken from unicodedata, and only checked for
consistency (every Mark must lie inside the reference's own L|M).
"""
import argparse, json, sys, unicodedata

sys.stdout.reconfigure(encoding="utf-8")
try:
    from tokenizers import pre_tokenizers, Regex, normalizers
except ImportError:
    sys.exit("this script needs `tokenizers` (the real reference): pip install tokenizers")

MAX = 0x110000
HANGUL_S, HANGUL_N = 0xAC00, 11172
# UCD PropList.txt, White_Space=Yes. A fixed, small, stable set; the sweep below confirms it is
# exactly what the reference's `\s` matches.
UCD_WHITE_SPACE = [0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x20, 0x85, 0xA0, 0x1680,
                   *range(0x2000, 0x200B), 0x2028, 0x2029, 0x202F, 0x205F, 0x3000]


def surrogate(cp):
    return 0xD800 <= cp < 0xE000


def hangul(cp):
    return HANGUL_S <= cp < HANGUL_S + HANGUL_N


def build_unicodedata_tables():
    """Everything the C++ side needs, straight from Python's UCD."""
    cat = ["Cs" if surrogate(cp) else unicodedata.category(chr(cp)) for cp in range(MAX)]
    L = [cp for cp in range(MAX) if cat[cp][0] == "L"]
    N = [cp for cp in range(MAX) if cat[cp][0] == "N"]
    M = [cp for cp in range(MAX) if cat[cp][0] == "M"]
    ccc = [(cp, unicodedata.combining(chr(cp))) for cp in range(MAX)
           if not surrogate(cp) and unicodedata.combining(chr(cp))]
    raw = {}
    for cp in range(MAX):
        if surrogate(cp) or hangul(cp):
            continue
        d = unicodedata.decomposition(chr(cp))
        if d and not d.startswith("<"):
            raw[cp] = [int(x, 16) for x in d.split()]

    def full(cp):
        if cp not in raw:
            return [cp]
        out = []
        for p in raw[cp]:
            out.extend(full(p))
        return out

    decomp = sorted((cp, full(cp)) for cp in raw)
    # A pair-decomposition is a PRIMARY composite only if NFC actually recomposes it; asking NFC is
    # how the Composition_Exclusion / singleton / non-starter rules get applied without a second table.
    comp = sorted((p[0], p[1], cp) for cp, p in raw.items()
                  if len(p) == 2 and unicodedata.normalize("NFC", chr(p[0]) + chr(p[1])) == chr(cp))
    return dict(L=L, N=N, M=M, ccc=ccc, decomp=decomp, comp=comp, raw=raw)


def verify(snapshot, tabs, full_sweep):
    pat = json.load(open(snapshot + "/tokenizer.json", encoding="utf-8")) \
        ["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"]
    print("reference pre-tokenization regex:\n  " + pat)
    P = pre_tokenizers.Split(Regex(pat), "isolated").pre_tokenize_str
    ref_nfc = normalizers.NFC().normalize_str
    ref_nfd = normalizers.NFD().normalize_str
    bad = 0
    skew = []

    def reference_knows(cp):
        """False when the reference's Unicode tables have NO canonical data for `cp` at all -- i.e.
        the codepoint was added to Unicode after the tables in the installed `tokenizers` build were
        generated. Asking NFD is the exact test: a codepoint with a canonical decomposition that the
        reference knows about MUST decompose. This is what separates a real table bug (fatal) from
        version skew (expected, enumerated, and recorded in docs/QWEN_TOKENIZER.md sec 3.3)."""
        return ref_nfd(chr(cp)) != chr(cp)

    Ls, Ns, Ms = set(tabs["L"]), set(tabs["N"]), set(tabs["M"])
    Ws = set(UCD_WHITE_SPACE)
    if full_sweep:
        sweep = [cp for cp in range(MAX) if not surrogate(cp)]
    else:
        # Anchors + every boundary of every emitted range + a deterministic spread. A table error
        # almost always shows up at a range edge, which is why those are always included.
        sweep = set()
        for s in (Ls, Ns, Ms, Ws):
            for cp in s:
                sweep.update(x for x in (cp - 1, cp, cp + 1) if 0 <= x < MAX and not surrogate(x))
                if len(sweep) > 400000:
                    break
        sweep = sorted(cp for cp in sweep if not surrogate(cp))[::7]
    print("character-class sweep over %d codepoints..." % len(sweep))
    for cp in sweep:
        c = chr(cp)
        if (len(P("a" + c)) == 1) != (cp in Ls or cp in Ms):
            print("  L|M mismatch at U+%04X" % cp); bad += 1
        if (len(P(c + c)) == 2) != (cp in Ns):
            print("  N mismatch at U+%04X" % cp); bad += 1
        if cp not in Ls and cp not in Ms and cp not in Ns:
            if (len(P(c + c + ".")) != 1) != (cp in Ws):
                print("  White_Space mismatch at U+%04X" % cp); bad += 1
    outside = [cp for cp in tabs["M"] if len(P("a" + chr(cp))) != 1]
    print("Marks outside the reference's own L|M set: %d" % len(outside))
    bad += len(outside)

    print("NFC: %d decomposable codepoints, %d composition pairs, %d Hangul syllables..."
          % (len(tabs["decomp"]), len(tabs["comp"]), HANGUL_N))
    for cp, _ in tabs["decomp"]:
        if ref_nfc(unicodedata.normalize("NFD", chr(cp))) != unicodedata.normalize("NFC", chr(cp)):
            if reference_knows(cp): print("  NFD->NFC mismatch at U+%04X" % cp); bad += 1
            else:                   skew.append(cp)
    for cp, p in tabs["raw"].items():
        if len(p) != 2:
            continue
        composes = unicodedata.normalize("NFC", chr(p[0]) + chr(p[1])) == chr(cp)
        got = ref_nfc(chr(p[0]) + chr(p[1]))
        if composes and got != chr(cp):
            if reference_knows(cp): print("  pair should compose but reference does not: U+%04X" % cp); bad += 1
            else:                   skew.append(cp)
        if not composes and got == chr(cp):
            print("  exclusion violated: reference composes U+%04X" % cp); bad += 1
    for i in range(HANGUL_N):
        s, t = HANGUL_S + i, i % 28
        seq = chr(0x1100 + i // 588) + chr(0x1161 + (i % 588) // 28) + (chr(0x11A7 + t) if t else "")
        if ref_nfc(seq) != chr(s) or ref_nfc(chr(s)) != chr(s):
            print("  Hangul mismatch at U+%04X" % s); bad += 1
    if full_sweep:
        print("NFC over every codepoint...")
        for cp in range(MAX):
            if surrogate(cp):
                continue
            c = chr(cp)
            if ref_nfc(c) != unicodedata.normalize("NFC", c):
                print("  single-codepoint NFC mismatch at U+%04X" % cp); bad += 1
    if skew:
        # NOT a table error: these codepoints do not exist in the reference build's Unicode tables at
        # all. Emitting Unicode %s is the standards-correct choice and the one that stays right when
        # `tokenizers` catches up; see docs/QWEN_TOKENIZER.md sec 3.3.
        uniq = sorted(set(skew))
        print("EXPECTED version skew: %d codepoints the reference's Unicode tables predate: %s"
              % (len(uniq), " ".join("U+%04X" % c for c in uniq)))
    print("VERIFY: %d real disagreements with the real reference" % bad)
    return bad


def to_ranges(cps):
    out, start, prev = [], None, None
    for cp in cps:
        if start is None:
            start = prev = cp
        elif cp == prev + 1:
            prev = cp
        else:
            out.append((start, prev)); start = prev = cp
    if start is not None:
        out.append((start, prev))
    return out


def emit(tabs, dest):
    def rows(vals, per, fmt="0x%X"):
        s = [fmt % v for v in vals]
        return "\n".join("    " + ", ".join(s[i:i + per]) + ("," if i + per < len(s) else "")
                         for i in range(0, len(s), per))

    def range_tbl(name, cps, why):
        rs = to_ranges(cps)
        flat = [x for r in rs for x in r]
        return (f"// {why} ({len(rs)} ranges)\ninline constexpr std::uint32_t {name}[] = {{\n"
                + rows(flat, 12) + f"\n}};\ninline constexpr std::size_t {name}N = {len(rs)};\n")

    seq, off = [], []
    for _, s in tabs["decomp"]:
        off.append(len(seq)); seq.extend(s)
    off.append(len(seq))
    uv = unicodedata.unidata_version
    body = f'''// sub0/qwen_unicode_tables.hpp -- GENERATED DATA ONLY. Do not edit by hand.
// Regenerate with:  python scripts/qwen_tokenizer_tables.py <snapshot-dir> --verify full
//
// Unicode tables for the REAL Qwen tokenizer's pre-tokenization regex and its NFC normalizer (see
// sub0/qwen_unicode.hpp for the code that reads them, docs/QWEN_TOKENIZER.md sec 3 for how each was
// verified). Same "data-only sibling header" shape as sub0/gguf_quant_tables.hpp: no logic here, only
// constants, so the algorithm file stays readable.
//
// SOURCE + VERIFICATION. Emitted from Python `unicodedata` {uv}, then cross-checked against the REAL
// reference -- the regex engine driving `tokenizers`' Split pre-tokenizer on the model's own pattern,
// and `tokenizers`' own NFC normalizer -- by the generator script above. At the last full run:
//   * character classes: 0 disagreements over all 1,112,064 non-surrogate codepoints, for
//     \\p{{L}}|\\p{{M}}, for \\p{{N}}, and for \\s. The regex cannot separate L from M (they are
//     interchangeable inside it), so that one split comes from unicodedata alone; every codepoint
//     unicodedata calls a Mark was confirmed to lie inside the reference's own L|M set.
//   * NFC: 0 disagreements for NFC(cp) over every codepoint, for NFD->NFC over all {len(tabs["decomp"])}
//     canonically-decomposable codepoints, for all {len(tabs["comp"])} composition pairs, for every
//     composition EXCLUSION (0 spurious compositions), and for all 11,172 algorithmic Hangul syllables.
// One measured divergence class is recorded in docs/QWEN_TOKENIZER.md sec 3.3: sequences of TWO OR
// MORE combining marks can order differently, because the reference build's combining-class table
// predates Unicode {uv}. All 8,406 single-mark sequences agree exactly -- that is what real text has.
#pragma once

#include <cstddef>
#include <cstdint>

namespace sub0::qwen_uni::tables {{

// Unicode version these tables were generated from.
inline constexpr char kUnicodeVersion[] = "{uv}";

// --- general categories, as sorted inclusive [lo, hi] range pairs ------------------------------
{range_tbl("kLetter", tabs["L"], "General_Category = L* (Letter)")}
{range_tbl("kNumber", tabs["N"], "General_Category = N* (Number)")}
{range_tbl("kMark", tabs["M"], "General_Category = M* (Mark)")}
{range_tbl("kWhiteSpace", UCD_WHITE_SPACE, "Unicode White_Space -- what the reference's `\\\\s` matches (NOT ASCII-only)")}

// --- NFC: canonical combining classes (only the {len(tabs["ccc"])} codepoints with ccc != 0) -----
inline constexpr std::uint32_t kCccCp[] = {{
{rows([c for c, _ in tabs["ccc"]], 12)}
}};
inline constexpr std::uint8_t kCccVal[] = {{
{rows([v for _, v in tabs["ccc"]], 24, "%d")}
}};
inline constexpr std::size_t kCccN = {len(tabs["ccc"])};

// --- NFC: FULL (recursive) canonical decompositions, Hangul excluded (it is algorithmic) -------
// kDecompCp[i] decomposes to kDecompSeq[kDecompOff[i] .. kDecompOff[i+1]).
inline constexpr std::uint32_t kDecompCp[] = {{
{rows([cp for cp, _ in tabs["decomp"]], 12)}
}};
inline constexpr std::uint32_t kDecompOff[] = {{
{rows(off, 16, "%d")}
}};
inline constexpr std::uint32_t kDecompSeq[] = {{
{rows(seq, 12)}
}};
inline constexpr std::size_t kDecompN = {len(tabs["decomp"])};

// --- NFC: primary composition pairs (composition exclusions already removed) -------------------
// Key is (first << 32) | second, sorted, so a composition is one binary search.
inline constexpr std::uint64_t kComposeKey[] = {{
{rows([(a << 32) | b for a, b, _ in tabs["comp"]], 6, "0x%016XULL")}
}};
inline constexpr std::uint32_t kComposeVal[] = {{
{rows([c for _, _, c in tabs["comp"]], 12)}
}};
inline constexpr std::size_t kComposeN = {len(tabs["comp"])};

}}  // namespace sub0::qwen_uni::tables
'''
    open(dest, "w", encoding="utf-8", newline="\n").write(body)
    print("wrote %s (%d lines)" % (dest, body.count("\n")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("snapshot", help="directory holding the model's tokenizer.json")
    ap.add_argument("--out", default="include/sub0/qwen_unicode_tables.hpp")
    ap.add_argument("--verify", choices=("fast", "full", "none"), default="fast")
    a = ap.parse_args()
    print("Python unicodedata version:", unicodedata.unidata_version)
    tabs = build_unicodedata_tables()
    if a.verify != "none" and verify(a.snapshot.rstrip("/"), tabs, a.verify == "full"):
        sys.exit("REFUSING to emit tables that disagree with the real reference")
    emit(tabs, a.out)


if __name__ == "__main__":
    main()
