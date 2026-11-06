# Corpus data workflow: findings and proposal

Status: PROPOSAL (not implemented). Written 2026-11-06 after a fresh Linux checkout could not run the
neutral gate. Findings marked *(verified)* were re-checked directly; the rest come from a read-only
review pass and should be re-checked before work starts.

## Findings

- `data/english.txt` is an LFS pointer **and** `git lfs pull` cannot fix it. `.gitattributes:13` sets
  `-filter`, so the committed blob IS the pointer text (`git check-attr filter` = unset; `git cat-file -p
  HEAD:data/english.txt` prints `version https://git-lfs...`, size 9742). *(verified)*
  The `.gitattributes` comment itself warns this happens for a file already stored in LFS.
- On a pointer, `sub0llm-configure --corpus data/english.txt` fails late with "no `<|endoftext|>` markers
  found" (`tools/configurator.cpp` ~2333), which blames the marker, not the pointer.
- `data/gsm8k.txt`, needed by HANDOVER step 8's neutral gate, is not in the checkout. It comes from
  `scripts/get_gsm8k.py`.
- Five fetchers exist (`scripts/get_{gsm8k,fineweb,cosmopedia,minipile,tinystories}.py`). None pins a
  Hugging Face revision; none records an expected size or SHA-256.
- `scripts/make_ppl_fixture.py` pins only the OUTPUT hash, so it reproduces only from byte-identical corpora.
  The fetch parameters the author used (fineweb `--shards`, cosmopedia `--target-gb`) are not recorded.
  `data/tokenizer_calibration.txt` has exact byte counts for four corpora and is the only size evidence.
- `scripts/doctor.py` checks only `english.txt` (fixed to detect pointers correctly in PR 5).

## Proposal

Smallest step first (no new surface, AGENTS section 8):
1. Configurator: detect a `version https://git-lfs` header and name the file in the message; name the path
   in "cannot read corpus".
2. Restore `data/english.txt` as a real plain blob (needs the real 9,742 bytes from the author's machine or
   the LFS remote).

Then:
3. `scripts/corpus.py` with `status`, `fetch <name>`, `verify <name>`, driven by `data/corpora.json`
   (name, source repo + revision, file, bytes, sha256, marker expectation, which workflows need it).
   `fetch` delegates to the existing `get_*.py`; `status` replaces a separate `list`.
   Seed byte sizes from the calibration ledger; leave sha256 null until a fetch is verified (never invent).
4. `doctor.py` calls `corpus.py status` (pointer detection across `data/*.txt`; absent corpora informational).
5. HANDOVER step 6: state that the fixture needs byte-identical corpora and record the author's fetch flags.

Not a runtime knob and no checkpoint/`Header` change, so AGENTS sections 2-4 are unaffected.

Open question for the owner: the fineweb/cosmopedia fetch parameters behind `ppl_blend_v2`.
