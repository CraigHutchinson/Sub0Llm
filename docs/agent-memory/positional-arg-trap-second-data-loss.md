---
name: positional-arg-trap-second-data-loss
description: "The `train [model] [corpus]` positional trap destroyed a SECOND corpus (tinystories.txt) before the guard landed; git-lfs cache saved it"
metadata: 
  node_type: memory
  type: project
  originSessionId: 2a6b481b-6211-474b-818a-995223d7d5ac
  modified: 2026-07-28T13:29:21.205Z
---

`sub0llm-train`'s positional usage is `[model] [corpus]`, so ONE positional binds to *model*, and
`save_model` then overwrites whatever that path pointed at.

This has now destroyed two corpora:
1. `data/fineweb_smoke.txt` (1.07 GB) -- found and regenerated during the same session.
2. `data/tinystories.txt` -- overwritten 2026-07-28 10:27 by a 34,647,488-byte `model.bin`
   (magic `S0L5`, header D_MODEL=196 / N_LAYERS=11 / N_HEADS=7, i.e. the `d196check` config).
   The guard commit `d6b7070` landed at 10:53 -- **26 minutes after** the loss, so the guard did not
   fail, it did not yet exist. It has since been observed firing correctly.

**How it was found**: NOT by noticing the file. `sub0llm-configure` was run against it and reported
`alpha words (total / unique): 4246349 / 2597114`. 2.6M unique words out of 4.2M is the signature of
random binary -- real English prose of that size has ~30-50K unique words. **A vocabulary whose
unique-word ratio approaches 1.0 means the input is not text.** That ratio is worth treating as a
corruption check in the configurator itself, not just something to notice by eye.

**Recovery**: `data/tinystories.txt` is gitignored but WAS tracked via git-lfs in an old commit
(`f2f09d0`), and the object was still in the local cache:
`.git/lfs/objects/1e/13/1e13dedf...` -- 22,133,335 bytes, sha256 verified, real prose. Restored.
So: **when a gitignored data file is lost, check `git log --all -- <path>` and `.git/lfs/objects/`
before concluding it is unrecoverable.** A git-lfs pointer blob is only 133 bytes and easy to miss.

**FIXED at the shape level, 2026-07-28 (`6006a1f`)**: no path argument is positional anywhere in the
CLI any more. `train` takes `--model`/`--corpus` and has NO positionals, so the old form is a parse
error. The only positional left in the whole CLI is `gen`'s prompt -- free text, not a path, so it
cannot be mistaken for a file or written to. **Rule: paths are always named.** The
`--model`-looks-like-a-corpus check stays as a second line of defence, since `--model <a corpus>` is
still typeable and a miss costs a corpus.

The restored copy had ZERO `<|endoftext|>` markers, so it was re-fetched from the canonical
`TinyStoriesV2-GPT4-train` split (2.2 GB, 2,717,698 stories). Related:
[[large-model-windows-build-run-gotchas]].
