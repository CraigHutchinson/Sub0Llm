---
name: defaults-are-the-best-options
description: User direction 2026-09-29 — once an option passes its gates and is recommended, promote it to an AUTO default; never leave a "recommended flags" list users must remember
metadata:
  type: feedback
---
"We should make the defaults the best options." Default-off applies only WHILE an option is being evaluated
(AGENTS.md §4). Once it passes its gates (G-PPL etc.) and is recommended, promote it to an AUTO default in
`tools/configurator.cpp`'s `resolve_decode_defaults`: -1 (bool) or 0 (thread count) means auto, resolving ON
for the build class it was measured on (currently `--moe-quant-experts 1` = the real-model recipe) and to the
old value everywhere else. The neutral header must stay byte-identical, and an explicit flag always wins.

**Why:** a best build that needs seven remembered flags is not the default path, and A/Bs kept being run
against stale flag lists.
**How to apply:** when merging a recommended optimization, add it to `resolve_decode_defaults` in the same
change, verify the neutral header diff is empty, and make A/B arms name only the delta from the default
(`--backbone-act-super 0`). AGENTS.md §4 records this. Related: [[compile-time-decisions-preferred]], [[qwen4-real-axes-configure-recipe]].
