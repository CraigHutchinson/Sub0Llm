# Reflection spike — can P2996 delete registry.hpp's serialization boilerplate?

**Status: COMPILED AND RUN, clean, first attempt.** Validated 2026-07-28 on Compiler Explorer's
`clang_bb_p2996` ("x86-64 clang (reflection - C++26)", the Bloomberg fork) with
`-std=c++26 -freflection-latest`: compile exit 0, zero diagnostics, program exit 0, both
`static_assert`s on field count satisfied, and the expected JSON on stdout for BOTH structs.

Throwaway, per `AGENTS.md` §11. Not built by CMake, not on any include path, nothing in `src/` or
`include/` depends on it. It either gets merged into the mainline or deleted.

## The question

`include/sub0/registry.hpp` declares each `RunConfig` field once in an X-macro and expands it three
ways (members, writer, reader). That macro is *good* — it is the single source of truth that kept
`config.json` from falling behind on `n_kv_heads` / LoopSplit / rope the way the old `meta.txt` did.
So this is **not** a correctness fix. Three narrower questions:

1. Does reflection remove the macro without losing single-source-of-truth?
2. Is the result more readable, or merely different?
3. **Does it also subsume the field list the X-macro never covered** — `write_state`/`read_state`'s
   9 fields, hand-listed in two places with no compiler check that they agree?

(3) is the real prize. `config.json` is already safe. `state.json` is not, and it is the same shape
of bug this repo already ate once.

## Why it cannot be compiled here

Measured on this machine, not recalled:

| | Result |
|---|---|
| `clang++ -freflection` (clang 22.1.6) | `error: unknown argument` |
| `#include <meta>` | `'meta' file not found`; no `<meta>` on the system |
| `^^S` at `-std=c++26` | parses as **Objective-C blocks**, not the reflection operator |

Upstream Clang has not integrated P2996 (adopted into C++26 at Sofia 2025). GCC trunk has most of
it. MSVC has none. Bloomberg's `clang-p2996` fork is the most complete implementation and its own
README says: **"DO NOT use this project to build any artifacts destined for production."**

**The GCC path is closed for this project**: `nvcc`'s host compiler on Windows must be MSVC or
clang-cl, so adopting GCC would mean moving CUDA builds to Linux/WSL — far more work than the
boilerplate it saves.

## How to validate it

Cheapest first.

1. **Compiler Explorer** — godbolt.org carries the `clang-p2996` fork. Paste
   `run_config_reflect.cpp`, select it, and set `-std=c++26 -freflection-latest`. Minutes, no build.
   `-freflection-latest` (not plain `-freflection`) is required: this file uses expansion statements
   (`template for`, P1306) and `define_static_array` (P3491), which sit behind that flag.
2. **Build the fork** — clone `bloomberg/clang-p2996`, branch `p2996`, build clang only. Hours and
   several GB. Only worth it if step 1 shows the design is right and a larger port is planned.

## Validated status

Compiled and executed on `clang_bb_p2996` via the Compiler Explorer API. **No fixes were needed** --
all three things this section previously flagged as uncertain turned out fine:

- `define_static_array` around `nonstatic_data_members_of(...)` works and is what makes the range
  usable in `template for`.
- `access_context::current()` at namespace scope inside a template gives the access needed for a
  caller-supplied `T`.
- `identifier_of`'s `string_view` streams and compares as written.

So all three questions are answered:

- **Q1 (does it keep single-source-of-truth?)** Yes. The struct is the only declaration; writer and
  reader derive from it. Strictly stronger than the X-macro, which needs a macro row per field.
- **Q2 (more readable, or just different?)** More readable. `to_json` is ~12 lines and reads as a
  loop over fields; the X-macro version is a 24-row list plus three expansion macros.
- **Q3 (does it subsume the OTHER field list?)** Yes, and this was the prize. `RunState` gets the
  same two functions at **zero marginal cost** -- no new code at all, which the run output shows.

### An accidental confirmation of a real design decision

The spike prints `arch_id` as `13804672438013794288` because it deliberately omits the name-keyed
hex special case the production writer has. That value is `0xbf940c712c7017f0` -- **larger than
2^53**. It is exactly the case `write_state`'s comment warns about: a JSON reader that lands integers
in a double would silently corrupt it, and `arch_id` is the ONE thing `compatible()` consults. The
spike demonstrating the failure mode by omission is a good argument for keeping the hex-string
encoding when this is eventually ported.

## What it would replace, measured

- 24-row X-macro list + 3 expansion macros (`_DECL`, `_WRITE`, `_READ`)
- `enum_name`/`enum_parse` string-keyed tables (P3394 annotations are the reflection-native answer)
- `write_state`/`read_state`'s two hand-written 9-field lists — **not covered by the X-macro today**

Net: two generic functions (`to_json`, `from_json_field`) plus one `emit`/`parse_scalar` overload per
*type* rather than per *field*, serving both structs and any future one at zero marginal cost.

## Recommendation before spending real time

The X-macro already delivers the correctness property for `config.json`. The strongest near-term
argument for reflection is **(3)** — and that can be had *today, without any toolchain change*, by
folding `RunState` into its own X-macro the way `RunConfig` already is. If the goal is to stop
`state.json` drifting, that is a same-day change on the current compiler.

Reflection is then a readability and queryability win (`field_count` as a `static_assert`,
annotations replacing name-keyed enum tables) rather than a correctness one — worth doing when
upstream Clang ships P2996, not worth building an experimental LLVM fork for.
