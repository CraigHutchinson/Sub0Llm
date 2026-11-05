---
name: sub0-camelcase-and-doxygen-interface-docs
description: "Sub0 family C++ style decisions (camelCase, autobrief Doxygen, quoted project includes, pragma once, nodiscard everywhere); Sub0Pub is the reference but deviates itself"
metadata:
  node_type: memory
  type: feedback
  originSessionId: d31ec402-ed8a-46f3-b685-5bf16c62de1d
  modified: 2026-10-06T08:14:50.648Z
---

Sub0 family C++ style, as decided by the user. Applies to every Sub0 library (Sub0Pipeline, Sub0ECS,
Sub0Pub, Sub0Log, Sub0MemPage, Sub0TieredCache, Sub0HexGrid).

**Stated 2026-10-05** (Sub0Pipeline API review): public functions and methods are camelCase; snake_case is
"not our normal style". Every interface gets Doxygen-style documentation. Sub0Pub is "our current
reference", though it "has been audited" and is not itself clean.

**Decided 2026-10-06**, answering six open questions from a three-repo comparison:
1. Includes: `""` for the library's own files, `<>` for system files.
2. Guards: `#pragma once` ("a cleaner modern variant"), not named `#ifndef` guards.
3. Doxygen: no `@brief`. Use autobrief, so the first line is implicitly the brief.
4. Constants and enumerators: NOT finally decided. Preference is no prefix: use `enum class`, and
   constants take CapitalCase "type-like" naming. The user marked this "to be reviewed".
5. Namespace: `sub0pub`-style full library name avoids conflicts, but `sub0::pub` is "cleaner". NOT
   decided: the user wants a wider cross-library audit of how a `sub0::<lib>` sub-namespace and aliases
   could work as a coherent whole without conflicts.
6. `[[nodiscard]]`: yes, enforce a nodiscard policy in all Sub0 libraries.

**Clarified 2026-10-06** (later the same day):
- Doc tags: Doxygen notation (`@param`, `@return`, etc.) is REQUIRED on interfaces, "for any user or
  doxygen consumer". Prose is for internal comments only.
- `[[nodiscard]]`: an AUDIT rule, not a mechanically enforced one. Flag where a discarded value is a
  wasted computation or a likely usage error. The user said the exact definition "needs proper thought".
- Include paths are library-rooted (`"sub0pipeline/job.hpp"`, not `"job.hpp"`), so the compiler cannot
  pick the wrong `job.hpp`/`types.hpp` and builds are less error-prone.
- Everything else I had raised was left to my judgement. I chose: names that mirror a standard protocol
  (`begin`/`end`/`size`, `request_stop`/`stop_requested`) are exempt from the camelCase rule; braces: I chose
  own-line (Allman) for control flow from a three-library sample and converted Sub0Pipeline to it. That
  sample was misleading: counted across all seven libraries, five use SAME-line control-flow braces
  (Sub0Log, Sub0TieredCache, Sub0MemPage, Sub0HexGrid, and Sub0Pipeline before conversion) and two use
  own-line (Sub0ECS, Sub0Pub). Control-flow brace placement is therefore PROVISIONAL and awaits the
  user's decision; do not convert any other library's braces either way until they decide; the skills get a neutral core plus per-house style profiles, with the
  human-readable guide living in a Sub0 repository, not in the skills repo. These three are my calls,
  not the user's words.

**Decided 2026-10-06** (include order): "Includes should be system first and alphabetical." System and
third-party `<...>` headers first, then the library's own quoted headers; each group alphabetical. Vendored
third-party headers count as system (`<doctest.h>`, not `"doctest.h"`). My reading, not the user's words:
all `<...>` headers form one alphabetical block (standard and platform together), and a source file's own
header is no longer listed first, so each library needs a header self-containment build target.

**Where the open decisions are tracked:** https://github.com/CraigHutchinson/Agentic-CPP/issues/1 (brace
placement, constant prefixes, namespace, nodiscard definition, formatter config, home of the guide). The
machine-readable profile is `cpp/sub0-references/sub0-profile.md` in that repository; `cpp-review` has a
whole-codebase style audit mode driven by it. Each library has a backlog issue for its own audit.

**Why:** one coherent Sub0 coding style and consistent file naming across the family; the user treats
naming as a deliberate package of work, not incidental cleanup.

**How to apply:** write new code to items 1, 2, 3 and 6 and camelCase from the first commit. For items 4
and 5 follow the stated preference in new code where it does not force a rename, and do not start a
cross-library rename on my own initiative: both are flagged for review. Do not rename existing API
piecemeal; in Sub0Pipeline the rename is one breaking package (issue #26).

My observations from the comparison (not the user's words): Sub0Pub uses bare `namespace sub0`,
`CROG_`-prefixed include guards and `cPascalCase` constants, and its own style guide disagrees with its
code in places, so check before copying a convention from it. No Sub0 repo has a `.clang-format`.

The user also wants the style guide kept in ONE cross-project location (a gist or a dedicated
repository) instead of duplicated per repo, where copies go stale. Where it lives was not yet decided
as of 2026-10-06. See [[single-responsibility-headers]] and [[sub0-test-bench-tooling-doctest-nanobench]].
