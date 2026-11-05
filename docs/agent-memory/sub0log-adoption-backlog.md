---
name: sub0log-adoption-backlog
description: "BACKLOG: adopt Sub0Log (github.com/CraigHutchinson/Sub0Log, pinned SHA b992f1a36e4579cd07f6401fb0ee7f866f6fc335) into Sub0Llm -- a structured, typed, crash-survivable C++23 diagnostic logging framework, currently design-only (REQUIREMENTS.md + README, no implementation yet)"
metadata: 
  node_type: memory
  type: project
  originSessionId: 84b5aaba-5d15-4c11-acba-06624e60ae97
  modified: 2026-09-01T09:22:08.742Z
---

**Status**: not started, not scheduled -- a backlog candidate flagged by the user 2026-09-01 while
Sub0Firn's spec work was in progress. Sub0Log is its own separate repo/project (like Sub0Firn), not
part of Sub0Llm, and has its own early-stage design (requirements + README, no implementation landed at
the pinned SHA).

## What Sub0Log actually is (real requirements, quoted from `REQUIREMENTS.md` at the pinned SHA)

A structured diagnostic logging framework for C++23: typed records (not pre-formatted text) written
directly into a file-backed memory mapping the kernel owns, so committed records survive a hard process
kill. Nine numbered requirements, each with a verbatim normative sentence:

1. **The producer does no work it can defer** -- "A call site must not format. No `std::format`, no
   `to_string`, no stream insertion on the calling thread." No heap allocation, no lock acquisition, no
   argument evaluation at all when logging is off.
2. **Records reach the consumer typed** -- "A consumer receives argument values in their original
   types. An integer arrives as an integer, not as characters that must be parsed back." Enables
   filtering/field-level assertions without text parsing.
3. **The stream survives an abrupt exit** -- "A record that has been committed must survive
   `TerminateProcess`/`SIGKILL` of the producing process. Records are written into memory the kernel
   owns, not into a staging buffer the process owns." Readers tolerate truncation, generation markers
   handle recycled storage.
4. **Safe to link into a plugin** -- a dynamically-loaded plugin can emit into its host's stream without
   linking the library's implementation (C ABI, no duplicated singletons, descriptors embedded not
   resolved at read time).
5. **Several processes, one stream** -- each process writes its own segment, no shared writable
   structure/cross-process lock; decoders merge segments; correlation IDs propagate through `exec`.
6. **Correlation is a field** -- an RAII-scope-set identifier attaches automatically to every record
   emitted within it, queryable as a field (not a timestamp heuristic).
7. **Deterministic in tests** -- a test can bind its own instance for a scope (the "established
   scoped-active pattern") and drain it synchronously, no production-code test branches.
8. **Portability** -- "Windows/MSVC, Linux and macOS with Clang and GCC, all first-class." C++23, no
   third-party dependencies in producer paths.
9. **Observability of the mechanism itself** -- "Dropped records are counted and the count is
   retrievable. A drop is never silent."

Core mechanism (README): "a file-backed mapping with one atomic" -- a producer claims a chunk with one
atomic op and copies its arguments in, so a record is in kernel-owned memory the instant the call
returns; no staging queue to lose on crash.

## Why this is relevant to Sub0Llm specifically

Sub0Llm runs multi-day training jobs where losing in-progress state/diagnostics to a crash is exactly
the kind of loss this project already treats as high-stakes (see `AGENTS.md` §3's checkpoint-format
blast-radius rule -- the same underlying concern: don't lose real, expensive, in-progress work). Current
logging is ad hoc (`sub0::log::warn`/etc., plain text). Sub0Log's typed-record + crash-survivable design
would give:
- Post-mortem diagnostics after a hard crash/kill mid-training-run (today: whatever made it to a text
  log file before the crash, nothing more).
- Typed, filterable records instead of parsing formatted text back apart (relevant to anything that
  currently greps/parses this project's own log output, e.g. throughput/tok-s reporting).
- A no-heap-allocation, no-formatting producer path -- directly compatible with `AGENTS.md` §1's own
  no-heap-in-hot-path rule; logging from inside a hot training-step path today would violate that rule if
  it did any formatting, so Sub0Log's design is a genuine fit, not just a nice-to-have.

## Adoption shape, not yet decided

Same integration-seam question as [[pivot-to-existing-model-support-qwen4-preview]]'s Sub0Firn spin-off:
vendored header-only client vs. linked library. Sub0Log is design-only at the pinned SHA (no
implementation to actually vendor yet) -- real adoption work can't start until Sub0Log itself has a
working implementation. No decision made on priority relative to the Qwen4-preview mechanism work
(GDN/n-gram/QSA) or Sub0Firn -- purely a backlog flag for now.
