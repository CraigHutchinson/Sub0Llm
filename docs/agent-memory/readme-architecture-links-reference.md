---
name: readme-architecture-links-reference
description: README's "Architecture options" table links each flag to a paper/blog explainer -- ishanjmukherjee.github.io blog is a good source for small-but-load-bearing architecture decisions
metadata:
  type: reference
  originSessionId: 2a6b481b-6211-474b-818a-995223d7d5ac
---

User pointed to [ishanjmukherjee.github.io/qk-norm](https://ishanjmukherjee.github.io/qk-norm) as a
good explainer and said the same blog covers other useful topics. Checked the blog's index: it also has
[why-not-tie-embeddings](https://ishanjmukherjee.github.io/why-not-tie-embeddings) (directly relevant to
this project's `--tie-embeddings`), plus posts on L1/L2 regularization and allreduce/collectives --
worth checking that index again if a future architecture decision needs a citation and the topic might
be covered there.

README's `## Configuration` > `### Architecture options` table (committed `8628e7a`) links each flag:
- `--tie-embeddings` -> the why-not-tie-embeddings post above
- `--qk-norm` -> the qk-norm post above
- `--gated-ffn` -> arXiv:2002.05202 (GLU Variants Improve Transformer, the SwiGLU paper)
- `--ternary` -> arXiv:2402.17764 (BitNet b1.58)
- `--pos-encoding` (RoPE) -> arXiv:2104.09864 (RoFormer)
- `--optimizer muon` -> kellerjordan.github.io/posts/muon/ (the canonical Muon writeup)
- `--attn-sinks` -> arXiv:2309.17453 (StreamingLLM / Efficient Streaming LMs with Attention Sinks)

All links were verified live (WebFetch/WebSearch) before adding, not pulled from training-data memory --
worth repeating that verification step if any of these ever need updating, since a stale/wrong citation
in a README is worse than none.
