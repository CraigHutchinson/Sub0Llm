---
name: structured-token-space-embedding-diffusion
description: "Concept — give token IDs a metric/geometry (proximity = similarity) so misspellings resolve via the masking mechanism; = continuous embedding-space diffusion (Diffusion-LM), vs our discrete categorical denoiser"
metadata: 
  node_type: memory
  type: project
  originSessionId: f6f73404-ddcc-4bc1-b640-2b89a8b5c475
---

Concept (raised 2026-06-23, user). In our DISCRETE diffusion engine a token id is an arbitrary integer and
the output is a categorical softmax over those arbitrary classes — there's no native notion that id 5248
("Billy's") is "near" id 8066 ("Billy’s"). A misspelled input word is effectively a corrupted/'masked'
position the denoiser *could* predict, but it'll predict the corpus-likely word, not necessarily the user's
intended one.

**The idea:** give the token-ID space a DIMENSIONALITY/geometry so proximity encodes closeness (orthographic
+ contextual). Then a misspelling is a small displacement in token space and can be resolved to the nearest
valid token through the natural masking/denoising mechanism — "an increment on the token space."

**This is continuous / embedding-space diffusion** (Diffusion-LM, Li et al. 2022; later embedding-diffusion
LMs): diffuse in ℝ^D over word-embedding vectors, denoise, ROUND to the nearest token. The metric is native,
so typos/OOV land near the right token and context disambiguates — exactly the user's framing. Contrast with
our current masked/categorical (discrete-state) diffusion which has no inter-token metric in the output.

**Tension / why not free:** (a) it's a different denoiser family (continuous-state) than our discrete masked
one — a real architecture fork, not a tweak; (b) rounding/clamping dynamics and the embedding-norm geometry
are finicky; (c) collides with our weight-tied head story (see [[ch32-char-composition-oov-falsified]]:
spelling geom ≠ prediction geom). Cheaper partial step in the DISCRETE engine: order/structure the vocab so
id-adjacency ≈ similarity, or add an auxiliary embedding-proximity term — but the clean form is full
embedding-space diffusion.

**Partition (user's instruction):** spell-correction-through-context is SEPARATE work from corpus tokenizing.
Normalising the training corpus (truecasing/quote-norm, see [[tokenizer-normalization-and-factored-tokens]])
is one workstream; resolving novel/misspelled words in USER PROMPTS (incl. unseen names via the planned
3-layer char-composition path) is another. Keep them apart. Relates to [[gist-is-compute-context-primitive]].
