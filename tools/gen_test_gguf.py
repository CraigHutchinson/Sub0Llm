#!/usr/bin/env python3
"""Generate a minimal valid GGUF file for testing the gguf_loader.

Creates a tiny Llama-style model (V=64, D=32, H=2, Hkv=1, L=2, d_ff=64)
with identity/near-zero weights so we can verify:
 - Parse without crash
 - Correct weight shapes
 - NLL is finite after loading

Usage:
  python3 tools/gen_test_gguf.py /tmp/test_model.gguf
"""

import struct
import math
import random
import sys
import os

# ── Model config ──────────────────────────────────────────────────────────────
V   = 64    # vocab
D   = 32    # embed_dim
H   = 2     # n_heads
Hkv = 1     # n_kv_heads (GQA: 1 KV head shared across 2 query heads)
L   = 2     # n_layers
dff = 64    # d_ff
Dh  = D // H   # head_dim = 16

random.seed(42)

def rand_f32(n):
    """Small random values (std~0.02) — typical Xavier init scale."""
    return [random.gauss(0, 0.02) for _ in range(n)]

def zeros(n):
    return [0.0] * n

def ones(n):
    return [1.0] * n

# ── GGUF binary helpers ───────────────────────────────────────────────────────

def pack_string(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b

def pack_kv_string(key: str, val: str) -> bytes:
    return pack_string(key) + struct.pack("<I", 8) + pack_string(val)

def pack_kv_u32(key: str, val: int) -> bytes:
    return pack_string(key) + struct.pack("<I", 4) + struct.pack("<I", val)

def pack_kv_f32(key: str, val: float) -> bytes:
    return pack_string(key) + struct.pack("<I", 6) + struct.pack("<f", val)

def pack_tensor_info(name: str, gguf_dims: list, ggml_type: int, offset: int) -> bytes:
    """gguf_dims: column-major (innermost first) — will be reversed by loader."""
    b  = pack_string(name)
    b += struct.pack("<I", len(gguf_dims))
    for d in gguf_dims:
        b += struct.pack("<q", d)   # int64
    b += struct.pack("<I", ggml_type)
    b += struct.pack("<Q", offset)
    return b

def pack_f32_data(floats: list) -> bytes:
    return struct.pack(f"<{len(floats)}f", *floats)

# ── Weight layout ─────────────────────────────────────────────────────────────
# GGUF stores dims in column-major order: dims[0] = innermost (fastest-varying).
# Our loader reverses them to get logical shape.
#
# token_embd.weight:  logical (V, D)  → GGUF dims [D, V]
# attn_q.weight:      logical (H*Dh, D) → GGUF dims [D, H*Dh]
# attn_output.weight: logical (D, H*Dh) → GGUF dims [H*Dh, D]
# attn_k.weight:      logical (Hkv*Dh, D) → GGUF dims [D, Hkv*Dh]
# attn_v.weight:      logical (Hkv*Dh, D) → GGUF dims [D, Hkv*Dh]
# ffn_gate.weight:    logical (d_ff, D) → GGUF dims [D, d_ff]
# ffn_up.weight:      logical (d_ff, D) → GGUF dims [D, d_ff]
# ffn_down.weight:    logical (D, d_ff) → GGUF dims [d_ff, D]
# norm weights:       logical (D,) → GGUF dims [D]

# Build a list of (name, gguf_dims, data_floats)
tensors = []

# token embedding: (V, D) — small random values
tok_emb = rand_f32(V * D)
tensors.append(("token_embd.weight", [D, V], tok_emb))

for l in range(L):
    prefix = f"blk.{l}"

    # norm1 — ones (RMSNorm weight init)
    tensors.append((f"{prefix}.attn_norm.weight", [D], ones(D)))

    # attn_q — (H*Dh, D) row-major → GGUF dims [D, H*Dh]
    tensors.append((f"{prefix}.attn_q.weight", [D, H*Dh], rand_f32(H*Dh*D)))

    # attn_output — (D, H*Dh) → GGUF dims [H*Dh, D]
    tensors.append((f"{prefix}.attn_output.weight", [H*Dh, D], rand_f32(D*H*Dh)))

    # attn_k — (Hkv*Dh, D) → GGUF dims [D, Hkv*Dh]
    tensors.append((f"{prefix}.attn_k.weight", [D, Hkv*Dh], rand_f32(Hkv*Dh*D)))

    # attn_v — same shape as k
    tensors.append((f"{prefix}.attn_v.weight", [D, Hkv*Dh], rand_f32(Hkv*Dh*D)))

    # norm2 — ones
    tensors.append((f"{prefix}.ffn_norm.weight", [D], ones(D)))

    # ffn_gate (d_ff, D) → GGUF dims [D, d_ff]
    tensors.append((f"{prefix}.ffn_gate.weight", [D, dff], rand_f32(dff*D)))

    # ffn_up (d_ff, D) → GGUF dims [D, d_ff]
    tensors.append((f"{prefix}.ffn_up.weight", [D, dff], rand_f32(dff*D)))

    # ffn_down (D, d_ff) → GGUF dims [d_ff, D]
    tensors.append((f"{prefix}.ffn_down.weight", [dff, D], rand_f32(D*dff)))

# final norm — ones
tensors.append(("output_norm.weight", [D], ones(D)))

# Qwen2-style: separate output.weight (untied lm_head) — different from token_embd
# Use small random values (non-zero) so has_separate_lm_head branch is exercised.
qwen2_mode = "--qwen2" in sys.argv
# Qwen3-style: explicit head_dim (key/value_length) that differs from D/H.
# We keep D=32, H=2, but set head_dim=8 (normally D/H=16).
# This exercises the cfg.head_dim path in the loader.
qwen3_mode = "--qwen3" in sys.argv
if qwen3_mode:
    Dh = 8   # override: explicit head_dim != D/H
if qwen2_mode:
    tensors.append(("output.weight", [D, V], rand_f32(V * D)))
    # Q/K biases (Qwen2-specific) — loader should skip these cleanly
    for lq in range(L):
        tensors.append((f"blk.{lq}.attn_q.bias", [H*Dh], zeros(H*Dh)))
        tensors.append((f"blk.{lq}.attn_k.bias", [Hkv*Dh], zeros(Hkv*Dh)))

# ── Metadata ──────────────────────────────────────────────────────────────────
arch = "qwen3" if qwen3_mode else ("qwen2" if qwen2_mode else "llama")
rope_base = 1000000.0 if (qwen2_mode or qwen3_mode) else 10000.0
metadata  = pack_kv_string("general.architecture", arch)
metadata += pack_kv_string("tokenizer.ggml.model", arch)
metadata += pack_kv_u32(f"{arch}.embedding_length", D)
metadata += pack_kv_u32(f"{arch}.block_count", L)
metadata += pack_kv_u32(f"{arch}.attention.head_count", H)
metadata += pack_kv_u32(f"{arch}.attention.head_count_kv", Hkv)
metadata += pack_kv_u32(f"{arch}.feed_forward_length", dff)
metadata += pack_kv_u32(f"{arch}.context_length", 512)
metadata += pack_kv_u32(f"{arch}.vocab_size", V)
metadata += pack_kv_f32(f"{arch}.rope.freq_base", rope_base)
metadata_kv_count = 10
if qwen3_mode:
    # key_length / value_length declare explicit head_dim to the loader
    metadata += pack_kv_u32(f"{arch}.attention.key_length", Dh)
    metadata += pack_kv_u32(f"{arch}.attention.value_length", Dh)
    metadata_kv_count += 2

# ── Tensor data offsets ───────────────────────────────────────────────────────
# We need to know the offset of each tensor in the data section.
# All tensors are F32 (type 0), offset is in bytes from data section start.
tensor_offsets = []
cur_offset = 0
for name, gguf_dims, data in tensors:
    tensor_offsets.append(cur_offset)
    numel = 1
    for d in gguf_dims:
        numel *= d
    cur_offset += numel * 4   # 4 bytes per float32

# ── Tensor infos section ──────────────────────────────────────────────────────
tensor_infos_data = b""
for (name, gguf_dims, data), offset in zip(tensors, tensor_offsets):
    tensor_infos_data += pack_tensor_info(name, gguf_dims, 0, offset)  # type 0 = F32

# ── Header ────────────────────────────────────────────────────────────────────
MAGIC   = struct.pack("<I", 0x46554747)   # "GGUF"
VERSION = struct.pack("<I", 3)
header  = MAGIC + VERSION
header += struct.pack("<Q", len(tensors))          # tensor_count
header += struct.pack("<Q", metadata_kv_count)     # metadata_kv_count
header += metadata
header += tensor_infos_data

# Data section must start at 32-byte aligned offset after header.
header_len = len(header)
padding_needed = (32 - (header_len % 32)) % 32
padding = b"\x00" * padding_needed

# ── Build final file ──────────────────────────────────────────────────────────
output = header + padding
for name, gguf_dims, data in tensors:
    output += pack_f32_data(data)

# ── Write ─────────────────────────────────────────────────────────────────────
# Last positional arg (not a flag) is the output path.
positional_args = [a for a in sys.argv[1:] if not a.startswith("--")]
out_path = positional_args[0] if positional_args else "/tmp/test_model.gguf"
with open(out_path, "wb") as f:
    f.write(output)

print(f"Written {len(output):,} bytes to {out_path}")
print(f"Config: V={V}, D={D}, H={H}, Hkv={Hkv}, L={L}, d_ff={dff}, Dh={Dh}")
print(f"Tensors: {len(tensors)}")
# per_block = 1(norm1) + H*2(W_Q,W_O) + Hkv*2(W_K,W_V) + 1(norm2) + 6(gate.W/b, up.W/b, down.W/b)
print(f"Expected n_params: 1 + {L}*(1 + {H}*2 + {Hkv}*2 + 1 + 6) + 1 = {1 + L*(1 + H*2 + Hkv*2 + 1 + 6) + 1}")
