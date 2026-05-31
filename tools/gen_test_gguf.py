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

# ── Metadata ──────────────────────────────────────────────────────────────────
metadata  = pack_kv_string("general.architecture", "llama")
metadata += pack_kv_string("tokenizer.ggml.model", "llama")
metadata += pack_kv_u32("llama.embedding_length", D)
metadata += pack_kv_u32("llama.block_count", L)
metadata += pack_kv_u32("llama.attention.head_count", H)
metadata += pack_kv_u32("llama.attention.head_count_kv", Hkv)
metadata += pack_kv_u32("llama.feed_forward_length", dff)
metadata += pack_kv_u32("llama.context_length", 512)
metadata += pack_kv_u32("llama.vocab_size", V)
metadata += pack_kv_f32("llama.rope.freq_base", 10000.0)
metadata_kv_count = 10

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
out_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/test_model.gguf"
with open(out_path, "wb") as f:
    f.write(output)

print(f"Written {len(output):,} bytes to {out_path}")
print(f"Config: V={V}, D={D}, H={H}, Hkv={Hkv}, L={L}, d_ff={dff}, Dh={Dh}")
print(f"Tensors: {len(tensors)}")
# per_block = 1(norm1) + H*2(W_Q,W_O) + Hkv*2(W_K,W_V) + 1(norm2) + 6(gate.W/b, up.W/b, down.W/b)
print(f"Expected n_params: 1 + {L}*(1 + {H}*2 + {Hkv}*2 + 1 + 6) + 1 = {1 + L*(1 + H*2 + Hkv*2 + 1 + 6) + 1}")
