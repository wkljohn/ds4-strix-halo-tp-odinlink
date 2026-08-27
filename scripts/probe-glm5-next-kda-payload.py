#!/usr/bin/env python3
"""Read a small real-payload KDA projection/recurrence slice.

This is intentionally not a full GLM forward pass.  It verifies that the
existing GGUF mmap conventions decode BF16 KDA tensors with the expected
orientation and that the channel-wise gate and delta update are wired in a
deterministic order.  It allocates no model-sized buffer and has no numpy
dependency.
"""
from __future__ import annotations

import math
import mmap
import struct
import sys
from pathlib import Path


def read_string(fp):
    n = struct.unpack("<Q", fp.read(8))[0]
    b = fp.read(n)
    if len(b) != n:
        raise ValueError("truncated GGUF string")
    return b.decode("utf-8", "replace")


def skip_value(fp, typ):
    sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1,
             10: 8, 11: 8, 12: 8}
    if typ in sizes:
        fp.seek(sizes[typ], 1)
    elif typ == 8:
        fp.seek(struct.unpack("<Q", fp.read(8))[0], 1)
    elif typ == 9:
        elem = struct.unpack("<I", fp.read(4))[0]
        n = struct.unpack("<Q", fp.read(8))[0]
        for _ in range(n):
            skip_value(fp, elem)
    else:
        raise ValueError(f"unsupported metadata type {typ}")


def load_directory(path):
    with path.open("rb") as fp:
        magic, version, nt, nk = struct.unpack("<IIQQ", fp.read(24))
        if magic != 0x46554747 or version != 3:
            raise ValueError("expected GGUF v3")
        alignment = 32
        meta = {}
        for _ in range(nk):
            key = read_string(fp)
            typ = struct.unpack("<I", fp.read(4))[0]
            if typ == 4:
                value = struct.unpack("<I", fp.read(4))[0]
                if key == "general.alignment" and value:
                    alignment = value
                meta[key] = value
            elif typ == 8:
                meta[key] = read_string(fp)
            else:
                skip_value(fp, typ)
        tensors = {}
        for _ in range(nt):
            name = read_string(fp)
            nd = struct.unpack("<I", fp.read(4))[0]
            dims = tuple(struct.unpack("<Q", fp.read(8))[0] for _ in range(nd))
            typ, offset = struct.unpack("<IQ", fp.read(12))
            tensors[name] = (dims, typ, offset)
        data_start = (fp.tell() + alignment - 1) // alignment * alignment
        return data_start, tensors


def bf16_at(blob, offset):
    h = struct.unpack_from("<H", blob, offset)[0]
    return struct.unpack("<f", struct.pack("<I", h << 16))[0]


def row(blob, base, rows, cols, row_index, count):
    if row_index >= rows or count > cols:
        raise ValueError("sample row outside tensor")
    start = base + row_index * cols * 2
    return [bf16_at(blob, start + 2 * j) for j in range(count)]


def f32_vec(blob, base, count):
    return [struct.unpack_from("<f", blob, base + 4 * i)[0]
            for i in range(count)]


def softplus(x):
    # Keep the payload smoke finite even if a future conversion produces a
    # large positive gate preactivation.
    return x + math.log1p(math.exp(-x)) if x > 0.0 else math.log1p(math.exp(x))


def main(argv):
    if len(argv) != 2:
        print(f"usage: {argv[0]} MODEL.gguf", file=sys.stderr)
        return 2
    path = Path(argv[1])
    data_start, tensors = load_directory(path)
    required = [
        "blk.0.kda_q.weight", "blk.0.kda_k.weight", "blk.0.kda_v.weight",
        "blk.0.kda_f_a.weight", "blk.0.kda_f_b.weight",
        "blk.0.kda_beta.weight", "blk.0.kda_dt_bias.weight",
        "blk.0.kda_a_log.weight",
    ]
    if any(name not in tensors for name in required):
        raise ValueError("KDA payload sample tensors are incomplete")
    with path.open("rb") as fp:
        blob = mmap.mmap(fp.fileno(), 0, access=mmap.ACCESS_READ)
    def tensor(name, shape, typ):
        dims, got_type, rel = tensors[name]
        if dims != shape or got_type != typ:
            raise ValueError(f"{name}: expected {(shape, typ)}, got {(dims, got_type)}")
        return data_start + rel

    # GGUF dimensions are [input, output]; sample output rows explicitly.
    q = tensor("blk.0.kda_q.weight", (4096, 8192), 30)
    k = tensor("blk.0.kda_k.weight", (4096, 8192), 30)
    v = tensor("blk.0.kda_v.weight", (4096, 8192), 30)
    fa = tensor("blk.0.kda_f_a.weight", (4096, 128), 30)
    fb = tensor("blk.0.kda_f_b.weight", (128, 8192), 30)
    beta_w = tensor("blk.0.kda_beta.weight", (4096, 64), 30)
    dt = tensor("blk.0.kda_dt_bias.weight", (8192,), 0)
    alog = tensor("blk.0.kda_a_log.weight", (64,), 0)
    dtv = f32_vec(blob, dt, 8192)
    alogv = f32_vec(blob, alog, 64)

    x = [math.sin(0.017 * (i + 1)) * 0.5 for i in range(4096)]
    # Use eight channels/values.  The complete tensors are still bounds
    # checked, while this probe reads only a few rows and remains quick.
    qv = [sum(a * b for a, b in zip(row(blob, q, 8192, 4096, i, 4096), x))
          for i in range(8)]
    kv = [sum(a * b for a, b in zip(row(blob, k, 8192, 4096, i, 4096), x))
          for i in range(8)]
    vv = [sum(a * b for a, b in zip(row(blob, v, 8192, 4096, i, 4096), x))
          for i in range(8)]
    low = [sum(a * b for a, b in zip(row(blob, fa, 128, 4096, i, 4096), x))
           for i in range(128)]
    gate_proj = [sum(a * b for a, b in zip(row(blob, fb, 8192, 128, i, 128), low))
                 for i in range(8)]
    gate = [-math.exp(alogv[i]) * softplus(gate_proj[i] + dtv[i])
            for i in range(8)]
    # beta is a scalar per value head in this model; use its first row sample.
    beta_raw = sum(a * b for a, b in zip(row(blob, beta_w, 64, 4096, 0, 4096), x))
    beta = 1.0 / (1.0 + math.exp(-beta_raw))
    state = [[0.01 * (i + j + 1) for j in range(8)] for i in range(8)]
    for i in range(8):
        decay = math.exp(gate[i])
        for j in range(8):
            state[i][j] *= decay
    pred = [sum(kv[i] * state[i][j] for i in range(8)) for j in range(8)]
    for i in range(8):
        for j in range(8):
            state[i][j] += beta * kv[i] * (vv[j] - pred[j])
    out = [sum(qv[i] * state[i][j] for i in range(8)) for j in range(8)]
    if not all(math.isfinite(z) for z in out + gate):
        raise ValueError("non-finite KDA payload probe result")
    h = 0xCBF29CE484222325
    for z in out:
        for b in struct.pack("<f", z):
            h = ((h ^ b) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    print("PASS GLM5-next KDA payload probe")
    print(f"sample_channels=8 beta={beta:.8g} output_fnv64={h:016x}")
    blob.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
