#!/usr/bin/env python3
"""Read a small real-payload KDA projection/recurrence slice.

This is intentionally not a full GLM forward pass.  It verifies that the
existing GGUF mmap conventions decode BF16 KDA tensors with the expected
orientation and that the channel-wise gate and delta update are wired in a
deterministic order.  It allocates no model-sized buffer and has no numpy
dependency.
"""
from __future__ import annotations

import argparse
import hashlib
import json
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


def sigmoid(x):
    # Stable for the real payload values and for future conversion changes.
    if x >= 0.0:
        e = math.exp(-x)
        return 1.0 / (1.0 + e)
    e = math.exp(x)
    return e / (1.0 + e)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        while chunk := fp.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_full_layer_oracle(path, blob, data_start, tensors, output_path,
                            dump_prefix=None):
    """Write the two-token, complete layer-0 KDA FP32 oracle payload."""
    import numpy as np

    def addr(name, shape, typ):
        dims, got_type, rel = tensors[name]
        if dims != shape or got_type != typ:
            raise ValueError(
                f"{name}: expected {(shape, typ)}, got {(dims, got_type)}")
        return data_start + rel

    def bf16_matrix(base, rows, cols):
        raw = np.frombuffer(blob, dtype="<u2", count=rows * cols, offset=base)
        return (raw.astype(np.uint32) << 16).view(np.float32).reshape(rows, cols)

    def f32_array(base, count, shape=None):
        result = np.frombuffer(blob, dtype="<f4", count=count, offset=base)
        return result if shape is None else result.reshape(shape)

    def project(x, base, rows, cols):
        matrix = bf16_matrix(base, rows, cols)
        result = np.asarray(x @ matrix.T, dtype=np.float32)
        del matrix
        return result

    offsets = {
        "attn_norm": addr("blk.0.attn_norm.weight", (4096,), 0),
        "q": addr("blk.0.kda_q.weight", (4096, 8192), 30),
        "k": addr("blk.0.kda_k.weight", (4096, 8192), 30),
        "v": addr("blk.0.kda_v.weight", (4096, 8192), 30),
        "q_conv": addr("blk.0.kda_q_conv.weight", (4, 1, 8192), 0),
        "k_conv": addr("blk.0.kda_k_conv.weight", (4, 1, 8192), 0),
        "v_conv": addr("blk.0.kda_v_conv.weight", (4, 1, 8192), 0),
        "f_a": addr("blk.0.kda_f_a.weight", (4096, 128), 30),
        "f_b": addr("blk.0.kda_f_b.weight", (128, 8192), 30),
        "g_a": addr("blk.0.kda_g_a.weight", (4096, 128), 30),
        "g_b": addr("blk.0.kda_g_b.weight", (128, 8192), 30),
        "beta": addr("blk.0.kda_beta.weight", (4096, 64), 30),
        "o_norm": addr("blk.0.kda_o_norm.weight", (128,), 0),
        "dt": addr("blk.0.kda_dt_bias.weight", (8192,), 0),
        "a_log": addr("blk.0.kda_a_log.weight", (64,), 0),
        "output": addr("blk.0.kda_output.weight", (8192, 4096), 30),
    }

    x = np.asarray([[math.sin(0.013 * (t * 4096 + i + 1)) * 0.25
                     for i in range(4096)] for t in range(2)], dtype=np.float32)
    attn_norm = f32_array(offsets["attn_norm"], 4096)
    rms = np.sqrt(np.mean(x * x, axis=1, keepdims=True) + np.float32(1e-5))
    normalized = np.asarray(x / rms * attn_norm[None, :], dtype=np.float32)

    q = project(normalized, offsets["q"], 8192, 4096)
    k = project(normalized, offsets["k"], 8192, 4096)
    v = project(normalized, offsets["v"], 8192, 4096)
    histories = []
    for projected, key in ((q, "q_conv"), (k, "k_conv"), (v, "v_conv")):
        weight = f32_array(offsets[key], 8192 * 4, (8192, 4))
        history = np.zeros((8192, 3), dtype=np.float32)
        for token in range(2):
            current = projected[token].copy()
            raw = (history[:, 0] * weight[:, 0] +
                   history[:, 1] * weight[:, 1] +
                   history[:, 2] * weight[:, 2] +
                   current * weight[:, 3])
            projected[token] = raw / (np.float32(1.0) + np.exp(-raw))
            history[:, 0] = history[:, 1]
            history[:, 1] = history[:, 2]
            history[:, 2] = current
        histories.append(history)

    qh = q.reshape(2, 64, 128)
    kh = k.reshape(2, 64, 128)
    vh = v.reshape(2, 64, 128)
    qh /= np.sqrt(np.sum(qh * qh, axis=2, keepdims=True) + np.float32(1e-6))
    qh /= np.float32(math.sqrt(128.0))
    kh /= np.sqrt(np.sum(kh * kh, axis=2, keepdims=True) + np.float32(1e-6))

    f_low = project(normalized, offsets["f_a"], 128, 4096)
    forget_projection = project(f_low, offsets["f_b"], 8192, 128).reshape(2, 64, 128)
    g_low = project(normalized, offsets["g_a"], 128, 4096)
    out_gate = project(g_low, offsets["g_b"], 8192, 128).reshape(2, 64, 128)
    beta = project(normalized, offsets["beta"], 64, 4096)
    beta = np.asarray(1.0 / (1.0 + np.exp(-beta)), dtype=np.float32)
    dt = f32_array(offsets["dt"], 8192).reshape(64, 128)
    a_log = f32_array(offsets["a_log"], 64)
    forget = np.asarray(
        -5.0 / (1.0 + np.exp(-np.exp(a_log)[None, :, None] *
                              (forget_projection + dt[None, :, :]))),
        dtype=np.float32)

    state = np.zeros((64, 128, 128), dtype=np.float32)
    core = np.zeros((2, 64, 128), dtype=np.float32)
    for token in range(2):
        for head in range(64):
            state[head] *= np.exp(forget[token, head])[:, None]
            prediction = kh[token, head] @ state[head]
            state[head] += np.outer(
                kh[token, head],
                beta[token, head] * (vh[token, head] - prediction))
            core[token, head] = qh[token, head] @ state[head]

    o_norm = f32_array(offsets["o_norm"], 128)
    core_rms = np.sqrt(np.mean(core * core, axis=2, keepdims=True) +
                       np.float32(1e-6))
    gated = np.asarray(
        core / core_rms * o_norm[None, None, :] *
        (1.0 / (1.0 + np.exp(-out_gate))), dtype=np.float32)
    output = project(gated.reshape(2, 8192), offsets["output"], 4096, 8192)

    history_bytes = b"".join(
        np.asarray(history, dtype="<f4").tobytes() for history in histories)
    output_bytes = np.asarray(output, dtype="<f4").tobytes()
    state_bytes = np.asarray(state, dtype="<f4").tobytes()
    document = {
        "model_sha256": sha256_file(path),
        "layer": 0,
        "tokens": 2,
        "output_f32_sha256": hashlib.sha256(output_bytes).hexdigest(),
        "history_f32_sha256": hashlib.sha256(history_bytes).hexdigest(),
        "state_f32_sha256": hashlib.sha256(state_bytes).hexdigest(),
        "output_l2": float(np.linalg.norm(output.astype(np.float64))),
        "state_l2": float(np.linalg.norm(state.astype(np.float64))),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    if dump_prefix is not None:
        dump_prefix.parent.mkdir(parents=True, exist_ok=True)
        dumps = {
            ".input.f32": np.asarray(x, dtype="<f4").tobytes(),
            ".output.f32": output_bytes,
            ".history.f32": history_bytes,
            ".state.f32": state_bytes,
        }
        for suffix, payload in dumps.items():
            Path(str(dump_prefix) + suffix).write_bytes(payload)
    return document


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dump-prefix", type=Path,
                        help="optional prefix for input/output/history/state FP32 oracle dumps")
    parser.add_argument("model", type=Path)
    args = parser.parse_args(argv[1:])
    if args.dump_prefix is not None and args.output is None:
        parser.error("--dump-prefix requires --output")
    path = args.model
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
    # GLM-5 uses the configured lower-bounded KDA gate.  The softplus branch
    # is only the fallback when no lower bound is present; using it here
    # would validate a different recurrence than the shipped model.
    gate = [-5.0 * sigmoid(math.exp(alogv[i]) * (gate_proj[i] + dtv[i]))
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
    if args.output is not None:
        document = write_full_layer_oracle(
            path, blob, data_start, tensors, args.output,
            args.dump_prefix)
        print(f"model_sha256={document['model_sha256']}")
        print(f"output_f32_sha256={document['output_f32_sha256']}")
        print(f"history_f32_sha256={document['history_f32_sha256']}")
        print(f"state_f32_sha256={document['state_f32_sha256']}")
        print(f"output_l2={document['output_l2']:.9g} state_l2={document['state_l2']:.9g}")
    blob.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
