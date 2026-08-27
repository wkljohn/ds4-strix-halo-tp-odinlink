#!/usr/bin/env python3
"""Run a two-token, one-layer GLM-5 KDA projection smoke on real GGUF data.

This is a layout/orientation gate, not a model implementation.  It maps the
BF16 tensors read-only, computes q/k/v, the low-rank gate and beta projections,
executes the 64-head channel-wise recurrence, and checks the result against a
second scalar recurrence.  No converted or persistent weight buffer is kept.
"""
from __future__ import annotations
import hashlib
import importlib.util
import math
import mmap
import struct
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("payload", HERE / "probe-glm5-next-kda-payload.py")
payload = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(payload)


def bf16_matrix(blob, base, rows, cols):
    raw = np.frombuffer(blob, dtype="<u2", count=rows * cols, offset=base)
    # GGUF stores the output rows contiguously for these tensors.
    return (raw.astype(np.uint32) << 16).view(np.float32).reshape(rows, cols)


def f32(blob, base, n):
    return np.frombuffer(blob, dtype="<f4", count=n, offset=base)


def main(argv):
    if len(argv) != 2:
        print(f"usage: {argv[0]} MODEL.gguf", file=sys.stderr)
        return 2
    path = Path(argv[1])
    data_start, tensors = payload.load_directory(path)
    with path.open("rb") as fp:
        blob = mmap.mmap(fp.fileno(), 0, access=mmap.ACCESS_READ)

    def addr(name, shape, typ):
        dims, got, rel = tensors[name]
        if dims != shape or got != typ:
            raise ValueError(f"{name}: expected {(shape, typ)}, got {(dims, got)}")
        return data_start + rel

    q_a = addr("blk.0.kda_q.weight", (4096, 8192), 30)
    k_a = addr("blk.0.kda_k.weight", (4096, 8192), 30)
    v_a = addr("blk.0.kda_v.weight", (4096, 8192), 30)
    f_a = addr("blk.0.kda_f_a.weight", (4096, 128), 30)
    f_b = addr("blk.0.kda_f_b.weight", (128, 8192), 30)
    beta_a = addr("blk.0.kda_beta.weight", (4096, 64), 30)
    dt = addr("blk.0.kda_dt_bias.weight", (8192,), 0)
    alog = addr("blk.0.kda_a_log.weight", (64,), 0)

    # Keep only two token vectors and one matrix at a time in converted memory.
    x = np.asarray([[math.sin(0.013 * (t * 4096 + i + 1)) * 0.25
                     for i in range(4096)] for t in range(2)], dtype=np.float32)
    def project(base, rows, cols):
        return x @ bf16_matrix(blob, base, rows, cols).T
    q = project(q_a, 8192, 4096)
    k = project(k_a, 8192, 4096)
    v = project(v_a, 8192, 4096)
    low = project(f_a, 128, 4096)
    f = low @ bf16_matrix(blob, f_b, 8192, 128).T
    beta = project(beta_a, 64, 4096)
    dtv = f32(blob, dt, 8192)
    alogv = f32(blob, alog, 64)

    # Reference recurrence, state shape [head,key,value].
    state = np.zeros((64, 128, 128), dtype=np.float32)
    out = np.zeros((2, 8192), dtype=np.float32)
    for t in range(2):
        for h in range(64):
            sl = slice(h * 128, (h + 1) * 128)
            qh, kh, vh = q[t, sl], k[t, sl], v[t, sl]
            g = -5.0 / (1.0 + np.exp(-np.exp(alogv[h]) *
                                      (f[t, sl] + dtv[sl])))
            b = 1.0 / (1.0 + np.exp(-beta[t, h]))
            pred = kh @ state[h]
            state[h] = np.exp(g)[:, None] * state[h] + np.outer(kh, b * (vh - pred))
            out[t, sl] = qh @ state[h]

    # Scalar re-evaluation catches accidental matrix-axis transposition.
    scalar = np.zeros_like(state)
    scalar_out = np.zeros_like(out)
    for t in range(2):
        for h in range(64):
            sl = slice(h * 128, (h + 1) * 128)
            for j in range(128):
                pred = sum(float(k[t, h * 128 + i]) * float(scalar[h, i, j])
                           for i in range(128))
                for i in range(128):
                    g = -5.0 / (1.0 + math.exp(-math.exp(float(alogv[h])) *
                              (float(f[t, h * 128 + i]) + float(dtv[h * 128 + i]))))
                    b = 1.0 / (1.0 + math.exp(-float(beta[t, h])))
                    scalar[h, i, j] = math.exp(g) * scalar[h, i, j] + \
                        float(k[t, h * 128 + i]) * b * (float(v[t, h * 128 + j]) - pred)
                scalar_out[t, h * 128 + j] = sum(float(q[t, h * 128 + i]) * float(scalar[h, i, j])
                                                   for i in range(128))
    err = float(np.max(np.abs(out - scalar_out)))
    digest = hashlib.sha256(out.astype("<f4").tobytes()).hexdigest()[:16]
    if not np.isfinite(out).all() or err > 2e-4:
        raise SystemExit(f"FAIL GLM5 KDA projection err={err:.9g}")
    print("PASS GLM5-next KDA BF16 projection/recurrence")
    print(f"tokens=2 heads=64 head_dim=128 max_err={err:.9g} output_sha256={digest}")
    # Release every ndarray view before closing the mmap export.
    del q, k, v, low, f, beta, dtv, alogv, x, state, out, scalar, scalar_out
    blob.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
