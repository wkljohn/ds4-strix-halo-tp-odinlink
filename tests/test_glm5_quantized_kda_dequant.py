#!/usr/bin/env python3
"""Cross-check the quantized KDA block decoder against llama.cpp semantics.

The reference loops below are an independent transcription of
`dequantize_row_q4_K` and `dequantize_row_q8_0` from llama.cpp
`ggml/src/ggml-quants.c`.  They intentionally do not call DS4 dequant helpers.
"""
from __future__ import annotations

import importlib.util
import mmap
import os
import struct
from pathlib import Path


def load_probe():
    source = Path(__file__).parents[1] / "scripts/probe-glm5-next-kda-payload.py"
    spec = importlib.util.spec_from_file_location("glm5_kda_payload", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def scale_min(group, scales):
    if group < 4:
        return scales[group] & 63, scales[group + 4] & 63
    return ((scales[group + 4] & 15) | ((scales[group - 4] >> 6) << 4),
            (scales[group + 4] >> 4) | ((scales[group] >> 6) << 4))


def llama_q4_k(blob, offset):
    d, dmin = struct.unpack_from("<ee", blob, offset)
    scales = blob[offset + 4:offset + 16]
    q = blob[offset + 16:offset + 144]
    out = []
    q_offset = 0
    group = 0
    for _ in range(0, 256, 64):
        sc0, min0 = scale_min(group, scales)
        sc1, min1 = scale_min(group + 1, scales)
        d0, m0 = d * sc0, dmin * min0
        d1, m1 = d * sc1, dmin * min1
        out.extend(d0 * (q[q_offset + i] & 15) - m0 for i in range(32))
        out.extend(d1 * (q[q_offset + i] >> 4) - m1 for i in range(32))
        q_offset += 32
        group += 2
    return out


def llama_q8_0(blob, offset):
    d = struct.unpack_from("<e", blob, offset)[0]
    return [d * struct.unpack_from("<b", blob, offset + 2 + i)[0]
            for i in range(32)]


def fnv64(values):
    h = 0xCBF29CE484222325
    for value in values:
        for byte in struct.pack("<f", value):
            h = ((h ^ byte) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def main():
    model_env = os.environ.get("DS4_GLM5_MODEL")
    if not model_env:
        raise SystemExit("DS4_GLM5_MODEL is required")
    model = Path(model_env)
    probe = load_probe()
    data_start, tensors = probe.load_directory(model)
    q_dims, q_type, q_rel = tensors["blk.0.kda_q.weight"]
    v_dims, v_type, v_rel = tensors["blk.0.kda_v.weight"]
    with model.open("rb") as fp:
        blob = mmap.mmap(fp.fileno(), 0, access=mmap.ACCESS_READ)
        checked = []
        if q_type == 12:
            for block in range(4):
                offset = data_start + q_rel + block * 144
                got = probe.q4_k_block(blob, offset)
                want = llama_q4_k(blob, offset)
                if got != want:
                    raise AssertionError(f"Q4_K KDA block {block} mismatch")
                checked.extend(got)
        elif q_type != 30:
            raise AssertionError(f"unexpected KDA q type {q_type}")
        if v_type == 8:
            for block in range(4):
                offset = data_start + v_rel + block * 34
                got = probe.q8_0_block(blob, offset)
                want = llama_q8_0(blob, offset)
                if got != want:
                    raise AssertionError(f"Q8_0 KDA block {block} mismatch")
                checked.extend(got)
        elif v_type != 30:
            raise AssertionError(f"unexpected KDA v type {v_type}")
        blob.close()
    print("PASS GLM5 quantized KDA dequant cross-check "
          f"q_shape={q_dims} q_type={q_type} v_shape={v_dims} v_type={v_type} "
          f"sample_fnv64={fnv64(checked):016x}")


if __name__ == "__main__":
    main()
