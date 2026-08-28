#!/usr/bin/env python3
"""Prove GLM-5.3 sparse-MLA F32 norm payloads are widened BF16 values."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import mmap
import struct
from pathlib import Path


SPARSE_LAYERS = tuple(range(3, 44, 4)) + (45,)


def load_helpers():
    source = Path(__file__).with_name("probe-glm5-next-kda-payload.py")
    spec = importlib.util.spec_from_file_location("glm5_kda_payload", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import GGUF helpers from {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    helpers = load_helpers()
    data_start, tensors = helpers.load_directory(args.model)
    records = []
    combined = hashlib.sha256()
    with args.model.open("rb") as fp:
        blob = mmap.mmap(fp.fileno(), 0, access=mmap.ACCESS_READ)
        for layer in SPARSE_LAYERS:
            for suffix, count in (
                    ("attn_q_a_norm.weight", 1536),
                    ("attn_kv_a_norm.weight", 512)):
                name = f"blk.{layer}.{suffix}"
                shape, typ, relative = tensors.get(name, (None, None, None))
                if shape != (count,) or typ != 0:
                    raise ValueError(
                        f"{name}: expected F32 {(count,)}, got {(shape, typ)}")
                offset = data_start + relative
                payload = bytes(memoryview(blob)[offset:offset + count * 4])
                digest = hashlib.sha256(payload).hexdigest()
                combined.update(payload)
                values = struct.unpack_from(f"<{count}f", payload)
                if not all(math.isfinite(value) for value in values):
                    raise ValueError(f"{name}: non-finite parameter")
                for index in range(count):
                    bits = struct.unpack_from("<I", payload, index * 4)[0]
                    if bits & 0xffff:
                        raise ValueError(
                            f"{name}[{index}] is not widened-BF16 exact: "
                            f"0x{bits:08x}")
                records.append({
                    "name": name,
                    "count": count,
                    "sha256": digest,
                    "minimum": min(values),
                    "maximum": max(values),
                })
        blob.close()

    document = {
        "status": "payload provenance gate; not an inference baseline",
        "model_size_bytes": args.model.stat().st_size,
        "sparse_layers": list(SPARSE_LAYERS),
        "tensor_count": len(records),
        "combined_payload_sha256": combined.hexdigest(),
        "all_f32_payloads_are_widened_bf16_exact": True,
        "tensors": records,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n")
    print("PASS GLM5-next MLA norm payload provenance")
    print(f"layers={len(SPARSE_LAYERS)} tensors={len(records)} "
          f"combined_sha256={document['combined_payload_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
