#!/usr/bin/env python3
"""Read-only structural checker for the GLM-5.3-Flash GGUF contract.

It reads only the GGUF header, metadata and tensor descriptors; tensor payloads
are never mapped or allocated.  Type 30 is GGML BF16; the checker still does
not interpret tensor payloads.
"""
from __future__ import annotations

import struct
import sys
from collections import Counter
from pathlib import Path


class Reader:
    def __init__(self, fp):
        self.fp = fp

    def u8(self): return struct.unpack("<B", self.fp.read(1))[0]
    def u32(self): return struct.unpack("<I", self.fp.read(4))[0]
    def u64(self): return struct.unpack("<Q", self.fp.read(8))[0]

    def string(self):
        n = self.u64()
        b = self.fp.read(n)
        if len(b) != n:
            raise ValueError("truncated GGUF string")
        return b.decode("utf-8", "replace")

    def skip_value(self, typ):
        sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4,
                 7: 1, 10: 8, 11: 8, 12: 8}
        if typ in sizes:
            self.fp.seek(sizes[typ], 1)
        elif typ == 8:
            self.fp.seek(self.u64(), 1)
        elif typ == 9:
            elem = self.u32()
            for _ in range(self.u64()):
                self.skip_value(elem)
        else:
            raise ValueError(f"unsupported metadata type {typ}")


def read(path: Path):
    with path.open("rb") as fp:
        r = Reader(fp)
        magic, version, nt, nk = struct.unpack("<IIQQ", fp.read(24))
        if magic != 0x46554747 or version != 3:
            raise ValueError("expected GGUF version 3")
        meta = {}
        for _ in range(nk):
            key = r.string()
            typ = r.u32()
            if typ == 8:
                meta[key] = r.string()
            elif typ == 4:
                meta[key] = r.u32()
            elif typ == 6:
                meta[key] = struct.unpack("<f", fp.read(4))[0]
            elif typ == 7:
                meta[key] = bool(r.u8())
            elif typ == 10:
                meta[key] = r.u64()
            elif typ == 9:
                elem = r.u32()
                count = r.u64()
                if elem == 4:
                    meta[key] = tuple(r.u32() for _ in range(count))
                elif elem == 5:
                    meta[key] = tuple(
                        struct.unpack("<i", fp.read(4))[0]
                        for _ in range(count)
                    )
                else:
                    for _ in range(count):
                        r.skip_value(elem)
            else:
                r.skip_value(typ)
        tensors = {}
        for _ in range(nt):
            name = r.string()
            nd = r.u32()
            dims = tuple(r.u64() for _ in range(nd))
            typ = r.u32()
            offset = r.u64()
            tensors[name] = (dims, typ, offset)
        return meta, tensors


def require(meta, key, expected):
    got = meta.get(key)
    if got != expected:
        raise ValueError(f"metadata {key}: expected {expected!r}, got {got!r}")


def main(argv):
    if len(argv) != 2:
        print(f"usage: {argv[0]} MODEL.gguf", file=sys.stderr)
        return 2
    meta, tensors = read(Path(argv[1]))
    require(meta, "general.architecture", "glm5-next")
    for key, value in {
        "glm5-next.block_count": 46,
        "glm5-next.trunk_block_count": 45,
        "glm5-next.nextn_predict_layers": 1,
        "glm5-next.embedding_length": 4096,
        "glm5-next.vocab_size": 154880,
        "glm5-next.expert_count": 288,
        "glm5-next.expert_used_count": 8,
        "glm5-next.expert_shared_count": 1,
        "glm5-next.expert_feed_forward_length": 2048,
        "glm5-next.expert_weights_scale": 2.5,
        "glm5-next.expert_weights_norm": True,
        "glm5-next.swiglu_limit": 10.0,
        "glm5-next.feed_forward_length": 12288,
        "glm5-next.attention.head_count": 64,
        "glm5-next.attention.key_length": 256,
        "glm5-next.attention.value_length": 256,
        "glm5-next.attention.indexer.pool_size": 4,
        "glm5-next.attention.indexer.top_k": 2048,
        "glm5-next.linear_attention.conv_kernel": 4,
        "glm5-next.linear_attention.gate_lower_bound": -5.0,
    }.items():
        require(meta, key, value)

    expected_layer_types = tuple(
        1 if layer == 45 or layer % 4 == 3 else 0
        for layer in range(46)
    )
    require(meta, "glm5-next.layer_types", expected_layer_types)

    # Q4_K is uniform; the compact Q2 reference is mixed IQ2_XXS gate/up
    # plus Q2_K down.  Keep the type alternatives explicit and shape-bound.
    required = {
        "blk.3.ffn_gate_exps.weight": ((4096, 2048, 288), (12, 16)),
        "blk.3.ffn_up_exps.weight": ((4096, 2048, 288), (12, 16)),
        "blk.3.ffn_down_exps.weight": ((2048, 4096, 288), (12, 10)),
        # BF16 is used by the Q4 reference; the compact Q2 conversion uses
        # Q4_K for KDA q/k.  Both are validated layouts.
        "blk.0.kda_q.weight": ((4096, 8192), (30, 12)),
        "blk.3.attn_q_a.weight": ((4096, 1536), 8),
        "blk.3.attn_q_a_norm.weight": ((1536,), 0),
        "blk.3.attn_kv_a_norm.weight": ((512,), 0),
        "blk.3.hc_attn_fn.weight": ((16384, 24), 30),
    }
    for name, (shape, typ) in required.items():
        got = tensors.get(name)
        allowed_types = typ if isinstance(typ, tuple) else (typ,)
        if got is None or got[0] != shape or got[1] not in allowed_types:
            raise ValueError(f"tensor {name}: expected shape/type {(shape, typ)!r}, got {got!r}")

    def count_suffix(suffix):
        return sum(name.endswith(suffix) for name in tensors)

    # Freeze the architecture schedule observed in the reference GGUF.
    # These counts are deliberately separate from metadata: tensor presence
    # catches partial or incorrectly converted files.
    for suffix, expected in {
        ".kda_q.weight": 34,
        ".attn_q_a.weight": 12,
        ".attn_q_a_norm.weight": 12,
        ".attn_kv_a_norm.weight": 12,
        ".ffn_gate_exps.weight": 43,
        ".hc_attn_fn.weight": 45,
        ".hc_ffn_fn.weight": 45,
        ".nextn.eh_proj.weight": 1,
    }.items():
        got = count_suffix(suffix)
        if got != expected:
            raise ValueError(f"tensor schedule {suffix}: expected {expected}, got {got}")

    def layer_indices(suffix):
        out = set()
        for name in tensors:
            if name.endswith(suffix) and name.startswith("blk."):
                out.add(int(name.split(".")[1]))
        return out

    expected_kda = {i for i in range(45) if i % 4 != 3}
    expected_mla = set(range(3, 44, 4)) | {45}
    if layer_indices(".kda_q.weight") != expected_kda:
        raise ValueError("KDA layer schedule does not match 3-dense/4th-sparse pattern")
    if layer_indices(".attn_q_a.weight") != expected_mla:
        raise ValueError("sparse MLA layer schedule does not match expected pattern")
    for suffix in (".ffn_gate_exps.weight", ".ffn_up_exps.weight",
                   ".ffn_down_exps.weight"):
        if layer_indices(suffix) != set(range(3, 46)):
            raise ValueError(f"routed expert schedule mismatch for {suffix}")

    if any(name.startswith("blk.45.hc_") for name in tensors):
        raise ValueError("block 45 unexpectedly contains mHC tensors")

    # Validate every descriptor in each family.  A count-only check would let
    # one malformed layer or an axis-swapped conversion pass unnoticed.
    family_shapes = {
        ".kda_q.weight": ((4096, 8192), (30, 12)),
        ".kda_k.weight": ((4096, 8192), (30, 12)),
        ".kda_v.weight": ((4096, 8192), (30, 8)),
        ".kda_output.weight": ((8192, 4096), (30, 8)),
        ".indexer.attn_q_b.weight": ((1536, 4096), 30),
        ".indexer.attn_k.weight": ((4096, 128), 30),
        ".indexer.proj.weight": ((4096, 32), 30),
        ".indexer.pool_ape.weight": ((128, 4), 30),
        ".indexer.pool_gate.weight": ((4096, 128), 30),
        ".indexer.k_norm.weight": ((128,), 0),
        ".indexer.k_norm.bias": ((128,), 0),
        ".attn_q_a.weight": ((4096, 1536), 8),
        ".attn_q_a_norm.weight": ((1536,), 0),
        ".attn_q_b.weight": ((1536, 16384), 8),
        ".attn_kv_a_mqa.weight": ((4096, 512), 8),
        ".attn_kv_a_norm.weight": ((512,), 0),
        ".attn_k_b.weight": ((256, 512, 64), 8),
        ".attn_v_b.weight": ((512, 256, 64), 8),
        ".attn_output.weight": ((16384, 4096), 8),
        ".ffn_gate_exps.weight": ((4096, 2048, 288), (12, 16)),
        ".ffn_up_exps.weight": ((4096, 2048, 288), (12, 16)),
        ".ffn_down_exps.weight": ((2048, 4096, 288), (12, 10)),
        ".hc_attn_fn.weight": ((16384, 24), 30),
        ".hc_ffn_fn.weight": ((16384, 24), 30),
    }
    for name, (shape, typ) in family_shapes.items():
        for tensor_name, (got_shape, got_type, _offset) in tensors.items():
            if (tensor_name.startswith("blk.") and tensor_name.endswith(name)
                    and not (name.startswith(".attn_") and ".indexer." in tensor_name)):
                allowed_types = typ if isinstance(typ, tuple) else (typ,)
                if got_shape != shape or got_type not in allowed_types:
                    raise ValueError(f"tensor {tensor_name}: expected {(shape, typ)!r}, got {(got_shape, got_type)!r}")

    layers = Counter()
    for name in tensors:
        if name.startswith("blk.") and "." in name[4:]:
            layers[name.split(".", 2)[1]] += 1
    routed_types = tuple(sorted({t[1] for name, t in tensors.items()
                                 if name.endswith((".ffn_gate_exps.weight",
                                                   ".ffn_up_exps.weight",
                                                   ".ffn_down_exps.weight"))}))
    if routed_types not in ((10, 12), (12,), (10, 16), (12, 16)):
        raise ValueError(f"unsupported routed expert type family: {routed_types}")
    print("PASS glm5-next metadata/tensor contract")
    print(f"tensors={len(tensors)} routed_expert_layers="
          f"{sum(1 for n in tensors if n.endswith('ffn_gate_exps.weight'))} "
          f"routed_types={routed_types}")
    print("type_counts=" + ",".join(f"{k}:{v}" for k, v in sorted(Counter(t[1] for t in tensors.values()).items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
