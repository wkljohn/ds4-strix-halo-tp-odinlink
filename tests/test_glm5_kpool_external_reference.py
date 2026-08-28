#!/usr/bin/env python3
"""Cross-check DS4's GLM-5.3 pool contract against pinned upstream Torch math.

The Torch function below independently transcribes
huggingface/transformers@155b89935a648278dd38c78184cbc40e6a65f14b,
``Glm5NextTextIndexer.get_pooled_states``.  It validates the compact
single-sequence contract in ``test_glm5_next_oracles.py``; the ROCm gate
separately validates the raw pre-``keep`` pool axis used by the device API.

The comparison is intentionally exact for the pinned Torch/CPU build. Some
pre-BF16 probabilities differ by one or two F32 ULPs and one randomized value
lands on a BF16 tie, so a Torch or CPU-vectorization change may require
revalidating this host oracle; such a failure is not by itself a device-kernel
regression.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import random

import torch


def load_ds4_oracle():
    source = Path(__file__).with_name("test_glm5_next_oracles.py")
    spec = importlib.util.spec_from_file_location("glm5_oracles", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def upstream_pool(keys, gates, valid, ape, pool_size=4):
    keys_t = torch.tensor(keys, dtype=torch.bfloat16).unsqueeze(0)
    gates_t = torch.tensor(gates, dtype=torch.bfloat16).unsqueeze(0)
    valid_t = torch.tensor(valid, dtype=torch.bool).unsqueeze(0)
    ape_t = torch.tensor(ape, dtype=torch.bfloat16)
    _, length, _ = keys_t.shape
    n_pools = (length + pool_size - 1) // pool_size
    first = torch.where(
        valid_t.any(-1), valid_t.long().argmax(-1),
        torch.full((1,), length, dtype=torch.long))
    offsets = torch.arange(n_pools * pool_size).view(1, n_pools, pool_size)
    indices = first[:, None, None] + offsets
    safe = indices.clamp(0, length - 1)
    batch = torch.arange(1)[:, None, None]
    grouped_keys = keys_t[batch, safe]
    grouped_gates = gates_t[batch, safe]
    grouped_valid = valid_t[batch, safe] & indices.lt(length)
    pool_valid = grouped_valid.all(-1)
    indices = indices.masked_fill(~grouped_valid, -1)
    logits = grouped_gates.float() + ape_t.float()[None, None]
    logits = logits.masked_fill(~grouped_valid[..., None], float("-inf"))
    probabilities = torch.nan_to_num(logits.softmax(dim=2)).to(grouped_keys.dtype)
    pooled = (probabilities * grouped_keys).sum(dim=2)
    keep = pool_valid.any(0)
    return pooled[:, keep][0].tolist(), indices[:, keep][0].tolist()


def run_case(length, first_valid, invalidate=None):
    width = 7
    keys = [[0.1 * row + 0.01 * channel for channel in range(width)]
            for row in range(length)]
    gates = [[-0.03 * row + 0.02 * channel for channel in range(width)]
             for row in range(length)]
    valid = [row >= first_valid for row in range(length)]
    if invalidate is not None:
        valid[invalidate] = False
    ape = [[0.07 * member - 0.005 * channel for channel in range(width)]
           for member in range(4)]
    upstream = upstream_pool(keys, gates, valid, ape)
    oracle = load_ds4_oracle()
    candidate = oracle.glm5_kpool(keys, gates, valid, ape)
    if upstream[1] != candidate[1]:
        raise AssertionError(
            f"pool index mismatch length={length} first={first_valid}: "
            f"{upstream[1]} != {candidate[1]}")
    if len(upstream[0]) != len(candidate[0]):
        raise AssertionError("pool count mismatch")
    maximum = max(
        (abs(a - b) for ua, ca in zip(upstream[0], candidate[0])
         for a, b in zip(ua, ca)), default=0.0)
    if maximum != 0.0:
        raise AssertionError(f"pool value mismatch max_abs={maximum:.9g}")


def main():
    for args in ((1, 0, None), (3, 0, None), (4, 0, None),
                 (5, 0, None), (9, 1, None), (17, 1, None),
                 (19, 2, None), (20, 0, 5)):
        run_case(*args)
    # Nontrivial BF16 values connect the pure-F32 oracle to the pinned Torch
    # expression; the simple padding cases alone use exact quarter weights and
    # would not detect misplaced probability/product boundaries.
    oracle = load_ds4_oracle()
    for seed in range(128):
        rng = random.Random(seed)
        width = 16
        keys = [[rng.uniform(-2.0, 2.0) for _ in range(width)]
                for _ in range(8)]
        gates = [[rng.uniform(-5.0, 5.0) for _ in range(width)]
                 for _ in range(8)]
        valid = [True] * 8
        ape = [[rng.uniform(-1.0, 1.0) for _ in range(width)]
               for _ in range(4)]
        upstream = upstream_pool(keys, gates, valid, ape)
        candidate = oracle.glm5_kpool(keys, gates, valid, ape)
        if upstream[1] != candidate[1] or upstream[0] != candidate[0]:
            raise AssertionError(f"random BF16 pool mismatch seed={seed}")
    print("PASS GLM5 kpool pinned-upstream boundary cross-check")


if __name__ == "__main__":
    main()
