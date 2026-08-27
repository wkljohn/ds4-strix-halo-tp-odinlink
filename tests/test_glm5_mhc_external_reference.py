#!/usr/bin/env python3
"""Cross-check a real-GGUF mHC payload against pinned upstream Torch math."""
from __future__ import annotations

import importlib.util
import mmap
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def deterministic_hidden(tokens=3):
    hidden = np.empty((tokens, 4, 4096), dtype=np.float32)
    indices = np.arange(4096, dtype=np.float32)
    for token in range(tokens):
        for stream in range(4):
            hidden[token, stream] = (
                np.sin(np.float32(0.0031) *
                       (indices + np.float32(1 + 17 * token + 31 * stream))) *
                np.float32(0.2) + np.float32(0.01 * (stream - 1.5)))
    return hidden


def upstream_torch(hidden, fn, base, scale, iterations=20,
                   hc_eps=1e-6, rms_eps=1e-5):
    """Pinned DeepseekV4HyperConnection.forward, without importing DS4 math."""
    streams = torch.from_numpy(hidden.copy())
    flat = streams.flatten(start_dim=1).float()
    flat = flat * torch.rsqrt(flat.pow(2).mean(dim=-1, keepdim=True) + rms_eps)
    mixed = F.linear(flat, torch.from_numpy(fn.copy()).float())
    pre_w, post_w, comb_w = mixed.split([4, 4, 16], dim=-1)
    pre_b, post_b, comb_b = torch.from_numpy(base.copy()).split([4, 4, 16])
    pre_scale, post_scale, comb_scale = torch.from_numpy(scale.copy()).unbind(0)
    pre = torch.sigmoid(pre_w * pre_scale + pre_b) + hc_eps
    post = 2 * torch.sigmoid(post_w * post_scale + post_b)
    comb = torch.softmax(
        comb_w.view(-1, 4, 4) * comb_scale + comb_b.view(4, 4), dim=-1) + hc_eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_eps)
    for _ in range(iterations - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + hc_eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_eps)
    collapsed = (pre.unsqueeze(-1) * streams).sum(dim=1)
    return post.numpy(), comb.numpy(), collapsed.numpy()


def main():
    model_value = os.environ.get("DS4_GLM5_MODEL")
    if not model_value:
        raise SystemExit("error: set DS4_GLM5_MODEL")
    model = Path(model_value)
    probe = load_module(
        Path(__file__).parents[1] / "scripts" / "probe-glm5-next-mhc-payload.py",
        "glm5_mhc_probe")
    helpers = probe.load_gguf_helpers()
    data_start, tensors = helpers.load_directory(model)
    with model.open("rb") as fp:
        blob = mmap.mmap(fp.fileno(), 0, access=mmap.ACCESS_READ)
        fn = probe.tensor_view(blob, data_start, tensors,
                               "blk.0.hc_attn_fn.weight", (16384, 24), 30).copy()
        base = probe.tensor_view(blob, data_start, tensors,
                                 "blk.0.hc_attn_base.weight", (24,), 0).copy()
        scale = probe.tensor_view(blob, data_start, tensors,
                                  "blk.0.hc_attn_scale.weight", (3,), 0).copy()
    hidden = deterministic_hidden()
    candidate = probe.mhc_reference(hidden, fn, base, scale)
    upstream = upstream_torch(hidden, fn, base, scale)
    errors = [float(np.max(np.abs(a - b))) for a, b in zip(candidate, upstream)]
    if errors[0] > 3e-6 or errors[1] > 3e-6 or errors[2] > 3e-6:
        raise AssertionError(f"real-GGUF mHC upstream mismatch {errors}")
    print("PASS real-GGUF mHC pinned-upstream cross-check "
          f"post={errors[0]:.9g} comb={errors[1]:.9g} "
          f"collapsed={errors[2]:.9g}")


if __name__ == "__main__":
    main()
