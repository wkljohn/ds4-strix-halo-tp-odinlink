#!/usr/bin/env python3
"""Cross-check the GLM-5.3 KDA oracle against the pinned upstream equation.

The Torch functions below are a minimal float64 transcription of the public
Transformers implementation at commit
155b89935a648278dd38c78184cbc40e6a65f14b:

  src/transformers/models/glm5_next/modular_glm5_next.py, lines 375-475

They deliberately do not import or share arithmetic helpers with DS4. The
comparison side imports DS4's small pure-Python FP64 oracle. This test confirms
the formula and production dimensions; it is not a promoted model baseline.
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import torch


def load_ds4_oracle():
    source = Path(__file__).with_name("test_glm5_next_oracles.py")
    spec = importlib.util.spec_from_file_location("glm5_oracles", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def upstream_l2norm(value, eps=1e-6):
    return value / torch.sqrt((value * value).sum(dim=-1, keepdim=True) + eps)


def upstream_safe_forget(forget_projection, dt_bias, a_log, lower_bound=-5.0):
    decay_rate = torch.exp(a_log)[None, :, None]
    return lower_bound * torch.sigmoid(
        decay_rate *
        (forget_projection + dt_bias.view(1, forget_projection.shape[1], -1)))


def upstream_recurrent_kda(query, key, value, gate, beta, initial_state):
    query = upstream_l2norm(query) / math.sqrt(query.shape[-1])
    key = upstream_l2norm(key)
    state = initial_state.clone()
    output = torch.empty_like(value)
    for token in range(query.shape[0]):
        state = state * gate[token].exp().unsqueeze(-1)
        prediction = (state * key[token].unsqueeze(-1)).sum(dim=-2)
        delta = (value[token] - prediction) * beta[token].unsqueeze(-1)
        state = state + key[token].unsqueeze(-1) * delta.unsqueeze(-2)
        output[token] = (state * query[token].unsqueeze(-1)).sum(dim=-2)
    return output, state


def normalize_python(row, eps=1e-6):
    denominator = math.sqrt(sum(x * x for x in row) + eps)
    return [x / denominator for x in row]


def main():
    oracle = load_ds4_oracle()
    tokens, heads, key_dim, value_dim = 2, 64, 128, 128
    query = torch.empty((tokens, heads, key_dim), dtype=torch.float64)
    key = torch.empty_like(query)
    value = torch.empty((tokens, heads, value_dim), dtype=torch.float64)
    forget_projection = torch.empty_like(query)
    beta_logits = torch.empty((tokens, heads), dtype=torch.float64)
    for token in range(tokens):
        for head in range(heads):
            for channel in range(key_dim):
                index = 1 + channel + 131 * head + 8191 * token
                query[token, head, channel] = math.sin(0.0011 * index) * 0.3
                key[token, head, channel] = math.cos(0.0013 * index) * 0.25
                value[token, head, channel] = math.sin(0.0017 * index) * 0.2
                forget_projection[token, head, channel] = (
                    math.cos(0.0007 * index) * 0.4)
            beta_logits[token, head] = math.sin(0.07 * (1 + token + head))

    dt_bias = torch.tensor(
        [math.sin(0.011 * (i + 1 + 137 * head)) * 0.3
         for head in range(heads) for i in range(key_dim)],
        dtype=torch.float64)
    a_log = torch.tensor(
        [math.log(0.5 + 0.01 * (i + 1)) for i in range(heads)],
        dtype=torch.float64)
    gate = upstream_safe_forget(forget_projection, dt_bias, a_log)
    beta = torch.sigmoid(beta_logits)
    initial = torch.empty((heads, key_dim, value_dim), dtype=torch.float64)
    for head in range(heads):
        for key_row in range(key_dim):
            for value_col in range(value_dim):
                initial[head, key_row, value_col] = (
                    math.sin(0.00003 *
                             (1 + value_col + 131 * key_row + 17011 * head)) *
                    0.005)

    upstream_output, upstream_state = upstream_recurrent_kda(
        query, key, value, gate, beta, initial)
    ds4_output = torch.empty_like(upstream_output)
    ds4_state = torch.empty_like(upstream_state)
    scale = math.sqrt(key_dim)
    for head in range(heads):
        state = initial[head].tolist()
        for token in range(tokens):
            q = [x / scale for x in normalize_python(query[token, head].tolist())]
            k = normalize_python(key[token, head].tolist())
            v = value[token, head].tolist()
            g = gate[token, head].tolist()
            got = oracle.kda_reference(
                state, q, k, v, float(beta[token, head]), g)
            ds4_output[token, head] = torch.tensor(got, dtype=torch.float64)
        ds4_state[head] = torch.tensor(state, dtype=torch.float64)

    output_error = (upstream_output - ds4_output).abs().max().item()
    state_error = (upstream_state - ds4_state).abs().max().item()
    if output_error > 2e-15 or state_error > 2e-15:
        raise AssertionError(
            f"external KDA formula mismatch output={output_error:.9g} "
            f"state={state_error:.9g}")
    print("PASS GLM5 KDA pinned-upstream FP64 cross-check "
          f"shape={tokens}x{heads}x{key_dim}x{value_dim} "
          f"output_max_abs={output_error:.9g} state_max_abs={state_error:.9g}")


if __name__ == "__main__":
    main()
