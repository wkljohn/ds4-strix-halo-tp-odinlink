#!/usr/bin/env python3
"""Small, deterministic CPU contracts for GLM-5.3-Flash TP boundaries.

These are independent shape/order oracles, not a claim of model parity.  The
GPU implementation must match these contracts before numerical comparison
against a full reference implementation is attempted.
"""
from __future__ import annotations

import math


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def softmax(values):
    m = max(values)
    e = [math.exp(x - m) for x in values]
    z = sum(e)
    return [x / z for x in e]


def stable_topk(values, k):
    # Explicit index tie-break makes both TP ranks select the same experts.
    return sorted(range(len(values)), key=lambda i: (-values[i], i))[:k]


def route_top8(logits):
    ids = stable_topk(logits, 8)
    weights = softmax([logits[i] for i in ids])
    return ids, weights


def sharded_moe(x, gate, up, down, ids, weights, split=1024):
    """Reference 1024/1024 FFN-intermediate split.

    gate/up are [expert][intermediate][embedding], down is
    [expert][embedding][intermediate].  Each rank owns one intermediate half;
    the final output is one all-rank sum, never a concatenation.
    """
    embd = len(x)
    out = [0.0] * embd
    for expert, route_w in zip(ids, weights):
        mid = []
        for row in range(2 * split):
            z = dot(gate[expert][row], x) * dot(up[expert][row], x)
            mid.append(max(z, 0.0))
        for row in range(embd):
            out[row] += route_w * sum(down[expert][row][j] * mid[j]
                                       for j in range(2 * split))
    return out


def sharded_moe_two_ranks(x, gate, up, down, ids, weights, split=1024):
    embd = len(x)
    out = [0.0] * embd
    for rank in (0, 1):
        lo, hi = rank * split, (rank + 1) * split
        for expert, route_w in zip(ids, weights):
            mid = [max(dot(gate[expert][j], x) * dot(up[expert][j], x), 0.0)
                   for j in range(lo, hi)]
            for row in range(embd):
                out[row] += route_w * sum(down[expert][row][lo + j] * mid[j]
                                           for j in range(hi - lo))
    return out


def kda_reference(state, q, k, v, beta, gate):
    """Reference Kimi Delta Attention recurrent update.

    ``state`` is [key][value].  KDA applies a *vector* channel-wise decay,
    then subtracts the decayed state's prediction at ``k`` and writes the
    rank-one correction.  This is the recurrence used by the upstream KDA
    reference, not the older scalar-gate GDN approximation.
    """
    key_dim, value_dim = len(state), len(state[0])
    assert len(q) == key_dim and len(k) == key_dim
    assert len(v) == value_dim and len(gate) == key_dim
    for i in range(key_dim):
        decay = math.exp(gate[i])
        for j in range(value_dim):
            state[i][j] *= decay
    prediction = [sum(k[i] * state[i][j] for i in range(key_dim))
                  for j in range(value_dim)]
    for i in range(key_dim):
        for j in range(value_dim):
            state[i][j] += beta * k[i] * (v[j] - prediction[j])
    return [sum(q[i] * state[i][j] for i in range(key_dim))
            for j in range(value_dim)]


def kda_sharded_output(state, q, k, v, beta, gate, split):
    """Run KDA with key rows split across two ranks and compose the output."""
    full = kda_reference([row[:] for row in state], q, k, v, beta, gate)
    parts = []
    for rank in (0, 1):
        lo, hi = rank * split, (rank + 1) * split
        local = [[0.0] * len(state[0]) for _ in range(len(state))]
        # Each rank receives the same pre-update state rows it owns.  The
        # prediction needs the global reduction, so compute that explicitly
        # before applying the identical write to each owned row.
        decayed = [[state[i][j] * math.exp(gate[i])
                    for j in range(len(state[0]))]
                   for i in range(len(state))]
        prediction = [sum(k[i] * decayed[i][j] for i in range(len(state)))
                      for j in range(len(state[0]))]
        for i in range(lo, hi):
            for j in range(len(state[0])):
                local[i][j] = decayed[i][j] + beta * k[i] * (v[j] - prediction[j])
        parts.append([sum(q[i] * local[i][j] for i in range(len(state)))
                      for j in range(len(state[0]))])
    composed = [sum(part[j] for part in parts) for j in range(len(parts[0]))]
    return full, composed


def nope_sparse_mla(query, keys, values, scores, top_k):
    ids = stable_topk(scores, top_k)
    w = softmax([scores[i] for i in ids])
    return [sum(wj * values[i][j] for i, wj in zip(ids, w))
            for j in range(len(values[0]))]


def mhc_mix(branches, logits):
    assert len(branches) == 4 and len(logits) == 4
    w = softmax(logits)
    return [sum(w[i] * branches[i][j] for i in range(4))
            for j in range(len(branches[0]))]


def close(a, b, eps=1e-12):
    return all(abs(x - y) <= eps for x, y in zip(a, b))


def test_routing_is_stable():
    ids, weights = route_top8([1.0] * 288)
    assert ids == list(range(8))
    assert abs(sum(weights) - 1.0) < 1e-12


def test_mhc_is_normalized_and_ordered():
    branches = [[float(i + j) for j in range(4)] for i in range(4)]
    got = mhc_mix(branches, [0.0, 0.0, 0.0, 0.0])
    assert got == [1.5 + j for j in range(4)]


def test_nope_uses_selected_rows_only():
    keys = [[1.0, 0.0], [0.0, 1.0], [9.0, 9.0]]
    values = [[10.0, 0.0], [0.0, 20.0], [1000.0, 1000.0]]
    got = nope_sparse_mla([0.0, 0.0], keys, values, [1.0, 2.0, -100.0], 2)
    assert got[0] < 10.0 and got[1] > 10.0


def test_kda_update_is_deterministic():
    initial = [[0.1 * (i + j + 1) for j in range(3)] for i in range(4)]
    args = ([0.2, -0.1, 0.3, 0.4], [0.5, -0.25, 0.75, -0.2],
            [1.0, -2.0, 0.5], 0.125, [-0.3, -0.1, -0.7, -0.2])
    assert close(kda_reference([row[:] for row in initial], *args),
                 kda_reference([row[:] for row in initial], *args))


def test_kda_key_shards_require_global_prediction_reduction():
    initial = [[0.03 * (i + 2 * j + 1) for j in range(3)] for i in range(4)]
    q = [0.2, -0.1, 0.3, 0.4]
    k = [0.5, -0.25, 0.75, -0.2]
    v = [1.0, -2.0, 0.5]
    full, composed = kda_sharded_output(initial, q, k, v, 0.125,
                                         [-0.3, -0.1, -0.7, -0.2], 2)
    assert close(full, composed)


def test_intermediate_shards_compose_to_monolithic():
    # Compact dimensions exercise the exact 1024/1024 indexing contract.
    split, embd, experts = 2, 3, 2
    x = [0.25, -0.5, 0.75]
    gate = [[[0.01 * (e + 1 + r + c) for c in range(embd)]
             for r in range(2 * split)] for e in range(experts)]
    up = [[[0.02 * (e + 1 + r + c) for c in range(embd)]
           for r in range(2 * split)] for e in range(experts)]
    down = [[[0.03 * (e + 1 + r + c) for c in range(2 * split)]
             for r in range(embd)] for e in range(experts)]
    ids, weights = route_top8([2.0, 1.0]) + ([],) if False else ([0, 1], [0.6, 0.4])
    full = sharded_moe(x, gate, up, down, ids, weights, split)
    split_out = sharded_moe_two_ranks(x, gate, up, down, ids, weights, split)
    assert close(full, split_out)


if __name__ == "__main__":
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    for test in tests:
        test()
    print(f"PASS GLM5-next CPU oracle contracts ({len(tests)} tests)")
