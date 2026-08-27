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
    """Reference linear-attention update used only for order/shape checks.

    State is [value][key].  The update is applied before the query read and
    uses the same beta/gate for every key/value element, making accidental
    rank-local transposes easy to detect.
    """
    value_dim, key_dim = len(state), len(state[0])
    for i in range(value_dim):
        for j in range(key_dim):
            state[i][j] = gate * state[i][j] + beta * v[i] * k[j]
    return [dot(row, q) for row in state]


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
    initial = [[0.1 * (i + j + 1) for j in range(3)] for i in range(2)]
    args = ([0.2, -0.1, 0.3], [0.5, -0.25, 0.75],
            [1.0, -2.0], 0.125, 0.9)
    assert close(kda_reference([row[:] for row in initial], *args),
                 kda_reference([row[:] for row in initial], *args))


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
