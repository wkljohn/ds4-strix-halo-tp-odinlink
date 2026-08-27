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


def sigmoid(value):
    if value >= 0.0:
        e = math.exp(-value)
        return 1.0 / (1.0 + e)
    e = math.exp(value)
    return e / (1.0 + e)


def stable_topk(values, k):
    # Explicit index tie-break makes both TP ranks select the same experts.
    return sorted(range(len(values)), key=lambda i: (-values[i], i))[:k]


def route_top8(logits):
    ids = stable_topk(logits, 8)
    weights = softmax([logits[i] for i in ids])
    return ids, weights


def glm5_route_top8(router_logits, correction_bias=None, scale=2.5):
    """GLM-5.3 sparse router contract (one group, sigmoid affinity).

    The correction bias affects selection only; normalized output weights come
    from the unbiased sigmoid scores and are multiplied by the model scale.
    """
    if correction_bias is None:
        correction_bias = [0.0] * len(router_logits)
    assert len(router_logits) == len(correction_bias)
    scores = [1.0 / (1.0 + math.exp(-x)) for x in router_logits]
    choice = [scores[i] + correction_bias[i] for i in range(len(scores))]
    ids = stable_topk(choice, 8)
    z = sum(scores[i] for i in ids) + 1e-20
    return ids, [scale * scores[i] / z for i in ids]


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


def conv4_sequence(history, values, weights):
    """Independent depthwise four-tap conv + SiLU sequence oracle."""
    channels = len(history)
    assert channels == len(weights)
    assert all(len(row) == 3 for row in history)
    assert all(len(row) == 4 for row in weights)
    state = [row[:] for row in history]
    output = []
    for token in values:
        assert len(token) == channels
        out_row = []
        for channel in range(channels):
            h0, h1, h2 = state[channel]
            x = token[channel]
            w0, w1, w2, w3 = weights[channel]
            raw = h0 * w0 + h1 * w1 + h2 * w2 + x * w3
            out_row.append(raw / (1.0 + math.exp(-raw)))
            state[channel] = [h1, h2, x]
        output.append(out_row)
    return output, state


def run_conv_chunks(values, weights, initial_history, chunks):
    assert sum(chunks) == len(values)
    history = [row[:] for row in initial_history]
    output = []
    pos = 0
    for count in chunks:
        chunk_out, history = conv4_sequence(
            history, values[pos:pos + count], weights)
        output.extend(chunk_out)
        pos += count
    return output, history


def _flatten(values):
    for value in values:
        if isinstance(value, (list, tuple)):
            yield from _flatten(value)
        else:
            yield value


def max_abs_rel(reference, candidate):
    ref = list(_flatten(reference))
    got = list(_flatten(candidate))
    assert len(ref) == len(got)
    abs_err = max((abs(a - b) for a, b in zip(ref, got)), default=0.0)
    rel_err = max((abs(a - b) / max(abs(a), 1e-30)
                   for a, b in zip(ref, got)), default=0.0)
    return abs_err, rel_err


def kda_sequence(state, q, k, v, beta, gate):
    """Sequential one-head KDA oracle returning output and copied state."""
    assert len(q) == len(k) == len(v) == len(beta) == len(gate)
    current = [row[:] for row in state]
    output = []
    for token in range(len(q)):
        assert len(q[token]) == len(k[token]) == len(v[token]) == 1
        assert len(beta[token]) == len(gate[token]) == 1
        output.append(kda_reference(
            current, q[token][0], k[token][0], v[token][0],
            beta[token][0], gate[token][0]))
    return output, current


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


def nope_sparse_mla(query, keys, values, top_k):
    """NoPE sparse attention: score keys from the query, then select top-k."""
    scores = [dot(query, key) for key in keys]
    ids = stable_topk(scores, top_k)
    w = softmax([scores[i] for i in ids])
    return [sum(wj * values[i][j] for i, wj in zip(ids, w))
            for j in range(len(values[0]))]


def rms_norm_unweighted(values, eps):
    rms = math.sqrt(sum(x * x for x in values) / len(values) + eps)
    return [x / rms for x in values]


def mhc_mapping(hidden_streams, fn, base, scale, iterations=20,
                hc_eps=1e-6, rms_eps=1e-5):
    """Exact GLM-5.3 mHC mapping used by the official Transformers graph.

    ``hidden_streams`` is [H][D], ``fn`` is [(2+H)*H][H*D], and the
    24 mapping outputs for H=4 are split into pre, post, and comb. The first
    comb normalization is column-only because its rows came from a softmax;
    the remaining Sinkhorn iterations alternate row and column.
    """
    hc = len(hidden_streams)
    width = len(hidden_streams[0])
    assert hc == 4 and all(len(row) == width for row in hidden_streams)
    assert len(fn) == (2 + hc) * hc
    assert all(len(row) == hc * width for row in fn)
    assert len(base) == len(fn) and len(scale) == 3

    flat = rms_norm_unweighted(
        [x for stream in hidden_streams for x in stream], rms_eps)
    mixed = [dot(row, flat) for row in fn]
    pre_logits = mixed[:hc]
    post_logits = mixed[hc:2 * hc]
    comb_logits = [mixed[2 * hc + i * hc:2 * hc + (i + 1) * hc]
                   for i in range(hc)]
    pre = [sigmoid(pre_logits[i] * scale[0] + base[i]) + hc_eps
           for i in range(hc)]
    post = [2.0 * sigmoid(post_logits[i] * scale[1] + base[hc + i])
            for i in range(hc)]
    comb = []
    for i in range(hc):
        logits = [comb_logits[i][j] * scale[2] +
                  base[2 * hc + i * hc + j] for j in range(hc)]
        comb.append([x + hc_eps for x in softmax(logits)])
    for j in range(hc):
        z = sum(comb[i][j] for i in range(hc)) + hc_eps
        for i in range(hc):
            comb[i][j] /= z
    for _ in range(iterations - 1):
        for i in range(hc):
            z = sum(comb[i]) + hc_eps
            comb[i] = [x / z for x in comb[i]]
        for j in range(hc):
            z = sum(comb[i][j] for i in range(hc)) + hc_eps
            for i in range(hc):
                comb[i][j] /= z
    collapsed = [sum(pre[i] * hidden_streams[i][j] for i in range(hc))
                 for j in range(width)]
    return post, comb, collapsed


def mhc_compose(residual_streams, branch, post, comb):
    """Apply ``post * branch + comb.T @ residual`` in model stream order."""
    hc = len(residual_streams)
    width = len(branch)
    return [[post[j] * branch[k] +
             sum(comb[i][j] * residual_streams[i][k] for i in range(hc))
             for k in range(width)] for j in range(hc)]


def glm5_kpool(keys, gate_scores, valid, ape, pool_size=4):
    """Reference learned key pooling, including first-valid-token alignment."""
    assert len(keys) == len(gate_scores) == len(valid)
    assert len(ape) == pool_size
    width = len(keys[0])
    assert all(len(row) == width for row in keys)
    assert all(len(row) == width for row in gate_scores)
    assert all(len(row) == width for row in ape)
    first = next((i for i, is_valid in enumerate(valid) if is_valid), len(valid))
    pools = []
    indices = []
    for start in range(first, len(keys), pool_size):
        idx = list(range(start, min(start + pool_size, len(keys))))
        if len(idx) != pool_size or not all(valid[i] for i in idx):
            continue
        pooled = []
        for channel in range(width):
            probability = softmax([
                gate_scores[token][channel] + ape[offset][channel]
                for offset, token in enumerate(idx)])
            pooled.append(sum(probability[offset] * keys[token][channel]
                              for offset, token in enumerate(idx)))
        pools.append(pooled)
        indices.append(idx)
    return pools, indices


def close(a, b, eps=1e-12):
    return all(abs(x - y) <= eps for x, y in zip(a, b))


def test_routing_is_stable():
    ids, weights = route_top8([1.0] * 288)
    assert ids == list(range(8))
    assert abs(sum(weights) - 1.0) < 1e-12


def test_glm5_router_bias_only_changes_choice():
    logits = [0.0] * 288
    bias = [0.0] * 288
    bias[17] = 0.5
    ids, weights = glm5_route_top8(logits, bias)
    assert ids[0] == 17
    assert len(ids) == 8 and abs(sum(weights) - 2.5) < 1e-12
    # Selection correction is not allowed to alter the affinity used for
    # expert weighting: equal raw logits produce equal selected weights.
    assert max(weights) - min(weights) < 1e-12


def test_mhc_is_normalized_and_ordered():
    streams = [[float(i + j) for j in range(3)] for i in range(4)]
    fn = [[0.0] * 12 for _ in range(24)]
    post, comb, collapsed = mhc_mapping(
        streams, fn, [0.0] * 24, [1.0, 1.0, 1.0])
    assert all(abs(x - 1.0) < 1e-12 for x in post)
    expected = [(0.5 + 1e-6) * (6.0 + 4.0 * j) for j in range(3)]
    assert close(collapsed, expected, 1e-12)
    for row in comb:
        assert all(abs(x - 0.25) < 2e-6 for x in row)


def test_mhc_sinkhorn_preserves_stream_mass():
    streams = [[float(10 * i + j) for j in range(3)] for i in range(4)]
    fn = [[0.0] * 12 for _ in range(24)]
    base = [0.0] * 8 + [
        3.0, -1.0, 0.5, 2.0, 0.1, 4.0, -2.0, 0.3,
        1.0, 0.2, 2.5, -0.4, -1.0, 0.7, 0.4, 3.5,
    ]
    post, comb, _ = mhc_mapping(streams, fn, base, [1.0, 1.0, 1.0])
    mixed = mhc_compose(streams, [0.0, 0.0, 0.0], post, comb)
    # The doubly-stochastic residual map preserves feature mass to the
    # explicit finite-iteration epsilon, unlike a per-row-only softmax.
    for feature in range(3):
        assert abs(sum(mixed[i][feature] for i in range(4)) -
                   sum(streams[i][feature] for i in range(4))) < 2e-2


def test_glm5_kpool_aligns_after_padding_and_omits_tail():
    keys = [[10.0 * token + channel for channel in range(2)]
            for token in range(8)]
    gates = [[0.0, 0.0] for _ in keys]
    valid = [False, False, True, True, True, True, True, True]
    ape = [[0.0, 0.0] for _ in range(4)]
    pools, indices = glm5_kpool(keys, gates, valid, ape)
    assert indices == [[2, 3, 4, 5]]
    assert close(pools[0], [35.0, 36.0])


def test_nope_uses_selected_rows_only():
    query = [1.0, 2.0]
    keys = [[1.0, 0.0], [0.0, 1.0], [-9.0, -9.0]]
    values = [[10.0, 0.0], [0.0, 20.0], [1000.0, 1000.0]]
    got = nope_sparse_mla(query, keys, values, 2)
    assert got[0] < 10.0 and got[1] > 10.0


def test_kda_update_is_deterministic():
    initial = [[0.1 * (i + j + 1) for j in range(3)] for i in range(4)]
    args = ([0.2, -0.1, 0.3, 0.4], [0.5, -0.25, 0.75, -0.2],
            [1.0, -2.0, 0.5], 0.125, [-0.3, -0.1, -0.7, -0.2])
    assert close(kda_reference([row[:] for row in initial], *args),
                 kda_reference([row[:] for row in initial], *args))


def test_conv4_sequence_keeps_the_last_three_samples():
    out, history = conv4_sequence(
        [[1.0, 2.0, 3.0]], [[4.0], [5.0]], [[0.1, 0.2, 0.3, 0.4]])
    assert abs(out[0][0] - 2.8577223804672998) < 1e-15
    assert abs(out[1][0] - 3.9280551601516338) < 1e-15
    assert history == [[3.0, 4.0, 5.0]]


def test_conv4_chunking_matches_one_call_at_boundaries():
    values = [[math.sin(0.031 * (t * 4 + c + 1))
               for c in range(4)] for t in range(129)]
    weights = [[0.03 * (tap + 1) * (c + 1)
                for tap in range(4)] for c in range(4)]
    initial = [[0.01 * (c + tap + 1) for tap in range(3)]
               for c in range(4)]
    one_call_out, one_call_history = run_conv_chunks(
        values, weights, initial, [129])
    for chunks in ([1, 128], [2, 127], [3, 126], [127, 1, 1]):
        got_out, got_history = run_conv_chunks(
            values, weights, initial, chunks)
        abs_err, rel_err = max_abs_rel(one_call_out, got_out)
        assert abs_err <= 1e-15 and rel_err <= 1e-15
        assert got_history == one_call_history


def test_kda_sequence_applies_decay_before_prediction():
    out, final = kda_sequence(
        [[2.0]], [[[3.0]]], [[[0.5]]], [[[4.0]]],
        [[0.25]], [[[math.log(0.5)]]])
    assert abs(final[0][0] - 1.4375) < 1e-15
    assert abs(out[0][0] - 4.3125) < 1e-15


def test_kda_sequence_stays_finite_for_adversarial_gates():
    q = [[[0.02 * (i + 1) for i in range(8)]] for _ in range(3)]
    k = [[[0.01 * (8 - i) for i in range(8)]] for _ in range(3)]
    v = [[[0.03 * (i - 2) for i in range(5)]] for _ in range(3)]
    beta = [[1e-7], [0.5], [1.0 - 1e-7]]
    initial = [[0.001 * (1 + i + 2 * j) for j in range(5)]
               for i in range(8)]
    cases = (
        [[-1e-7] * 8 for _ in range(3)],
        [[-5.0] * 8 for _ in range(3)],
        [[-1e-7, -5.0, -0.2, -3.0, -0.9, -4.0, -0.01, -2.0]
         for _ in range(3)],
    )
    for gate in cases:
        out, final = kda_sequence(
            [row[:] for row in initial], q, k, v, beta,
            [[row] for row in gate])
        assert all(math.isfinite(x) for row in final for x in row)
        assert all(math.isfinite(x) for row in out for x in row)


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
