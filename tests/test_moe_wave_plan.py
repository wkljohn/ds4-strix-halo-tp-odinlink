#!/usr/bin/env python3
"""Structural oracle for an exact routed-MoE producer wave plan.

The production implementation is not allowed to sort each token wave
independently.  It must retain the full-batch stable expert buckets and full
counts, then expose contiguous subranges of those buckets.  This test proves
the descriptor and compact-repack indexing rules before a GPU kernel is added.
"""

import math
import random
import struct
import unittest


TILE = 16


def full_plan(selected, total_experts):
    buckets = [[] for _ in range(total_experts)]
    for pair, expert in enumerate(selected):
        if 0 <= expert < total_experts:
            buckets[expert].append(pair)
    offsets = [0]
    sorted_pairs = []
    for bucket in buckets:
        sorted_pairs.extend(bucket)
        offsets.append(len(sorted_pairs))
    counts = [len(bucket) for bucket in buckets]
    return counts, offsets, sorted_pairs


def wave_tiles(counts, offsets, sorted_pairs, used_experts,
               row_first, row_end):
    """Return (expert, global-bucket-start, exclusive-end) descriptors."""
    tiles = []
    for expert, count in enumerate(counts):
        bucket = sorted_pairs[offsets[expert]:offsets[expert + 1]]
        local = [idx for idx, pair in enumerate(bucket)
                 if row_first <= pair // used_experts < row_end]
        if not local:
            continue
        start = local[0]
        end = local[-1] + 1
        # Stable full-batch buckets make each token wave contiguous.
        assert local == list(range(start, end))
        for tile_start in range(start, end, TILE):
            tiles.append((expert, tile_start, end))
        # Kernel path selection must use the full count, not end - start.
        assert count == offsets[expert + 1] - offsets[expert]
    return tiles


def pairs_from_tiles(tiles, offsets, sorted_pairs):
    pairs = []
    for expert, start, end in tiles:
        for local in range(start, min(start + TILE, end)):
            pairs.append(sorted_pairs[offsets[expert] + local])
    return pairs


def f32(value):
    return struct.unpack("f", struct.pack("f", value))[0]


def canonical_sum(pair_values, selected, tokens, used_experts):
    result = []
    for token in range(tokens):
        acc = f32(0.0)
        for slot in range(used_experts):
            pair = token * used_experts + slot
            if selected[pair] >= 0:
                acc = f32(acc + pair_values[pair])
        result.append(acc)
    return result


class MoeWavePlanTest(unittest.TestCase):
    def make_routes(self, tokens=73, used=6, total=32):
        # Adversarial distribution: empty experts, exact 15/16/17 boundaries,
        # wave-edge routes, repeated experts, and peer-owned negative sentinels.
        rng = random.Random(0xD54)
        selected = []
        for token in range(tokens):
            for slot in range(used):
                if (token * used + slot) % 19 == 0:
                    selected.append(-1)
                elif token in (0, tokens // 2 - 1, tokens // 2, tokens - 1):
                    selected.append((slot * 7 + token) % 11)
                else:
                    selected.append(rng.randrange(0, total - 4))
        # Force exact threshold neighborhoods without changing pair order.
        live = [i for i, expert in enumerate(selected) if expert >= 0]
        cursor = 0
        for expert, count in ((28, 1), (29, 15), (30, 16), (31, 17)):
            for pair in live[cursor:cursor + count]:
                selected[pair] = expert
            cursor += count
        return selected

    def check_wave_count(self, nwaves):
        tokens, used, total = 73, 6, 32
        selected = self.make_routes(tokens, used, total)
        counts, offsets, sorted_pairs = full_plan(selected, total)
        covered = []
        pair_values = [f32(math.sin(pair * 0.071) * 3.0)
                       for pair in range(len(selected))]
        wave_values = [f32(0.0)] * len(selected)
        for wave in range(nwaves):
            first = tokens * wave // nwaves
            end = tokens * (wave + 1) // nwaves
            tiles = wave_tiles(counts, offsets, sorted_pairs, used, first, end)
            pairs = pairs_from_tiles(tiles, offsets, sorted_pairs)
            self.assertTrue(all(first <= pair // used < end for pair in pairs))
            self.assertEqual(len(pairs), len(set(pairs)))
            covered.extend(pairs)
            for pair in pairs:
                wave_values[pair] = pair_values[pair]

            # Compact wave repack must map back to the full transposed layout.
            pair_first = first * used
            pair_count = (end - first) * used
            for half_block in (0, 1, 7, 15):
                for local in (0, pair_count // 2, pair_count - 1):
                    global_pair = pair_first + local
                    compact = half_block * pair_count + local
                    full = half_block * len(selected) + global_pair
                    self.assertEqual(compact - half_block * pair_count, local)
                    self.assertEqual(full - half_block * len(selected),
                                     global_pair)

        expected = [pair for pair, expert in enumerate(selected) if expert >= 0]
        self.assertEqual(sorted(covered), expected)
        self.assertEqual(len(covered), len(set(covered)))

        # The down path writes distinct pair-major rows. Only this canonical,
        # non-atomic slot-order reduction is eligible for exact producer waves.
        full_sum = canonical_sum(pair_values, selected, tokens, used)
        wave_sum = canonical_sum(wave_values, selected, tokens, used)
        self.assertEqual(
            [struct.pack("f", value) for value in wave_sum],
            [struct.pack("f", value) for value in full_sum])

        # Threshold decisions remain based on the full bucket histogram.
        self.assertEqual(counts[28:32], [1, 15, 16, 17])

    def test_two_waves(self):
        self.check_wave_count(2)

    def test_four_waves(self):
        self.check_wave_count(4)


if __name__ == "__main__":
    unittest.main()
