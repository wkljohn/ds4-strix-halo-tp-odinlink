#!/usr/bin/env python3
"""Pin GLM-5.3 visible-tail selection to upstream Transformers semantics.

The reference transcribes ``Glm5NextTextIndexer.append_visible_tail`` from
huggingface/transformers@155b89935a648278dd38c78184cbc40e6a65f14b.
The expected result is independently derived by enumerating the contiguous
visible run after complete four-token pools.
"""


def upstream_append(topk, token_visible, key_valid, pool_size=4):
    max_tail_width = pool_size - 1
    first_key = next((i for i, valid in enumerate(key_valid) if valid),
                     len(key_valid))
    visible_count = sum(token_visible)
    tail_count = visible_count % pool_size
    tail_start = first_key + visible_count - tail_count
    tail = []
    for offset in range(max_tail_width):
        index = tail_start + offset
        valid = (offset < tail_count and index < len(key_valid)
                 and token_visible[index])
        tail.append(index if valid else -1)
    return [*topk, *tail]


def independently_expected(topk, token_visible, pool_size=4):
    visible = [index for index, value in enumerate(token_visible) if value]
    covered = len(visible) // pool_size * pool_size
    tail = visible[covered:]
    return [*topk, *tail, *([-1] * (pool_size - 1 - len(tail)))]


def run_case(key_valid, query_position, topk):
    token_visible = [valid and index <= query_position
                     for index, valid in enumerate(key_valid)]
    candidate = upstream_append(topk, token_visible, key_valid)
    expected = independently_expected(topk, token_visible)
    if candidate != expected:
        raise AssertionError(
            f"tail mismatch position={query_position}: "
            f"{candidate} != {expected}")


def main():
    for length in (1, 3, 4, 5, 2047, 2048, 2049, 8195):
        valid = [True] * length
        run_case(valid, length - 1, [12, 8])
    for length, first in ((9, 1), (19, 2), (23, 3)):
        valid = [False] * first + [True] * (length - first)
        run_case(valid, length - 1, [7, 3])
        run_case(valid, min(length - 1, first + 5), [7, 3])
    run_case([False, False, False], 2, [])
    print("PASS GLM5 indexer pinned-upstream visible-tail cross-check")


if __name__ == "__main__":
    main()
