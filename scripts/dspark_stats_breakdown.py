#!/usr/bin/env python3
"""Decompose `ds4: DSpark stats ...` lines into a per-cycle time budget.

Usage: dspark_stats_breakdown.py <coordinator log> [...]

Prints one row per generation, then the warm median t/s across rows 2..N
(row 1 is the cold run and is excluded, matching the acceptance gate).
"""
import re
import statistics
import sys


def parse(line):
    d = {}
    for k, v in re.findall(r"(\w+)=([0-9.]+)", line):
        d[k] = float(v)
    return d


def main(paths):
    rows = []
    for path in paths:
        gen = None
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                m = re.search(r"generation: ([0-9.]+) t/s", line)
                if m:
                    gen = float(m.group(1))
                elif "DSpark stats" in line and gen is not None:
                    rows.append((path, gen, parse(line)))
                    gen = None

    if not rows:
        print("no DSpark stats lines found")
        return 1

    hdr = (f"{'t/s':>6} {'cyc':>4} {'tok':>4} {'acc%':>6} {'avg':>5} "
           f"{'anchor':>7} {'prop':>7} {'verify':>8} {'replay':>7} {'cycle':>7}")
    print(hdr)
    print("-" * len(hdr))
    for _path, gen, d in rows:
        cycles = d.get("cycles", 0) or 1
        tokens = d.get("first_tokens", 0) + d.get("accepted_draft", 0)
        anchor = d.get("target", 0) / cycles
        prop = d.get("propose", 0) / cycles
        verify = d.get("verify", 0) / cycles
        replay = d.get("replay", 0) / cycles
        cycle = anchor + prop + verify + replay + d.get("snapshot", 0) / cycles
        print(f"{gen:6.2f} {cycles:4.0f} {tokens:4.0f} "
              f"{d.get('accept_rate', 0):6.2f} {d.get('avg_accept', 0):5.2f} "
              f"{anchor:7.1f} {prop:7.1f} {verify:8.1f} {replay:7.1f} {cycle:7.1f}")

    gens = [g for _p, g, _d in rows]
    warm = gens[1:] if len(gens) > 1 else gens
    print()
    print(f"runs={len(gens)}  all={[f'{g:.2f}' for g in gens]}")
    print(f"warm median (excluding first) = {statistics.median(warm):.2f} t/s")
    print(f"GATE >17 t/s: {'PASS' if statistics.median(warm) > 17.0 else 'FAIL'}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1:]))
