#!/usr/bin/env python3
"""Generate the fixed cross-disciplinary TP decode prompt (v1)."""

from __future__ import annotations

import argparse
import sys


HEADER = """You are reviewing a mixed technical dossier. Preserve exact facts, distinguish observations from hypotheses, and synthesize only after reading the available records. The records deliberately alternate disciplines so hardware paths are not tuned against one style of text.\n\n"""

TEMPLATES = (
    (
        "software-debugging",
        "Service shard {n} processed {requests} requests after build r{build}. "
        "Its median queue delay was {latency} microseconds, retry count was {retries}, "
        "and checksum {checksum} matched on both hosts. The suspected race concerns an "
        "event recorded on a producer stream before a registered buffer becomes visible. "
        "A valid repair must preserve ordering, bounded failure, and the ordinary path.",
    ),
    (
        "quantitative-science",
        "Experiment {n} heated a {mass}-gram alloy sample from {temp0} to {temp1} kelvin "
        "while pressure remained {pressure} kilopascals. The measured energy was {energy} "
        "joules with uncertainty {uncertainty} percent. Investigators must compare the "
        "implied specific heat with the control, propagate uncertainty, and avoid treating "
        "correlation between temperature and drift as proof of causation.",
    ),
    (
        "policy-document-retrieval",
        "Clause {n} applies to records labeled class-{klass}. Retention lasts {years} years, "
        "but a litigation hold overrides deletion and an access request must be acknowledged "
        "within {days} business days. The controlling exception code is PX-{code}. Auditors "
        "must cite the clause, separate mandatory language from guidance, and report conflicts "
        "instead of silently choosing the newest sentence.",
    ),
    (
        "structured-data-analysis",
        "Dataset row {n} names region R-{region}, quarter Q{quarter}, revenue {revenue}, "
        "cost {cost}, returned units {returns}, and cohort key C-{cohort}. Missing values use "
        "the literal NA and are not zeros. Analysis should compute margins consistently, flag "
        "outliers without discarding them, and keep row identifiers attached to every derived "
        "claim so another analyst can reproduce the result.",
    ),
)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def render(cycles: int = 24) -> str:
    chunks = [HEADER]
    for cycle in range(1, cycles + 1):
        values = {
            "n": cycle,
            "requests": 1700 + cycle * 37,
            "build": 400 + cycle,
            "latency": 41 + cycle % 13,
            "retries": cycle % 4,
            "checksum": f"{(0x9E3779B1 * cycle) & 0xFFFFFFFF:08x}",
            "mass": 80 + cycle * 3,
            "temp0": 285 + cycle,
            "temp1": 330 + cycle * 2,
            "pressure": 98 + cycle % 7,
            "energy": 1220 + cycle * 29,
            "uncertainty": f"{1.0 + (cycle % 6) * 0.2:.1f}",
            "klass": chr(ord("A") + cycle % 5),
            "years": 2 + cycle % 8,
            "days": 3 + cycle % 9,
            "code": 700 + cycle * 11,
            "region": 10 + cycle % 9,
            "quarter": 1 + cycle % 4,
            "revenue": 50000 + cycle * 1307,
            "cost": 28000 + cycle * 811,
            "returns": cycle * 7 % 53,
            "cohort": 900 + cycle * 17,
        }
        for discipline, template in TEMPLATES:
            chunks.append(f"[{discipline} record {cycle}]\n{template.format(**values)}\n\n")
    chunks.append(
        "Task: summarize the strongest supported conclusion from the records available in "
        "your context, name one uncertainty, and state which exact record identifiers support "
        "your answer. Do not invent a missing record.\n"
    )
    return "".join(chunks)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cycles",
        type=positive_int,
        default=24,
        help="number of four-discipline record cycles to emit (default: 24)",
    )
    args = parser.parse_args()
    sys.stdout.write(render(args.cycles))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
