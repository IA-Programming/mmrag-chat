"""DEV ONLY: render the latest S12 gate results from experiments/ledger.jsonl as Markdown tables.

python -m tools.eval.report > /tmp/table.md
"""

from __future__ import annotations

from typing import Any

from tools.eval.ledger import read

COLS = [
    ("candidate", "Candidate"),
    ("files_mb", "Files MB"),
    ("steady_mb", "Steady MB"),
    ("peak_mb", "Peak MB"),
    ("peak_pct", "Peak %"),
    ("load_cold_s", "Cold load s"),
    ("g5_drift_pct", "Load/unload drift %"),
    ("ttft_p95_s", "TTFT p95 s"),
    ("decode_tok_s_median", "Decode tok/s"),
    ("retrieval_hit@4", "Retrieval hit@4"),
    ("accuracy", "Answer accuracy"),
    ("refusal_rate", "Refusal on negatives"),
    ("g4_window", "Multi-turn window(4)"),
    ("g4_none", "Multi-turn none"),
]


def latest_by_candidate() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for r in read():
        if r.exp_id == "S12-gates" and r.candidate_id:
            out[r.candidate_id] = r
    return out


def fmt(v: Any) -> str:
    if v is None:
        return "–"
    if isinstance(v, float):
        return f"{v:.3f}" if v < 1.5 else f"{v:.1f}"
    return str(v)


def table() -> str:
    rows = latest_by_candidate()
    head = "| " + " | ".join(h for _, h in COLS) + " | Gates | Verdict |"
    sep = "|" + "---|" * (len(COLS) + 2)
    lines = [head, sep]
    for cid, r in sorted(rows.items(), key=lambda kv: (kv[1].verdict != "shipped", -(kv[1].metrics.get("accuracy") or -1))):
        m = dict(r.metrics, candidate=cid)
        gates = " ".join(f"{k}:{v}" for k, v in r.gates.items())
        lines.append("| " + " | ".join(fmt(m.get(k)) for k, _ in COLS) + f" | {gates} | {r.verdict} |")
    return "\n".join(lines)


if __name__ == "__main__":
    print(table())
