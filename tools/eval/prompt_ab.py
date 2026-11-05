"""DEV ONLY (EXP-PROMPT-02): prompt layout/instruction A/B on real dataset questions (cached index).

docker run ... mmrag-runtime:bullseye python -m tools.eval.prompt_ab --candidate lfm3-vl-2b-qad-q4_0 --n 8
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

from core.config import DATASETS_DIR
from core.index.store import IndexHandle
from core.rag import prompt_builder as pb
from tools.eval.ledger import append
from tools.eval.run import Rig, all_candidate_presets, correct, load_jsonl, refused, turn

XLANG = " The question may be in a different language than the sources: still use the sources, and answer in the language of the question."
VARIANTS: dict[str, dict[str, Any]] = {
    "current": {},
    "cross_language": {"system": pb.SYSTEM_PROMPT + XLANG},
    "no_bracket_lead": {
        "system": pb.SYSTEM_PROMPT + " Never reply with only a citation: always state the answer itself in words first.",
    },
    "question_first": {"question_first": True},
    "both": {"question_first": True, "system": pb.SYSTEM_PROMPT + " Never reply with only a citation: always state the answer itself in words first."},
}
BARE = re.compile(r"^\s*(\[S\d+\]\s*)+\.?\s*$")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", default="lfm3-vl-2b-qad-q4_0")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--types", nargs="*", default=None, help="query types to sample (default: cross_modal + single/near-dup)")
    ap.add_argument("--variants", nargs="*", default=None)
    ap.add_argument("--workdir", type=Path, default=Path("experiments/_raw/eval"))
    args = ap.parse_args()
    ds = DATASETS_DIR / "synth_v1"
    preset = next(p for p in all_candidate_presets(False) if p.id == args.candidate)
    index = IndexHandle.open(next(args.workdir.glob("index-*")))
    qs = [q for q in load_jsonl(ds / "queries" / "queries.jsonl") if not q.get("attachment")]
    if args.types:
        pick = [q for q in qs if q["type"] in args.types][: args.n]
    else:
        pick = [q for q in qs if q["type"] == "cross_modal"][: args.n // 2] + [q for q in qs if q["type"] in ("single_modal", "near_duplicate")][
            : args.n - args.n // 2
        ]
    rig = Rig.start([preset], Path("/rt/llama-server"), args.workdir / "ab")
    orig_system, orig_build = pb.SYSTEM_PROMPT, pb.build_prompt
    results: dict[str, Any] = {}
    try:
        for name, v in VARIANTS.items():
            if args.variants and name not in args.variants:
                continue
            pb.SYSTEM_PROMPT = v.get("system", orig_system)

            def build(v: dict[str, Any] = v, **kw: Any) -> Any:
                out = orig_build(**kw)
                msgs = out.messages
                msgs[0]["content"] = pb.SYSTEM_PROMPT
                if v.get("question_first"):
                    q = f"Question: {kw['query']}"
                    c = msgs[-1]["content"]
                    if isinstance(c, str):
                        msgs[-1]["content"] = q + "\n\n" + c
                    else:
                        msgs[-1]["content"] = [{"type": "text", "text": q}, *c]
                return out

            import core.rag.pipeline as pl

            pl.build_prompt = build  # type: ignore[assignment]
            rows = []
            for q in pick:
                t0 = time.time()
                r = turn(rig, preset, index, q["text"])
                rows.append({"id": q["id"], "type": q["type"], "correct": correct(r["answer"], q), "bare": bool(BARE.match(r["answer"])), "refused": refused(r["answer"]),
                             "ttft_ms": r["stats"].get("ttft_ms"), "s": round(time.time() - t0, 1), "answer": r["answer"][:160]})  # fmt: skip
            results[name] = {"accuracy": round(sum(x["correct"] for x in rows) / len(rows), 3), "bare_citations": sum(x["bare"] for x in rows), "rows": rows}
            print(name, results[name]["accuracy"], "bare:", results[name]["bare_citations"], flush=True)
    finally:
        pl.build_prompt = orig_build  # type: ignore[assignment]
        pb.SYSTEM_PROMPT = orig_system
        rig.stop()
    summary = {k: {"accuracy": v["accuracy"], "bare_citations": v["bare_citations"]} for k, v in results.items()}
    append("EXP-PROMPT-02", candidate_id=args.candidate, params={"queries": [q["id"] for q in pick]}, metrics=summary, verdict="see metrics")
    Path("experiments/_raw/prompt_ab.json").write_text(json.dumps(results, indent=2))
    _ = refused
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
