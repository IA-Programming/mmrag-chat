"""DEV ONLY (EXP-HIST-01): history policy x contextualization on the scripted dialogues.

Follow-up turns ("How many steps does HER tower have?") retrieve nothing useful when the query
is embedded alone; this measures window(4)/none x off/concat_last_user/llm_rewrite on all
dialogues (turn correctness, retrieval of the expected source, latency, extra calls).
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from core.config import DATASETS_DIR
from core.index.store import IndexHandle
from core.rag.history import HistoryConfig
from tools.eval.ledger import append
from tools.eval.run import Rig, all_candidate_presets, basename_hits, correct, load_jsonl, turn

CONFIGS = [("window", "off"), ("window", "concat_last_user"), ("window", "llm_rewrite"), ("none", "off")]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", default="lfm3-vl-2b-qad-q4_0")
    ap.add_argument("--workdir", type=Path, default=Path("experiments/_raw/eval-a"))
    ap.add_argument("--dialogues", type=int, default=10)
    args = ap.parse_args()
    ds = DATASETS_DIR / "synth_v1"
    p = next(x for x in all_candidate_presets(False) if x.id == args.candidate)
    index = IndexHandle.open(next(args.workdir.glob("index-*")))
    dialogues = load_jsonl(ds / "dialogues" / "dialogues.jsonl")[: args.dialogues]
    rig = Rig.start([p], Path("/rt/llama-server"), args.workdir / "hist")
    out = {}
    try:
        for policy, ctx in CONFIGS:
            rows = []
            for d in dialogues:
                hist: list[dict[str, str]] = []
                for i, t in enumerate(d["turns"]):
                    t0 = time.time()
                    r = turn(rig, p, index, t["user"], history=hist, hist=HistoryConfig(policy, 4, ctx))
                    rows.append({"turn": i, "correct": correct(r["answer"], t), "retrieved": basename_hits(r["sources"], t.get("expected_sources", [])),
                                 "s": round(time.time() - t0, 1), "extra_calls": r["stats"].get("extra_generation_calls", 0)})  # fmt: skip
                    hist += [{"role": "user", "text": t["user"]}, {"role": "assistant", "text": r["answer"]}]
            follow = [x for x in rows if x["turn"] > 0]
            out[f"{policy}+{ctx}"] = {
                "turns": len(rows),
                "correct_all": round(sum(x["correct"] for x in rows) / len(rows), 3),
                "correct_followups": round(sum(x["correct"] for x in follow) / len(follow), 3),
                "retrieved_followups": round(sum(x["retrieved"] for x in follow) / len(follow), 3),
                "median_turn_s": statistics.median(x["s"] for x in rows),
                "extra_calls_per_turn": round(sum(x["extra_calls"] for x in rows) / len(rows), 2),
            }
            print(policy, ctx, out[f"{policy}+{ctx}"], flush=True)
    finally:
        rig.stop()
    best = max(out, key=lambda k: (out[k]["correct_all"], -out[k]["median_turn_s"]))
    append("EXP-HIST-01", candidate_id=args.candidate, params={"dialogues": len(dialogues)}, metrics=out, verdict=f"best {best}")
    Path("experiments/_raw/history_exp.json").write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
