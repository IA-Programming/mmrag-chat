"""DEV ONLY (EXP-VIS-01): generator image-token cap vs cross-modal accuracy and TTFT.

Image sources (and video frames) dominate TTFT on 2 cores: every image is encoded by the
projector and adds hundreds of prompt tokens. This sweeps llama-server's --image-max-tokens
for the generator preset on the cross-modal questions (cached index).
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from core.config import DATASETS_DIR
from core.index.store import IndexHandle
from tools.eval.ledger import append
from tools.eval.run import Rig, all_candidate_presets, correct, load_jsonl, turn


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", default="lfm3-vl-2b-qad-q4_0")
    ap.add_argument("--caps", nargs="*", type=int, default=[0, 256, 128, 64])
    ap.add_argument("--workdir", type=Path, default=Path("experiments/_raw/eval-a"))
    args = ap.parse_args()
    ds = DATASETS_DIR / "synth_v1"
    base = next(p for p in all_candidate_presets(False) if p.id == args.candidate)
    index = IndexHandle.open(next(args.workdir.glob("index-*")))
    qs = [q for q in load_jsonl(ds / "queries" / "queries.jsonl") if q["type"] == "cross_modal"]
    out = {}
    for cap in args.caps:
        p = base.model_copy(deep=True)
        p.id = f"{base.id}-img{cap}"
        if cap:
            p.extra_args["image-max-tokens"] = cap
            p.extra_args["image-min-tokens"] = min(64, cap)
        rig = Rig.start([p], Path("/rt/llama-server"), args.workdir / f"vis{cap}")
        rows = []
        try:
            for q in qs:
                att = ds / q["attachment"] if q.get("attachment") else None
                if att is not None and att.suffix.lower() not in (".png", ".jpg", ".jpeg"):
                    continue
                r = turn(rig, p, index, q["text"], attachment=att)
                rows.append(
                    {"id": q["id"], "correct": correct(r["answer"], q), "ttft_ms": r["stats"].get("ttft_ms"), "prompt_tokens": r["stats"].get("prompt_tokens")}
                )
        finally:
            rig.stop()
        tt = sorted(x["ttft_ms"] for x in rows if x["ttft_ms"])
        out[str(cap or "default")] = {"n": len(rows), "accuracy": round(sum(x["correct"] for x in rows) / len(rows), 3), "ttft_median_s": round(statistics.median(tt) / 1000, 1),
                                 "ttft_p95_s": round(tt[int(0.95 * (len(tt) - 1))] / 1000, 1), "prompt_tokens_median": statistics.median(x["prompt_tokens"] or 0 for x in rows)}  # fmt: skip
        print(cap, out[str(cap or "default")], flush=True)
    append("EXP-VIS-01", candidate_id=args.candidate, params={"queries": "cross_modal"}, metrics=out, verdict="see metrics")
    Path("experiments/_raw/vision_tokens.json").write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
