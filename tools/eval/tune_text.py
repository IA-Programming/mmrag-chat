"""DEV ONLY (DR-006, EXP-EMB-01): text chunk size x overlap sweep on the synthetic dataset.

Metric that matters for answers: does a chunk that CONTAINS the answer land in the top-k?
(file-level hit@k hides the problem: the right document with the wrong chunk). Also reports
MRR, tokens per chunk (prompt cost of k sources) and embedding time.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from core.config import DATASETS_DIR
from core.embed.client import EmbedService, truncate_renorm
from core.embed.preprocess import chunk_text, estimate_tokens
from core.runtime.presets import load_embed_params
from core.schemas import TextParams
from tools.eval.ledger import append
from tools.eval.run import Rig, load_jsonl


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", nargs="*", type=int, default=[128, 192, 256, 384, 512])
    ap.add_argument("--overlaps", nargs="*", type=float, default=[0.0, 0.15])
    ap.add_argument("--k", type=int, default=4)
    args = ap.parse_args()
    ds = DATASETS_DIR / "synth_v1"
    docs = {p.name: p.read_text() for p in sorted((ds / "corpus" / "text").glob("*.md"))}
    qs = [q for q in load_jsonl(ds / "queries" / "queries.jsonl") if not q.get("attachment") and not q["must_refuse"]
          and any(e.endswith(".md") for e in q["expected_sources"])]  # fmt: skip
    rig = Rig.start([], Path("/rt/llama-server"), Path("/tmp/tune"))
    params = load_embed_params()
    emb = EmbedService(rig.client, params)
    results = []
    try:
        with rig.governor.turn("tune", "x"), rig.governor.using("emb-text"):
            qv = np.vstack([emb.embed_query(q["text"], model="emb-text") for q in qs])
            for size in args.sizes:
                for ov in args.overlaps:
                    tp = TextParams(chunk_tokens=size, overlap_tokens=int(size * ov))
                    chunks: list[tuple[str, str]] = []
                    for name, text in docs.items():
                        chunks += [(name, c) for c, _ in chunk_text(text, tp)]
                    t0 = time.time()
                    vecs = []
                    for i in range(0, len(chunks), 8):
                        vecs.append(emb.embed_pieces([params.prefix.document.format(content=c) for _, c in chunks[i : i + 8]], "emb-text"))
                    embed_s = time.time() - t0
                    m = truncate_renorm(np.vstack(vecs), 768)
                    order = np.argsort(-(qv @ m.T), axis=1)
                    ans_hit = {1: 0, args.k: 0, 8: 0}
                    file_hit = 0
                    rr = []
                    for qi, q in enumerate(qs):
                        exp = {Path(e).name for e in q["expected_sources"]}
                        kws = [k.lower() for k in q.get("answer_keywords", [])]
                        ranked = order[qi]
                        first_ans = next(
                            (r + 1 for r, ci in enumerate(ranked[:50]) if chunks[ci][0] in exp and any(k in chunks[ci][1].lower() for k in kws)), None
                        )
                        for kk in ans_hit:
                            ans_hit[kk] += int(first_ans is not None and first_ans <= kk)
                        file_hit += int(any(chunks[ci][0] in exp for ci in ranked[: args.k]))
                        rr.append(1 / first_ans if first_ans else 0.0)
                    n = len(qs)
                    tok = float(np.mean([estimate_tokens(c) for _, c in chunks]))
                    row = {"chunk_tokens": size, "overlap": int(size * ov), "chunks": len(chunks), "tokens_per_chunk": round(tok, 1),
                           "prompt_tokens_k_sources": round(tok * args.k), "embed_s": round(embed_s, 1),
                           "answer_hit@1": round(ans_hit[1] / n, 3), f"answer_hit@{args.k}": round(ans_hit[args.k] / n, 3), "answer_hit@8": round(ans_hit[8] / n, 3),
                           f"file_hit@{args.k}": round(file_hit / n, 3), "answer_mrr": round(float(np.mean(rr)), 3)}  # fmt: skip
                    results.append(row)
                    print(json.dumps(row), flush=True)
    finally:
        rig.stop()
    best = max(results, key=lambda r: (r[f"answer_hit@{args.k}"], -r["prompt_tokens_k_sources"]))
    append("EXP-EMB-01", params={"queries": len(qs), "k": args.k}, metrics={"grid": results}, verdict=f"best chunk_tokens={best['chunk_tokens']} overlap={best['overlap']}",
           notes="answer_hit = a chunk containing the answer keyword is in the top-k (file-level hit hides wrong-chunk retrieval).")  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
