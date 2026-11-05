"""DEV ONLY (DR-005, S12): run candidates through gates G0-G5 inside the capped emulator.

    docker run --rm --memory=2700m --memory-swap=2700m --cpus=2 --cpuset-cpus=0,1 \\
      -v $PWD:/app -v ~/.cache/mmrag/models:/models:ro -v $PWD/build/llama/pkg:/rt:ro -w /app \\
      mmrag-runtime:bullseye python -m tools.eval.run --candidates lfm3-vl-2b-qad-q4_0 qwen3.5-0.8b-q4_k_m

Uses the same core code as the app (router, governor, indexer, ChatPipeline). Memory is the
llama-server process tree + a fixed app overhead (gates.yaml), sampled every 0.2 s. Every gate
result is appended to experiments/ledger.jsonl; a summary goes to experiments/eval_report.md.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import psutil
import yaml

from core.config import DATASETS_DIR
from core.embed.client import EmbedService
from core.embed.indexer import run_index_job, stage_upload
from core.errors import AppError
from core.index.store import IndexHandle
from core.jobs import Job
from core.obs.log import Logger, LogRing
from core.rag.history import HistoryConfig
from core.rag.pipeline import Attachment, ChatPipeline, RetrievalConfig, TurnRequest
from core.rag.prompt_builder import REFUSAL
from core.rag.sampling import resolve
from core.runtime.governor import ALWAYS_RESIDENT, MM_EMBEDDER, Governor
from core.runtime.presets import load_embed_params, load_model_presets, load_sampling_presets, write_ini
from core.runtime.router import RouterClient, RouterProcess
from core.runtime.telemetry import Telemetry
from core.schemas import Capabilities, KVConfig, ModelPreset, ModelPresets
from tools.eval.ledger import append

ROOT = Path(__file__).resolve().parents[2]
GATES = yaml.safe_load((ROOT / "tools" / "eval" / "gates.yaml").read_text())
CANDIDATES = yaml.safe_load((ROOT / "tools" / "eval" / "candidates.yaml").read_text())
MODELS_ROOT = Path(os.environ.get("MMRAG_EVAL_MODELS", "/models"))
REFUSAL_RE = re.compile(r"not (in|part of) the indexed data|no (relevant )?information|cannot find|not mentioned|do(es)? not contain", re.I)
CAP = float(GATES["mem_cap_mb"])
OVERHEAD = float(GATES["app_overhead_mb"])


# --------------------------------------------------------------------------- presets for candidates
def _size_mb(repo: str, name: str) -> float:
    p = MODELS_ROOT / repo.replace("/", "__") / name
    return p.stat().st_size / 1e6 if p.exists() else 0.0


def candidate_preset(c: dict[str, Any], model_file: str | None = None, suffix: str = "") -> ModelPreset:
    files = dict(c["files"])
    repo = c["repo"]
    if model_file:
        files["model"] = model_file
        repo = c["quant_ladder"]["repo"]
    text_only = bool(c.get("text_only")) or bool(suffix)
    if text_only:
        files.pop("mmproj", None)
    family = c.get("family", "")
    est = sum(_size_mb(repo if k == "model" else c["repo"], v) for k, v in files.items()) + 250
    full_attention = family in ("ministral", "gemma4")
    return ModelPreset(
        id=(c["id"].replace("q4_k_m", suffix.strip("-").removesuffix("-text")) + "-text") if suffix else c["id"],
        label=(c["id"].replace("q4_k_m", suffix.strip("-").removesuffix("-text")) + "-text") if suffix else c["id"],
        role="generator",
        family=family,
        files={k: f"{(repo if k == 'model' else c['repo']).replace('/', '__')}/{v}" for k, v in files.items()},
        n_ctx=4096 if full_attention else 8192,
        kv=KVConfig(k="q8_0", v="q8_0"),
        flash_attn="auto",
        threads=2,
        batch=512,
        ubatch=256,
        capabilities=Capabilities(
            image="mmproj" in files,
            audio=family == "lfm3" and "omni" in c["id"] and "mmproj" in files,
            video="frames" if "mmproj" in files else False,
            reasoning_control="chat_template_kwargs" if family in ("lfm3", "qwen3.5", "minicpm", "lfm2.5", "gemma4") else "none",
        ),
        sampling_preset=c.get("sampling_preset"),
        est_peak_mb=int(est),
        extra_args={"load-mode": "none", "no-warmup": True},
        source={"repo": c["repo"], "revision": c["revision"]},
    )


def all_candidate_presets(ladder: bool) -> list[ModelPreset]:
    out = []
    for c in CANDIDATES["generators"]:
        out.append(candidate_preset(c))
        if ladder and c.get("quant_ladder"):
            for f in c["quant_ladder"]["files"]:
                tag = re.sub(r".*-(UD-)?", "", f.removesuffix(".gguf")).lower()
                out.append(candidate_preset(c, f, suffix=f"-{tag}-text"))
    return out


def embedder_presets() -> list[ModelPreset]:
    base = load_model_presets()
    out = []
    for pid in (ALWAYS_RESIDENT, MM_EMBEDDER):
        p = base.get(pid).model_copy(deep=True)
        p.files = {k: f"{p.source['repo'].replace('/', '__')}/{p.source.get(k, v)}" for k, v in p.files.items()}
        out.append(p)
    return out


# --------------------------------------------------------------------------- telemetry
class TreeMeter:
    def __init__(self, pid: int) -> None:
        self.pid = pid

    def used_mb(self) -> float:
        try:
            root = psutil.Process(self.pid)
            procs = [root, *root.children(recursive=True)]
        except psutil.Error:
            return OVERHEAD
        total = 0
        for p in procs:
            try:
                total += p.memory_info().rss
            except psutil.Error:
                pass
        return total / 1e6 + OVERHEAD


class Sampler:
    def __init__(self, meter: TreeMeter, interval: float = 0.2) -> None:
        self.meter, self.interval = meter, interval
        self.peak = 0.0
        self._stop = threading.Event()

    def __enter__(self) -> Sampler:
        self.peak = self.meter.used_mb()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            self.peak = max(self.peak, self.meter.used_mb())

    def __exit__(self, *a: object) -> None:
        self._stop.set()
        self._t.join(2)
        self.peak = max(self.peak, self.meter.used_mb())


def evict_page_cache(paths: list[Path]) -> None:
    """Cold-load timing: drop the files from the page cache (EXP-LOAD-01)."""
    for p in paths:
        try:
            fd = os.open(p, os.O_RDONLY)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            os.close(fd)
        except OSError:
            pass


# --------------------------------------------------------------------------- rig
@dataclass
class Rig:
    presets: ModelPresets
    router: RouterProcess
    client: RouterClient
    governor: Governor
    meter: TreeMeter
    log: Logger
    telemetry: Telemetry

    @classmethod
    def start(cls, generators: list[ModelPreset], server: Path, workdir: Path) -> Rig:
        presets = ModelPresets(schema=1, presets=[*embedder_presets(), *generators])
        ini = write_ini(workdir / "presets.ini", presets.presets, lambda n: MODELS_ROOT / n, threads=2)
        log = Logger(LogRing(max_bytes=2_000_000), stdout=False, level="INFO")
        router = RouterProcess([str(server)], ini, log_dir=workdir, logger=log, models_max=4)
        router.start(timeout=60)
        client = RouterClient(router.base_url, router.api_key, logger=log)
        assert router.proc is not None
        meter = TreeMeter(router.proc.pid)
        gov = Governor(
            client, presets, meter.used_mb, hard_budget_mb=CAP * GATES["G1"]["peak_max_pct"] / 100, logger=log, load_timeout_s=GATES["G0"]["ready_within_s"]
        )
        return cls(presets, router, client, gov, meter, log, Telemetry(int(CAP), source=meter.used_mb))

    def stop(self) -> None:
        self.client.close()
        self.router.stop()


# --------------------------------------------------------------------------- dataset
def load_jsonl(p: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()] if p.exists() else []


def index_dataset(rig: Rig, ds: Path, cache: Path, *, max_frames: int, include: set[str]) -> IndexHandle:
    params = load_embed_params()
    params.video.max_frames = max_frames
    key = f"{json.loads((ds / 'manifest.json').read_text()).get('bytes_total', 0)}-{max_frames}-{'-'.join(sorted(include))}"
    root = cache / f"index-{key}"
    if (root / "index.tv").exists():
        return IndexHandle.open(root)
    files = sorted(p for p in (ds / "corpus").rglob("*") if p.is_file() and p.parent.name in include)
    session = cache / "session"
    items = [stage_upload(session, p.name, p.read_bytes(), used_bytes=0, quota_mb=100_000, max_pixels=params.image.max_pixels) for p in files]
    h = IndexHandle(root, fingerprint="eval")
    job = Job(id="eval", sid="eval", kind="index")
    t0 = time.time()
    with Sampler(rig.meter) as s:
        res = run_index_job(job, items=items, index=h, governor=rig.governor, embed=EmbedService(rig.client, params), params=params, telemetry=rig.telemetry,
                            log=rig.log, preview_dir=root / "previews")  # fmt: skip
    append("EVAL-INDEX", env=env(), params={"files": len(items), "max_frames": max_frames, "include": sorted(include)},
           metrics={"seconds": round(time.time() - t0, 1), "peak_mb": round(s.peak, 1), "ok": len(res["ok"]), "failed": res["failed"][:10], "stats": res["index"]},
           verdict="indexed")  # fmt: skip
    return h


def env() -> dict[str, Any]:
    return {"where": "emulator", "cpus": len(os.sched_getaffinity(0)), "mem_cap_mb": CAP, "app_overhead_mb": OVERHEAD}


# --------------------------------------------------------------------------- scoring
def correct(answer: str, q: dict[str, Any]) -> bool:
    a = answer.lower()
    if q.get("answer_regex") and re.search(q["answer_regex"], answer, re.I):
        return True
    kws = [k.lower() for k in q.get("answer_keywords", [])]
    return bool(kws) and all(k in a for k in kws[:1]) and any(k in a for k in kws)


def refused(answer: str) -> bool:
    return REFUSAL.lower() in answer.lower() or bool(REFUSAL_RE.search(answer))


def basename_hits(sources: list[dict[str, Any]], expected: list[str]) -> bool:
    names = {Path(e).name for e in expected}
    return any(s["filename"] in names for s in sources)


# --------------------------------------------------------------------------- turns
def turn(rig: Rig, preset: ModelPreset, index: IndexHandle, query: str, *, history: list[dict[str, Any]] | None = None, attachment: Path | None = None,
         hist: HistoryConfig | None = None, max_tokens: int = 192) -> dict[str, Any]:  # fmt: skip
    sampling = load_sampling_presets()
    eff = resolve(sampling, preset.sampling_preset, overrides={"max_tokens": max_tokens, "reasoning": False}, family=preset.family)
    att = None
    if attachment is not None:
        data = attachment.read_bytes()
        mod = "image" if attachment.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp") else "audio"
        att = Attachment(mod, data, "image/png" if mod == "image" else "audio/wav", attachment.name)
    pipe = ChatPipeline(rig.client, rig.governor, EmbedService(rig.client, load_embed_params()), load_embed_params(), rig.telemetry, rig.log)
    req = TurnRequest(sid="eval", query=query, history=history or [], preset=preset, effective=eff, history_cfg=hist or HistoryConfig("window", 4, "off"),
                      retrieval=RetrievalConfig(k=4), attachment=att)  # fmt: skip
    t = pipe.turn(req, index)
    with Sampler(rig.meter) as s:
        answer = "".join(t.stream())
    return {"answer": answer, "stats": t.stats, "sources": t.source_rows(), "peak_mb": s.peak, "error": t.error.user_message if t.error else None}


# --------------------------------------------------------------------------- gates
@dataclass
class Result:
    candidate: str
    gates: dict[str, str | None] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    verdict: str = ""


def pct(mb: float) -> float:
    return round(100 * mb / CAP, 1)


def evaluate(rig: Rig, p: ModelPreset, index: IndexHandle, ds: Path, max_queries: int) -> Result:
    r = Result(p.id)
    files = [MODELS_ROOT / f for f in p.files.values()]
    r.metrics["files_mb"] = round(sum(f.stat().st_size for f in files if f.exists()) / 1e6, 1)
    # ---------------- G0: load inside the cap (cold page cache)
    rig.governor.unload_all(keep=(ALWAYS_RESIDENT,))
    evict_page_cache(files)
    try:
        with rig.governor.turn("eval", "G0"), Sampler(rig.meter) as s:
            t0 = time.time()
            rig.governor.ensure(p.id)
            rig.governor.release(p.id)
            load_cold_s = time.time() - t0
        r.metrics["load_cold_s"] = round(load_cold_s, 2)
        r.metrics["load_peak_mb"] = round(s.peak, 1)
        out = rig.client.chat(p.id, [{"role": "user", "content": "Reply with: ready"}], {"max_tokens": 16, "chat_template_kwargs": {"enable_thinking": False}})
        r.metrics["g0_reply"] = (out["choices"][0]["message"].get("content") or "")[:40]
        r.gates["G0"] = "pass"
    except AppError as exc:
        r.gates["G0"] = "fail"
        r.metrics["g0_error"] = f"{exc.user_message} | {exc.detail}"[:400]
        r.verdict = "rejected at G0"
        return r
    # ---------------- G1: memory per phase
    steady = rig.meter.used_mb()
    r.metrics["steady_mb"] = round(steady, 1)
    queries = load_jsonl(ds / "queries" / "queries.jsonl")
    text_q = next((q for q in queries if q["type"] == "single_modal" and not q.get("attachment")), None)
    img_q = next((q for q in queries if q.get("attachment") and str(q["attachment"]).lower().endswith((".png", ".jpg", ".jpeg"))), None)
    phase_peaks = {}
    if text_q:
        tr = turn(rig, p, index, text_q["text"])
        phase_peaks["text_turn"] = tr["peak_mb"]
    if img_q:
        ir = turn(rig, p, index, img_q["text"], attachment=ds / img_q["attachment"])
        phase_peaks["image_turn_with_swap"] = ir["peak_mb"]
        r.metrics["image_turn_error"] = ir["error"]
    r.metrics["phase_peaks_mb"] = {k: round(v, 1) for k, v in phase_peaks.items()}
    peak = max([steady, r.metrics["load_peak_mb"], *phase_peaks.values()])
    r.metrics["peak_mb"] = round(peak, 1)
    r.metrics["steady_pct"], r.metrics["peak_pct"] = pct(steady), pct(peak)
    g1_ok = pct(steady) <= GATES["G1"]["steady_max_pct"] and pct(peak) <= GATES["G1"]["peak_max_pct"]
    r.gates["G1"] = "pass" if g1_ok else "fail"
    if not g1_ok:
        r.verdict = "rejected at G1"
        return r
    # ---------------- G5 quick: load/unload cycles drift
    rss_loaded = []
    t0 = time.time()
    for _ in range(GATES["G5"]["cycles"]):
        with rig.governor.turn("eval", "G5"):
            rig.governor.unload(p.id)
            rig.governor.ensure(p.id)
            rig.governor.release(p.id)
        rss_loaded.append(rig.meter.used_mb())
    drift = 100 * (rss_loaded[-1] - rss_loaded[0]) / rss_loaded[0]
    r.metrics["g5_cycles"] = len(rss_loaded)
    r.metrics["g5_drift_pct"] = round(drift, 2)
    r.metrics["load_warm_s_avg"] = round((time.time() - t0) / len(rss_loaded), 2)
    r.gates["G5q"] = "pass" if abs(drift) <= GATES["G5"]["rss_drift_max_pct"] else "fail"
    # ---------------- G2 + G3: latency and grounded accuracy
    sample = stratified(queries, max_queries)
    rows = []
    for q in sample:
        att = ds / q["attachment"] if q.get("attachment") else None
        if att is not None and att.suffix.lower() not in (".png", ".jpg", ".jpeg", ".wav", ".mp3"):
            continue
        res = turn(rig, p, index, q["text"], attachment=att)
        rows.append({"id": q["id"], "type": q["type"], "lang": q.get("lang", "en"), "must_refuse": q.get("must_refuse", False), "correct": correct(res["answer"], q),
                     "refused": refused(res["answer"]), "retrieved": basename_hits(res["sources"], q.get("expected_sources", [])),
                     "ttft_ms": res["stats"].get("ttft_ms"), "decode": res["stats"].get("decode_tok_s"), "total_ms": res["stats"].get("total_ms"),
                     "peak_mb": round(res["peak_mb"], 1), "answer": res["answer"][:240], "error": res["error"]})  # fmt: skip
    pos = [x for x in rows if not x["must_refuse"]]
    neg = [x for x in rows if x["must_refuse"]]
    ttfts = sorted(x["ttft_ms"] for x in rows if x["ttft_ms"])
    p95 = ttfts[min(len(ttfts) - 1, int(0.95 * len(ttfts)))] / 1000 if ttfts else None
    dec = [x["decode"] for x in rows if x["decode"]]
    r.metrics["queries"] = len(rows)
    r.metrics["accuracy"] = round(sum(x["correct"] for x in pos) / len(pos), 3) if pos else None
    r.metrics["refusal_rate"] = round(sum(x["refused"] for x in neg) / len(neg), 3) if neg else None
    r.metrics["false_refusals"] = sum(x["refused"] and not x["correct"] for x in pos)
    r.metrics["retrieval_hit@4"] = round(sum(x["retrieved"] for x in pos) / len(pos), 3) if pos else None
    r.metrics["by_type"] = {t: round(statistics.mean(x["correct"] for x in pos if x["type"] == t), 3) for t in sorted({x["type"] for x in pos})}
    r.metrics["ttft_p95_s"] = round(p95, 2) if p95 else None
    r.metrics["decode_tok_s_median"] = round(statistics.median(dec), 2) if dec else None
    r.metrics["turn_peak_mb"] = max((x["peak_mb"] for x in rows), default=None)
    r.gates["G2"] = "pass" if p95 is not None and p95 <= GATES["G2"]["ttft_p95_max_s"] else "soft-fail"
    r.metrics["examples"] = rows[:6]
    (ROOT / "experiments" / "_raw").mkdir(parents=True, exist_ok=True)
    (ROOT / "experiments" / "_raw" / f"g3-{p.id}.json").write_text(json.dumps(rows, indent=2))
    # ---------------- G4: multi-turn coherence
    dialogues = load_jsonl(ds / "dialogues" / "dialogues.jsonl")[:4]
    for policy in ("window", "none"):
        ok = total = 0
        for d in dialogues:
            hist: list[dict[str, Any]] = []
            for t in d["turns"][:3]:
                res = turn(rig, p, index, t["user"], history=hist, hist=HistoryConfig(policy, 4, "off"))
                total += 1
                ok += int(correct(res["answer"], t))
                hist += [{"role": "user", "text": t["user"]}, {"role": "assistant", "text": res["answer"]}]
        r.metrics[f"g4_{policy}"] = round(ok / total, 3) if total else None
    acc = r.metrics["accuracy"] or 0.0
    r.gates["G3"] = "pass" if acc >= GATES["G3"]["accuracy_min_abs"] and (r.metrics["refusal_rate"] or 0) >= GATES["G3"]["refusal_min"] else "soft-fail"
    r.gates["G4"] = "pass" if (r.metrics.get("g4_window") or 0) >= GATES["G4"]["coherent_min"] else "soft-fail"
    r.verdict = "candidate"
    return r


def stratified(queries: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    by: dict[str, list[dict[str, Any]]] = {}
    for q in queries:
        by.setdefault(q["type"], []).append(q)
    out: list[dict[str, Any]] = []
    while len(out) < n and any(by.values()):
        for t in sorted(by):
            if by[t] and len(out) < n:
                out.append(by[t].pop(0))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", nargs="*", default=None)
    ap.add_argument("--ladder", action="store_true")
    ap.add_argument("--server", type=Path, default=Path("/rt/llama-server"))
    ap.add_argument("--dataset", type=Path, default=DATASETS_DIR / "synth_v1")
    ap.add_argument("--max-queries", type=int, default=36)
    ap.add_argument("--max-frames", type=int, default=4)
    ap.add_argument("--include", nargs="*", default=["text", "images", "audio", "video"])
    ap.add_argument("--workdir", type=Path, default=ROOT / "experiments" / "_raw" / "eval")
    args = ap.parse_args(argv)
    args.workdir.mkdir(parents=True, exist_ok=True)
    cands = [
        p
        for p in all_candidate_presets(args.ladder)
        if args.candidates is None or p.id in args.candidates or any(p.id.startswith(c + "-") for c in args.candidates)
    ]
    cands = [p for p in cands if all((MODELS_ROOT / f).exists() for f in p.files.values())]
    print("candidates:", [p.id for p in cands], flush=True)
    rig = Rig.start(cands, args.server, args.workdir)
    results = []
    try:
        index = index_dataset(rig, args.dataset, args.workdir, max_frames=args.max_frames, include=set(args.include))
        print("index:", index.stats(), flush=True)
        rig.governor.unload(MM_EMBEDDER)
        for p in cands:
            print(f"== {p.id}", flush=True)
            try:
                r = evaluate(rig, p, index, args.dataset, args.max_queries)
            except Exception as exc:
                r = Result(
                    p.id, gates={"G0": "error"}, metrics={"error": f"{type(exc).__name__}: {exc}", "tb": traceback.format_exc()[-1200:]}, verdict="error"
                )
            finally:
                try:
                    rig.governor.unload_all(keep=(ALWAYS_RESIDENT,))
                except Exception:
                    pass
            append("S12-gates", candidate_id=p.id, env=env(), params={"n_ctx": p.n_ctx, "kv": p.kv.model_dump(), "files": p.files, "max_queries": args.max_queries},
                   metrics={k: v for k, v in r.metrics.items() if k != "examples"}, gates=r.gates, verdict=r.verdict)  # fmt: skip
            results.append(r)
            print(
                json.dumps(
                    {
                        "id": r.candidate,
                        "gates": r.gates,
                        "verdict": r.verdict,
                        **{
                            k: r.metrics.get(k)
                            for k in (
                                "steady_pct",
                                "peak_pct",
                                "accuracy",
                                "refusal_rate",
                                "retrieval_hit@4",
                                "ttft_p95_s",
                                "decode_tok_s_median",
                                "g4_window",
                                "g4_none",
                                "load_cold_s",
                                "error",
                                "g0_error",
                            )
                        },
                    }
                ),
                flush=True,
            )
    finally:
        rig.stop()
    out = ROOT / "experiments" / "_raw" / f"eval-{int(time.time())}.json"
    out.write_text(json.dumps([r.__dict__ for r in results], indent=2, default=str))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
