# Development

## Setup and checks

```bash
python3.11 -m venv .venv && . .venv/bin/activate
make setup          # pip install -r requirements.txt -r requirements-dev.txt
make check          # lint + type + test + apptest + secretscan (what CI runs)
make run-fake       # the whole UI against tests/fakes/fake_llama_server.py (no models)
```

| Command | What it runs |
|---|---|
| `make lint` | `ruff check . && ruff format --check .` |
| `make type` | `mypy core app` (pydantic plugin) |
| `make test` | offline unit tests (`pytest -m "not needs_runtime and not memory and not eval"`) |
| `make apptest` | page flows with `streamlit.testing.v1.AppTest` + the fake llama-server |
| `make test-int` | integration tests against the real build (run inside the emulator, see below) |
| `make secretscan` | no key-shaped strings or secret files tracked |
| `make audit` | `pip-audit -r requirements.txt` |
| `make lock` | re-resolve `constraints.txt` (full runtime lock for Python 3.11 / manylinux_2_31) |

## The capped emulator (never trust a number measured outside it)

```bash
docker compose -f docker/compose.cap.yml run --rm selftest     # cap must SIGKILL near 2.7 GB and expose 2 CPUs
docker compose -f docker/compose.cap.yml up app                 # app at http://127.0.0.1:8501
```

`--cpus=2` alone still shows 4 cores, so the compose file also pins `cpuset: "0,1"` and the presets pass `-t 2`. Memory figures transfer to Cloud; latency figures do not (the dev Xeon is faster than Cloud's shared cores) — use them to rank.

## Building the runtime (DR-001)

```bash
python -m tools.llama.build_llama --tag b12119 --build-image      # Debian 11 + clang 18 container
```

Produces `build/llama/pkg/` (`llama-server`, shared libs, 14 CPU-variant backends) and `build-info.json`. The script fails if any ELF needs GLIBC > 2.31, references GLIBCXX, or links a non-system library. Smoke tests (`tools/llama/smoke_tests.py`, stdlib only) run in `python:3.11-slim-bullseye` under the cap and check: version, router health, text embedding, **two different images give different vectors** (the silent-ignore regression), flat parts rejected, chat, vision chat, unload frees memory.

Patch queue policy: a llama.cpp PR is applied only if it maps to a requirement, has a test that fails without it, is pinned by head SHA, and is re-checked on every rebuild. Currently empty.

## Models

```bash
HF_TOKEN=... python -m tools.fetch_models --only embeddinggemma-2 lfm3-vl-2b-qad-q4_0   # pinned by repo revision, SHA-256 checked
python -m tools.verify_sampling_presets            # compare bundle/sampling_presets.json with the live model cards
```

`tools/eval/candidates.yaml` lists every candidate with its exact files and revision. `HF_TOKEN` is read from the environment and never printed.

## Evaluating candidates (S12)

```bash
docker run --rm --memory=2700m --memory-swap=2700m --cpus=2 --cpuset-cpus=0,1 \
  -v $PWD:/app -w /app -v ~/.cache/mmrag/models:/models:ro -v $PWD/build/llama/pkg:/rt:ro \
  mmrag-runtime:bullseye python -m tools.eval.run --candidates lfm3-vl-2b-qad-q4_0 qwen3.5-0.8b-q4_k_m
```

Gates (thresholds in `tools/eval/gates.yaml`), cheapest first: **G0** loads inside the cap; **G1** steady ≤ 80 % and peak ≤ 95 % of 2.7 GB (memory = llama-server process tree + 160 MB measured app overhead); **G5q** 20 load/unload cycles without RSS drift; **G2** TTFT/decode; **G3** grounded accuracy + refusal on the synthetic dataset; **G4** multi-turn coherence with `window(4)` vs `none`. Every run appends to `experiments/ledger.jsonl` (append-only). Results: [EVALUATION.md](EVALUATION.md).

## Publishing a bundle (DR-002)

1. Mark the chosen generators `shipped: true` in `bundle/model_presets.json`.
2. Edit `bundle/release-request.json` (`bundle_id`, `llama_tag`, `generators`, `candidates`) and push: `.github/workflows/bundle-release.yml` builds the runtime on Debian 11, smoke-tests it under the cap, fetches the pinned models and publishes everything as a GitHub release.
3. Copy the release's `manifest.json` into `bundle/manifest.json` and commit — the deployed app then downloads exactly those bytes.

For local testing without a release: `python -m tools.release --bundle-id dev --runtime-dir build/llama/pkg --out build/bundle --link --manifest-path build/bundle/manifest.json`, then start the app with `MMRAG_BUNDLE_SOURCE_DIR=build/bundle MMRAG_MANIFEST=build/bundle/manifest.json`.

## Synthetic dataset

`python -m tools.dataset.generate --seed 7 --out datasets/synth_v1` regenerates `datasets/synth_v1/` byte-for-byte (needs `espeak-ng` for the spoken clips). A fictional world with 290 planted facts across 40 documents, 56 images, 28 spoken clips and 14 videos; 80 queries (single-modal, cross-modal, negative, near-duplicate, multilingual) and 10 dialogues.

## Debugging a deployment

Testers can download a **field report** (Logs page): config without secrets, versions, manifest hash, environment probe and their own session events — no media, no keys. `python -m tools.replay field-report.zip` prints the reported configuration, the probe differences between Cloud and your machine, and the last warnings/errors.

## Conventions

* `core/` raises `AppError(user_message)`; pages show `user_message` (plus an event id), never a traceback.
* Session state only through `app/state.py`; widget keys `w_<page>_<name>`.
* No heavy work in the render path: model loads only from explicit actions or workers, always through the governor.
* Every new pipeline stage logs `<event>.start` / `<event>.end` with duration via `Logger.span`.
* Measure, then decide: append to the ledger, never edit past entries.
