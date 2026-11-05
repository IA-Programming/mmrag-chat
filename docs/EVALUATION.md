# Evaluation (how the shipped models were chosen)

Everything here was measured inside the **capped emulator** (`python:3.11-slim-bullseye`, `--memory=2700m --memory-swap=2700m --cpus=2 --cpuset-cpus=0,1`) with the Debian-11 llama.cpp build `b12119`. Every number is in [`experiments/ledger.jsonl`](../experiments/ledger.jsonl) (append-only), produced by [`tools/eval/run.py`](../tools/eval/run.py) and friends. Memory = llama-server process tree + 160 MB measured app overhead. Latency on the dev Xeon is *better* than Community Cloud's shared cores: use it to rank, not as a promise.

## Gates

| Gate | Check | Pass | Type |
|---|---|---|---|
| G0 | loads inside the cap, cold page cache | no OOM, ready ≤ 120 s | hard |
| G1 | steady state and peaks per phase (load, text turn, image turn with swap) | steady ≤ 80 %, peak ≤ 95 % of 2,700 MB | hard |
| G5q | 20 load/unload cycles | RSS drift ≤ 5 % | hard |
| G2 | TTFT p95, decode tok/s | TTFT p95 ≤ 15 s (dev CPU) | soft |
| G3 | answer accuracy (planted facts, keyword/regex) + refusal on negative questions | ≥ 60 %, refusal ≥ 80 % | soft, ranks |
| G4 | 3-turn dialogues with pronouns/ellipsis, `window(4)` vs `none` | ≥ 70 % | soft |

Dataset: [`datasets/synth_v1`](../datasets/synth_v1) — a fictional world (so nothing can be answered from pretraining) with 290 planted facts in 40 documents, 56 images, 28 spoken clips and 14 videos; 80 questions (single-modal, cross-modal text→image/audio/video and image/audio→text, negative, near-duplicate, Spanish/Japanese) and 10 dialogues.

<!-- EVAL-RESULTS -->

## Experiments that shaped the defaults

| ID | Question | Result |
|---|---|---|
| EXP-LOAD-MODE-01 | mmap vs no-mmap loading | mmap + Q4_0 repacking keeps the weights resident twice (2.10 GB for LFM3-VL-2B); `load-mode none` 1.34 GB at the same speed → used for every preset |
| EXP-EMB-01 | text chunk size × overlap (45 questions, k = 4) | 128 tokens / no overlap: the answer-bearing chunk is in the top-4 for 84 % of questions with 522 prompt tokens, vs 78 % with 991 tokens at the old 256/32; overlap always hurt |
| EXP-PROMPT-01 | where the user's attachment goes | after the sources LFM3-VL replied with a bare `[S1]`; first → full, correct sentences |
| EXP-PROMPT-02 | prompt wording A/B (8 questions × 4 variants) | current prompt 8/8, no bare citations → kept |
| EXP-UI-01 | `st.navigation` vs `st.tabs` | navigation: 1 vs 6 page bodies per rerun, p95 24 vs 51 ms → tabs deleted |
| EXP-GEMINI-01 | free-tier Gemini model for bring-your-own-key | `gemini-3.5-flash-lite`: text/image/audio/video accepted; `seed`, `top_k`, penalties rejected |
| prefill vs ubatch | 256 / 512 / 1024 | 49.7 / 52.3 / 53.4 tok/s (+110 MB at 1024) → 256 kept; prefill is the TTFT bottleneck on 2 cores |
| M-CAP-selftest | does the cap bite? | allocation child SIGKILLed after 2.6 GB → the platform cap kills, it does not throttle |
