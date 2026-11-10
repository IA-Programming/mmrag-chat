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

## Result

**Shipped: `lfm3-vl-2b-qad-q4_0`** (LFM3-VL-2B, quantization-aware Q4_0, Q8_0 projector, 8K context) — best answer accuracy (tied), perfect refusal on unanswerable questions, the smallest memory of the 2B models, and it can *see* the retrieved images and video frames. With the text embedder resident the whole app chats at ~2.0 GB (73 % of the cap) and peaks at 2.06 GB (76 %) during an image question (generator → multimodal embedder → generator swap).

* **Latency on 2 pinned cores:** text questions 13–15 s to the first token, image/video questions ~15 s median (p95 17.6 s after capping images at 128 tokens, EXP-VIS-01), then ~11 tokens/s. Prefill (~50 tokens/s) is the bottleneck — that is why chunks are 128 tokens (EXP-EMB-01).
* **Close alternatives** (one-line change in `bundle/model_presets.json`): `lfm3-1.7b-qad-q4_0` (same accuracy, text-only, TTFT p95 14 s, 66 %), `lfm3-omni-2b-q4_0` (the only one that *hears* voice questions, but answered 2 of 5 unanswerable questions anyway), `qwen3.5-0.8b-q4_k_m` (fastest decoding, 58 %, lower accuracy).
* **Memory-wise impossible here:** MiniCPM-V-4.7-1B (its smallest projector is f16), Gemma-4-E2B, and Ministral-3-3B except its 2-bit text-only quant (see below).

### Quality and latency (candidates that passed the hard gates)

| Candidate | Inputs | Chat memory (text turn) | Image turn + swap | Answer accuracy | Refusal on negatives | Retrieval hit@4 | TTFT p95 | Decode | Multi-turn window(4) / none |
|---|---|---|---|---|---|---|---|---|---|
| **lfm3-vl-2b-qad-q4_0 (shipped)** | text, image | 1971 MB (73 %) | 2058 MB (76 %) | 0.68 | 1.0 | 0.90 | 28.8 s | 11.3 tok/s | 0.42 / 0.42 |
| lfm3-1.7b-qad-q4_0 | text | 1781 MB (66 %) | 1781 MB (66 %) | 0.68 | 1.0 | 0.90 | 14.4 s | 11.7 tok/s | 0.42 / 0.33 |
| lfm3-omni-2b-q4_0 | text, image, audio | 2067 MB (77 %) | 2154 MB (80 %) | 0.68 | 0.6 | 0.90 | 29.2 s | 11.7 tok/s | 0.42 / 0.25 |
| qwen3.5-2b-q4_k_m-text | text | 2094 MB (78 %) | 2094 MB (78 %) | 0.63 | 1.0 | 0.90 | 10.6 s | 9.5 tok/s | 0.42 / 0.42 |
| lfm3-vl-2b-q4_k_m | text, image | 2022 MB (75 %) | 2111 MB (78 %) | 0.58 | 0.8 | 0.90 | 26.2 s | 11.3 tok/s | 0.42 / 0.25 |
| lfm2.5-vl-1.6b-q4_0 | text, image | 1964 MB (73 %) | 2027 MB (75 %) | 0.58 | 0.4 | 0.90 | 31.8 s | 16.5 tok/s | 0.42 / 0.33 |
| qwen3.5-0.8b-q4_k_m | text, image | 1564 MB (58 %) | 1602 MB (59 %) | 0.58 | 0.8 | 0.90 | 15.3 s | 18.2 tok/s | 0.42 / 0.42 |

TTFT p95 above was measured before the 128-token image cap (EXP-VIS-01); text-only questions were 13–15 s for every 2B model. Accuracy counts answerable questions (keyword/regex on the planted fact); refusal counts unanswerable ones. 24 stratified questions + 4 dialogues per candidate.

### Hard-gate rejections (numbers recorded, models not shipped)

| Candidate | Why |
|---|---|
| minicpm-v-4.7-1b-q4_k_m | rejected at G1: steady 86.0 % / peak 91.8 % of 2,700 MB |
| ministral-3-3b-q4_k_m | rejected at G0: 'ministral-3-3b-q4_k_m' does not fit in memory right now (needs ~3238 MB, 202 MB in use, budget 2565 MB). |
| gemma-4-e2b-q4_0 | rejected at G0: 'gemma-4-e2b-q4_0' does not fit in memory right now (needs ~3648 MB, 202 MB in use, budget 2565 MB). |
| ministral-3-3b-iq4_xs-text | rejected at G1: steady 93.2 % / peak 93.3 % of 2,700 MB |
| ministral-3-3b-q3_k_m-text | rejected at G1: steady 87.4 % / peak 87.4 % of 2,700 MB |
| ministral-3-3b-q2_k-text | first quant that fits (text-only): steady 74.8 %, peak 86.5 % - not evaluated further (2-bit) |
| ministral-3-3b-iq2_m-text | rejected at G1: steady 81.9 % / peak 82.0 % of 2,700 MB |

<!-- EVAL-RESULTS-END -->

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
| EXP-VIS-01 | generator image-token cap (23 cross-modal questions) | default 0.48 acc / p95 30.5 s; 128 tokens 0.43 / 17.6 s; 64 tokens 0.43 / 14.8 s → 128 shipped |
| EXP-HIST-01 | history x contextualization (10 dialogues, 33 turns) | window(4) + `llm_rewrite`: follow-up answers 0.48 → 0.61, follow-up retrieval 0.57 → 1.00, +2.7 s and one extra call per follow-up → default; `concat_last_user` fixes retrieval (0.83) but not answers |
| EXP-PROMPT-03 | "the question may be in another language" | no change on 9 Spanish/Japanese questions (5/9 both) → not added |
| M-CAP-selftest | does the cap bite? | allocation child SIGKILLed after 2.6 GB → the platform cap kills, it does not throttle |
