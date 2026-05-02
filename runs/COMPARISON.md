# Cross-machine comparison

Baseline runs of `Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf` using the standardized
`server.sh` flags (`-c 32768 -fa on -ctk q8_0 -ctv q8_0 -ngl 999`,
Qwen3-recommended sampler) and `bench.sh` with prompt caching disabled,
5 runs per probe.

## Hardware tested

| Tag | Chip | Memory | GPU | OS | Runtime |
| --- | --- | --- | --- | --- | --- |
| **M2 Max 64GB** | Apple M2 Max (8P+4E) | 64 GB unified (400 GB/s) | Metal (integrated) | macOS 25.4 | native (brew llama.cpp) |
| **3090 ×1** | Intel i9-13900K (32 cores) | 126 GB DDR + 24 GB VRAM | 1 × RTX 3090 (936 GB/s) | Ubuntu 24.04 | docker ggml-org/llama.cpp:server-cuda |
| **3090 ×2** | Intel i9-13900K (32 cores) | 126 GB DDR + 48 GB VRAM | 2 × RTX 3090 | Ubuntu 24.04 | same image, `--gpus all` |
| **agentics-hosted** | unknown (remote OpenAI-compatible API) | unknown | unknown | unknown | identifies as `llama.cpp version 9000` via `.timings` |

## Results — out-of-the-box baselines (median of 5 runs, caching off)

| Probe | Metric | M2 Max | 3090 ×1 | 3090 ×2 | agentics-hosted |
| --- | --- | --- | --- | --- | --- |
| P1 — short prompt, 600-tok gen | gen tok/s | 41.3 | 143.9 | 144.6 | 57.6 |
| P2 — 5870-tok prompt, 60-tok gen | pre-fill tok/s | 882.6 | 3 150.8 | 4 729.8 | 1 005.4 |
| P2 | TTFT (ms) | 6 643 | 1 863 | 1 241 | 5 839 |
| P2 | gen tok/s | 35.6 | 137.3 | 139.3 | 55.7 |
| P3 — thinking-mode reasoning | pre-fill tok/s | 397.8 | 1 110.3 | 1 125.5 | 373.3 |
| P3 | gen tok/s | 42.4 | 141.6 | 143.5 | 57.0 |

## Results — best optimized configs

| Probe | Metric | M2 Max f16+ub2048 | 3090 ×1 f16+ub2048 | 3090 ×2 f16+ub2048 |
| --- | --- | --- | --- | --- |
| P1 | gen tok/s | 42.3 | 146.7 | **148.4** |
| P2 | pre-fill tok/s | 1 160.5 | 4 669.3 | **5 430.6** |
| P2 | TTFT (ms) | 5 058 | 1 257 | **1 081** |
| P2 | gen tok/s | 44.1 | 143.1 | **145.4** |
| P3 | gen tok/s | 42.2 | 145.5 | **147.9** |

Optimization gain over baseline (P2 pre-fill): M2 Max **+31 %**,
3090 ×1 **+48 %**, 3090 ×2 **+15 %**.

## Key observations

### 1. Generation is bandwidth-bound; pre-fill is compute-bound

The numbers track the underlying hardware ratios closely:

- **Memory bandwidth**: 400 → 936 GB/s (single 3090) → 1 872 GB/s (theoretical dual)
  → generation tok/s: 41 → 144 → 145. The 3.5 × jump from M2 → 3090 matches
  the bandwidth ratio. The dual-GPU configuration buys almost nothing for
  generation because autoregressive decoding is fundamentally sequential.
- **GPU FP16 compute**: ~13.6 → ~36 → ~72 TFLOPS, plus Ampere has tensor cores
  → pre-fill tok/s: 883 → 3 151 → 4 730. That's a 3.6 × gain on a single 3090
  and 5.4 × on dual.

### 2. Variance is much lower on datacenter GPUs

| Config | P1 gen min / med / max | spread |
| --- | --- | --- |
| M2 Max | 39.9 / 41.3 / 46.7 | 17 % |
| 3090 ×1 | 143.5 / 143.9 / 144.1 | 0.4 % |
| 3090 ×2 | 144.1 / 144.6 / 144.6 | 0.3 % |

The M2 Max is showing thermal throttling under sustained load — exactly
why we report min/median/max instead of mean. Datacenter cooling collapses
the variance.

### 3. Dual GPU only helps long pre-fill

Pre-fill on a long prompt parallelizes across GPUs (1.50 × speedup on the
5 870-token P2). Generation does not (1.005 ×). Pre-fill on short prompts
(40–86 tokens in P1/P3) is too small to amortize splitting overhead and
shows no meaningful gain.

### 4. TTFT is where dual GPU shines

Long-prompt time-to-first-token (the user-visible "is it thinking?" pause):
6.6 s → 1.9 s → 1.2 s. RAG and document-QA workflows feel substantially
snappier on dual GPU; chat with short prompts feels identical.

## Hosted (agentics.org.nz) shape

The hosted endpoint at `https://api.agentics.org.nz/v1` runs the same
GGUF (`unsloth/Qwen3.6-35B-A3B-GGUF`) and identifies via the `.timings`
block as the same llama.cpp build 9000 we have locally. The numbers
suggest the underlying box sits roughly between an out-of-the-box M2
Max and an optimized one for pre-fill, with notably higher generation
throughput than M2 Max:

- **Generation:** ~57 tok/s, **+27 % over M2 Max baseline** (35.6) and
  **+30 % over M2 Max best** (44.1). Below 3090 single (143). Profile
  is consistent with an M-series Ultra-class memory bandwidth
  (~600–800 GB/s) or a different GPU class entirely.
- **Pre-fill:** ~1 005 tok/s, between M2 Max baseline (882) and M2 Max
  best (1 161). Far below 3090 single (3 150) and dual (4 730).
- **Variance:** under 1 % spread across 5 runs (gen 55.1–56.3). Tighter
  than the 3090 (`0.4 %`) and far tighter than the M2 Max under load
  (17 % thermal swing). Suggests well-cooled stable hardware.

What that combination plausibly is: **an Apple Silicon Ultra** (M2
Ultra at 800 GB/s gives roughly the right gen speed; pre-fill being
modest is consistent with no tensor cores), a single mid-range workstation
GPU (e.g. A4000 / A5000 class), or an Apple M3/M4 Max with sustained
load not throttling.

In practice for an end user calling this API: expect ~60 tokens/sec
for chat-style output, with a long-prompt first-token latency of ~6 s
(comparable to local M2 Max) — i.e. roughly "M-Series Mac in the cloud."



- All runs use the same GGUF (byte-identical, SHA verified at download).
- All runs use the same `server.sh` flags except for the runtime
  (`native` on Mac, `docker` on Linux) and `GPUS` selection.
- `bench.sh` disables prompt caching (`cache_prompt: false`) so each
  request actually pre-fills the prompt. The buggy first iteration of
  this benchmark accidentally cached prompt prefixes, giving 100 ×-too-fast
  numbers on runs 2–5; this version verifies via `prompt_n` in the
  response.
- 5 runs × 3 probes = 15 requests per machine. Reported numbers are the
  **median**; min and max are stored in the JSON for variance analysis.

## M2 Max optimization sweep

Six configurations tested against the M2 Max baseline.

| Config                                       | P1 gen | P2 prefill | P2 gen | P3 think gen |
| -------------------------------------------- | -----: | ---------: | -----: | -----------: |
| Baseline (q8 KV, `-ub` default 512)          | 41.3   | 882.6      | 35.6   | 42.4         |
| `-ub 1024 -t 8` (q8 KV)                      | 46.1   | 1 000.4 (+13 %) | 38.4 | 39.0         |
| `-ub 2048 -t 8` (q8 KV)                      | 45.9   | 1 054.7 (+19 %) | 34.5 | 38.8         |
| `-c 16384 -ub 1024 -t 8` (q8 KV)             | 40.4   | 1 000.9 (+13 %) | 36.8 | 38.7         |
| **`-ctk f16 -ctv f16 -ub 1024 -t 8`**        | 42.4   | 1 085.6 (+23 %) | **43.7 (+23 %)** | 42.1 |
| **`-ctk f16 -ctv f16 -ub 2048 -t 8`** ⭐     | 42.3   | **1 160.5 (+31 %)** | **44.1 (+24 %)** | 42.2 |
| `-ub 1024 -t 8` + spec-decode (Qwen3-0.6B)   | 40.0   | 986.4      | **21.9 ⬇** | 42.8 |

### Best M2 Max config: f16 KV + `-ub 2048`

The biggest surprise: **un-quantizing the KV cache (f16) was the
biggest single win on M2 Max**, contradicting the "smaller KV =
faster" intuition. Cumulative wins over the standardized baseline:

- pre-fill: **+31 %**  (882.6 → 1 160.5 tok/s)
- generation (long context): **+24 %**  (35.6 → 44.1 tok/s)
- TTFT on 5 K prompt: 6 643 ms → 5 058 ms (**−24 %**)

Why f16 beats q8 KV here: on Metal the dequantize-then-attend
kernel sequence has a per-token overhead that exceeds the bandwidth
savings from reading half as many bytes. q8 KV is still the right
choice on memory-constrained machines (e.g. 32 GB Mac), but at
64 GB the extra ~6–10 GB of KV memory is free.

### What didn't help

- **`-c 16384`** (smaller context). Pre-allocated KV size doesn't
  affect throughput, only memory budget. Use it to fit on smaller
  machines, not for speed.
- **`-ub 4096+`** wasn't tested; the 1024 → 2048 gain was already
  showing strong diminishing returns (+5 %).
- **Speculative decoding with Qwen3-0.6B**. 18.75 % acceptance on
  greedy probes at temp=0 — different-generation draft can't match
  Qwen3.6's argmax often enough. Drafts are rejected wholesale,
  costing a full verify pass for nothing. Would likely work with a
  same-family `Qwen3.6-0.6B` draft (currently gated).

### Recommended M2 Max flags (64 GB+)

```
-c 32768 -ub 2048 -t 8
-fa on -ctk f16 -ctv f16
-ngl 999 --no-context-shift
```

For 32 GB machines: keep `-ctk q8_0 -ctv q8_0` and use `-c 16384`.

## RTX 3090 optimization sweep

Same probes, single and dual GPU, identical sweep matrix to the M2 run.

| Config                                        | P1 gen | P2 prefill | P2 gen | P3 think gen |
| --------------------------------------------- | -----: | ---------: | -----: | -----------: |
| 1×3090 baseline (q8 KV, `-ub` default 512)    | 143.9  | 3 150.8    | 137.3  | 141.6        |
| 1×3090 `-ub 2048`                             | 143.2  | 4 646.5 (+47 %) | 137.0 | 141.3 |
| 1×3090 `-ctk/-ctv f16`                        | 147.8  | 3 179.4    | 143.8 (+5 %) | 145.9 |
| **1×3090 f16 KV + `-ub 2048`** ⭐             | 146.7  | **4 669.3 (+48 %)** | 143.1 | 145.5 |
| 2×3090 baseline                               | 144.6  | 4 729.8    | 139.3  | 143.5        |
| **2×3090 f16 KV + `-ub 2048`** ⭐⭐           | 148.4  | **5 430.6 (+72 % vs 1× baseline)** | **145.4** | **147.9** |
| 1×3090 + spec-decode (vocab-mismatch fallback) | ≈baseline | ≈baseline | ≈baseline | ≈baseline |
| 2×3090 + spec-decode (vocab translation engaged) | 100.3 | 4 770.0 | **48.4 ⬇** | 90.5 |

### What worked: `-ub 2048`

Same flag, dramatically bigger effect than on Metal: **+47 % pre-fill on
single 3090** (vs +19 % on M2 Max). Ampere tensor cores scale with batch
size much better than Metal's matmul kernels — the default `-ub 512` was
leaving most of the GPU's throughput on the table.

### What worked but only modestly: f16 KV

Single 3090 f16 KV: **+5 % gen** (vs M2's +14 %). The CUDA q8 dequantize
kernels are more efficient than Metal's, so the gain from skipping them
is smaller. Still a free win on a 24 GB-or-larger card.

### Dual GPU shrinks as a value-add when single is optimized

| Comparison | Pre-fill ratio |
| --- | ---: |
| Baseline 1× → baseline 2× | 1.50× |
| Optimized 1× → optimized 2× | 1.16× |

`-ub 2048` already saturates a single 3090 on this workload. Adding a
second card only buys +16 % more pre-fill (and zero generation gain),
because the marginal compute is not the bottleneck — PCIe cross-talk and
synchronization eat most of the theoretical 2×.

**Practical implication**: a single optimized 3090 is preferable to a
naive dual-3090 for inference of this size class. If you have two cards,
keep one for inference (optimized) and use the other for something else.

### What didn't work: speculative decoding

Same negative result as on M2, with the cause now fully understood:

- **Single GPU**: draft context allocation OOMs (target uses 22.2 / 24 GB).
  Server silently falls back to non-spec mode → numbers identical to
  baseline, but spec decode wasn't actually running.
- **Dual GPU**: draft loads on GPU 1, but the server logs
  `target and draft vocabs are not compatible - tokens will be translated
  between the two`. Token-level translation between Qwen3 and Qwen3.6
  vocabs is both lossy and slow → P2 gen drops 65 % (139.3 → 48.4 tok/s).

Spec decode requires a same-tokenizer-family draft. Until
`Qwen3.6-0.6B-GGUF` is publicly available, this lever is unavailable.

### Recommended 3090 flags

```
-c 32768 -ub 2048
-fa on -ctk f16 -ctv f16
-ngl 999 --no-context-shift
# --gpus device=0 in the docker run; dual GPU adds ~15 % prefill at best
```

## Adding a new column

```bash
git clone https://github.com/perttu/llama-test.git && cd llama-test
# install prerequisites — see README.md
./server.sh                 # native macOS / Linux
RUNTIME=docker ./server.sh  # Linux + NVIDIA via Docker
# in another terminal:
./bench.sh --label "<chip> <RAM>"
git add runs/*.json && git commit -m "<chip> baseline" && git push
```

Then update the tables above by reading the new `runs/*.json`.
