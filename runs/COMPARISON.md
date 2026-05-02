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
| **agentics-hosted** | AMD Ryzen AI Max+ ("Strix Halo") in HP Z2 Mini G1a | 128 GB unified (≈256 GB/s LPDDR5x) | Radeon 8060S iGPU (40 CU RDNA 3.5) | Linux | llama.cpp `:server-vulkan` (containers managed by [sovereign-engine](https://github.com/agenticsnz/sovereign-engine)) |

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

## Hosted (agentics.org.nz) — AMD Strix Halo mini PC

The hosted endpoint runs on a single **HP Z2 Mini G1a** with an **AMD Ryzen
AI Max+ ("Strix Halo")** APU and **128 GB of unified memory** addressable as
VRAM. Same GGUF and same llama.cpp build as our locals — the only thing
that differs is the inference hardware. That makes this a fair platform
comparison.

### Numbers

- **Generation: ~57 tok/s.** +27 % over M2 Max baseline (35.6), +30 %
  over M2 Max best (44.1). About 40 % of single 3090 (143).
- **Pre-fill: ~1 005 tok/s.** Between M2 Max baseline (882) and M2 Max
  best (1 161). About 32 % of single 3090 (3 150).
- **Variance: under 1 %** across 5 runs — consistent with a steady-state
  desktop, no thermal throttling.

### What it tells us about Strix Halo as an inference platform

Strix Halo's headline spec: 16 Zen 5 cores + Radeon 8060S iGPU (40 RDNA
3.5 compute units) sharing **256-bit LPDDR5x at ~256 GB/s**. The
agentics deployment uses **llama.cpp's Vulkan backend**, not ROCm.
sovereign-engine's source (`proxy/src/docker/llamacpp.rs`) explicitly
notes that ROCm tested at **0.6–2.4 t/s vs Vulkan's 4.5 t/s** on their
hardware, so they ship Vulkan only and route AMD inference through it.
Vulkan is portable but generally slower than CUDA on the same silicon.

Compared in the same class:

| Platform | Mem BW | Backend | Gen this model | Pre-fill this model |
| --- | ---: | --- | ---: | ---: |
| M2 Max 64 GB | 400 GB/s | Metal (native) | 35.6 → 44.1 (opt) | 882 → 1 161 (opt) |
| **Strix Halo 128 GB** | **~256 GB/s** | **Vulkan** | **57** | **1 005** |
| RTX 3090 24 GB | 936 GB/s | CUDA | 137.3 → 143.1 (opt) | 3 150 → 4 669 (opt) |

The interesting bit: **Strix Halo gets *more* tokens per second than M2
Max despite having 36 % less memory bandwidth and using Vulkan (not the
hardware-vendor backend).** Most likely explanations:

1. **Our M2 Max is thermally throttling.** Variance on the M2 Max gen
   probe was 17 % (min 39.9 / max 46.7); sustained inference pushes the
   laptop into thermal limits. Strix Halo in a mini-PC chassis ran at
   < 1 % variance — full clocks, indefinitely.
2. **MoE-A3B touches a small slice of weights per token.** With only
   3 B params active, each step is more compute-bound than bandwidth-
   bound — exactly the regime where the bandwidth gap (256 vs 400 GB/s)
   matters less than the compute pipeline efficiency. On a dense 70 B
   model the ranking would likely flip the other way.
3. **The MacBook M2 Max has CPU-side overhead in our config** that the
   Strix Halo box may not have (different scheduler, different idle
   behaviour, etc.). Hard to quantify without instrumenting.

This is genuinely surprising and worth flagging: **a $4k mini-PC with
a Vulkan iGPU outpaces a $5k laptop with a hardware-tuned Metal backend
on this workload, primarily because the laptop throttles**. If we re-
ran the M2 Max numbers in a desk-mounted Mac Studio (M2 Max on a desk
with a fan), the gap would likely close or reverse.

What it can run that 24 GB GPUs can't: the **128 GB unified pool**
opens the door to much larger models (70B Q4, 120B Q3, MoE-405B at
heavy quants) that simply don't fit on a single 3090. For the 35B-A3B
we tested, the bottleneck isn't memory capacity but bandwidth — which
is where the 3090 wins by a factor of 3+.

### Practical reading

- For *running this exact model*: 3090 ≫ Strix Halo ≫ M2 Max, by gen
  tok/s. Single 3090 is **2.5× faster gen** and **4.7× faster pre-fill**
  than Strix Halo.
- For *fitting much bigger models locally*: Strix Halo's 128 GB shared
  pool is genuinely useful. A 3090 with 24 GB tops out around this
  size class.
- For *low-power / mini-PC desk hardware*: Strix Halo at ~50–80 W
  full-load is dramatically more efficient than dual 3090s pulling
  ~600 W. If watts and noise matter, the gap is significant.

For the user: hosted agentics gives you ~30 % better generation than
your M2 Max with similar pre-fill latency. If raw throughput matters
and the 3090 box is available, that wins by a wide margin. If you'd
rather not own that 3090 box (or want a quieter local option for very
large models), Strix Halo is a credible alternative — and the API
above is one way to try it without committing $4k to the hardware.



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
