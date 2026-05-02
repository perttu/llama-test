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

## Results (median of 5 runs, prompt caching disabled)

| Probe | Metric | M2 Max | 3090 ×1 | 3090 ×2 |
| --- | --- | --- | --- | --- |
| P1 — short prompt, 600-tok generation | gen tok/s | 41.3 | 143.9 | 144.6 |
| P2 — 5870-tok prompt, 60-tok generation | pre-fill tok/s | 882.6 | 3 150.8 | **4 729.8** |
| P2 | TTFT (ms) | 6 643 | 1 863 | **1 241** |
| P2 | gen tok/s | 35.6 | 137.3 | 139.3 |
| P3 — thinking-mode reasoning, ~2K-tok gen | pre-fill tok/s | 397.8 | 1 110.3 | 1 125.5 |
| P3 | gen tok/s | 42.4 | 141.6 | 143.5 |

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

## Methodology notes

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
