# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A cross-platform single-stream throughput benchmark for `llama-server` (and any
OpenAI-compatible chat-completions endpoint). Bash + `jq` + `curl`, no Python.
Designed so two people on different hardware (macOS + Metal, Linux + CUDA,
Linux + Vulkan, hosted APIs like agentics/Kimi/OpenRouter) can compare numbers
fairly using identical prompts and identical server flags.

It is deliberately **not** a substitute for:
- `vllm/benchmarks/benchmark_serving.py` (concurrency, P99 latency, ShareGPT distribution)
- `LLMPerf` (production load testing)
- `promptfoo` / `lm-evaluation-harness` (quality / capability evals)

If a request asks for "concurrent users", "P99 TTFT", "ShareGPT", or
"is the model good at X" — the answer is **add a separate tool, do not extend
this one**. The simplicity is the feature.

## Common commands

```bash
# Start a local server with the standardized flags
./server.sh                                          # macOS native
RUNTIME=docker ./server.sh                           # Linux + NVIDIA via Docker
RUNTIME=docker GPUS=device=0 ./server.sh             # single GPU only
CTX=16384 RUNTIME=docker ./server.sh                 # smaller context for tight VRAM

# Run the benchmark (wait for server to log "HTTP server listening" first)
./bench.sh --label "M2 Max 64GB"                     # 5 runs per probe (default)
./bench.sh --quick --label "smoke"                   # 1 run per probe (~30s)
./bench.sh --runs 3 --label "..."                    # custom run count
HOST=http://localhost:8011 ./bench.sh                # different port

# Run against a hosted OpenAI-compatible endpoint (.env contains API_KEY=...)
set -a && source .env && set +a
HOST=https://api.example.com MODEL=org/model-name ./bench.sh --label "hosted"

# Stop a server
lsof -ti :8001 | xargs -r kill                       # local
docker ps | grep llama && docker kill <id>           # docker
```

There are no tests, no lint, no build step. Edits to `bench.sh` should be
verified with `./bench.sh --quick --label smoke-test` then `rm runs/*smoke*`.

## Architecture

Three pieces, each does one thing:

1. **`server.sh`** — opinionated `llama-server` launcher. Two runtimes (`native`
   and `docker`) plus `GPUS` env for multi-GPU selection. Flags here are the
   *standardized baseline* — anyone running this script across machines is
   testing the same configuration. Don't edit casually; document deviations
   in the `--label` of a bench run instead.

2. **`bench.sh`** — runs three fixed probes (P1 short-prompt generation, P2
   long-prompt pre-fill, P3 thinking-mode reasoning) N times each, with
   prompt caching **disabled** (`cache_prompt: false`), reports min/median/max
   per probe, and writes a JSON file to `runs/`.

3. **`prompts/p{1,2,3}.txt`** — byte-identical prompt fixtures committed to
   git. **Do not edit these** — changes invalidate every comparison anyone
   has done before. P2 specifically is sized to ~5870 tokens for the Qwen3
   tokenizer; the count appears as `prompt_n` in results and is the proof
   that pre-fill actually ran on every probe.

`runs/` holds completed result JSONs (one per machine × config) and a
`COMPARISON.md` doc that's updated by hand as new rows arrive.

## Non-obvious things that bit us, that will bit you again

- **Prompt caching is the #1 footgun.** If `prompt_n` in the result equals
  the prompt's actual token count on every run (e.g. 5870 for P2), caching
  is properly disabled. If `prompt_n` drops to single digits on runs 2–5,
  llama.cpp is reusing cached prefixes and the throughput numbers are
  inflated by 10–100×. The bench sets `cache_prompt: false` per request
  precisely to defeat this — keep it that way.

- **min/median/max isn't decoration.** Apple Silicon laptops thermally
  throttle under sustained inference (we measured 17 % spread on M2 Max
  vs <1 % on a desktop mini-PC and <0.5 % on a 3090). Reporting the mean
  alone would hide this. Don't switch to mean.

- **Locale matters.** `printf '%.1f' "36.299"` errors out under
  `LC_NUMERIC=fi_FI` (comma-decimal). The script forces `LC_ALL=C`; do
  not remove it.

- **BSD awk has no `asort`.** macOS ships BSD awk; the aggregation pipeline
  uses external `sort -n` instead of in-awk sorting for portability. Don't
  rewrite the aggregation in a more "elegant" gawk-idiomatic way unless
  you require gawk explicitly.

- **`${AUTH_HDR[@]+"${AUTH_HDR[@]}"}` is the correct idiom**, not
  `"${AUTH_HDR[@]}"`. Under `set -u`, expanding an empty array is an error;
  the `+` form expands to nothing if the array is empty/unset. Don't
  "simplify" it.

- **The HOST_TYPE remote-clearing block must run AFTER the local-hardware
  fingerprint scan.** If it runs before, the fingerprint detection
  overwrites the cleared values and the JSON gets the *client* hardware
  recorded as if it were the server. We hit this exact bug; preserve the
  ordering.

- **`HOST` should NOT include `/v1`.** The bench appends `/v1/models` and
  `/v1/chat/completions` itself. Hosted endpoints are typically advertised
  as `https://api.x.com/v1`, but you must pass `https://api.x.com`.

- **`.env` exists but is gitignored.** It holds `API_KEY=...` for hosted
  benches. The original `.gitignore` did not exclude it; an early commit
  could easily have leaked the key. The current `.gitignore` covers it.
  Verify before any `git add -A`.

- **The Docker `--gpus` flag depends on a working CDI spec.** On Ubuntu
  with NVIDIA, `/etc/cdi/nvidia.yaml` must exist and match the loaded
  driver version. `nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml`
  regenerates it; needed after any nvidia-driver upgrade or reinstall.

## Key findings from results in `runs/`

Future you may want to reference these without re-running the sweeps:

- **M2 Max best config**: `-ctk f16 -ctv f16 -ub 2048 -t 8` (+31 % pre-fill,
  +24 % gen vs baseline).
- **3090 best config**: `-ub 2048` is the single biggest lever (+47 %
  pre-fill); f16 KV adds another +5 % gen. Dual-GPU adds ~16 % pre-fill
  over optimized single, zero gen improvement.
- **Speculative decoding** with `Qwen3-0.6B` as draft for `Qwen3.6-35B-A3B`
  *regresses* generation 43–65 % because the vocabularies don't match
  (cross-generation Qwen). Needs a same-tokenizer draft model. Don't
  re-test until `Qwen3.6-0.6B-GGUF` is publicly available.
- **Strix Halo / agentics** outperforms laptop M2 Max despite lower
  bandwidth, primarily because the M2 Max thermal-throttles. The agentics
  endpoint runs `llama.cpp:server-vulkan` (not ROCm or CUDA) per the
  upstream `sovereign-engine` source.

## When committing results

Pulling a result from another machine and adding it to the comparison is
a routine task:

```bash
scp host:llama-bench/runs/<filename>.json runs/
git add runs/<filename>.json
git commit -m "<machine-tag> baseline / config description"
git push
```

Then update the relevant table in `runs/COMPARISON.md` by hand. Don't
auto-generate that doc — the analysis prose around each table is the
useful part.
