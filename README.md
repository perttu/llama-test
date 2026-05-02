# llama-test

A small, reproducible benchmark for `llama-server` on Apple Silicon.
Built so two people running the same model on different chips can
compare numbers fairly.

The bench targets **Qwen3.6-35B-A3B (Q4_K_XL)** by default, but the
script doesn't care which model you use — the alias just has to match
the running server.

## What it measures

Three probes, each run multiple times. Prompt caching is **disabled**
in every request (`cache_prompt: false`), so each run is an honest
measurement.

| Probe | Bottleneck | Prompt | Generation |
| --- | --- | --- | --- |
| **P1 gen-speed** | memory bandwidth | tiny | 600 tokens |
| **P2 pre-fill** | GPU compute | ~6K tokens | 60 tokens |
| **P3 thinking-mode** | mixed (thinking + answer) | tiny | up to 2000 tokens |

For each probe we report **min / median / max** across runs so thermal
throttling and other variance are visible, not averaged away.

## Prerequisites

```bash
brew install llama.cpp jq
```

Plus the GGUF you want to test. For the default model:

```bash
mkdir -p ~/models
curl -L -C - -o ~/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \
  https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF/resolve/main/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf
```

(File is ~21 GB. Needs at least 28 GB free disk during download for
safety, and 64 GB unified memory to run with the default 32K context.)

## Run it

Two terminals:

```bash
# terminal 1 — start the server (blocks; ctrl-C to stop)
./server.sh

# terminal 2 — wait for the server to log "HTTP server listening", then:
./bench.sh --label "M2 Max 64GB"
```

Output:

- a markdown summary on stdout
- a JSON file in `runs/` named `<timestamp>_<label>.json`

## Common knobs

```bash
./bench.sh --runs 3                # fewer runs (faster)
./bench.sh --quick                 # 1 run each (~30s smoke test)
HOST=http://localhost:8011 ./bench.sh   # different port
MODEL=my-other-alias ./bench.sh    # different --alias

# server side
PORT=8011 ./server.sh
CTX=65536 ./server.sh              # bigger context (more KV memory)
MODEL=/path/to/other.gguf ./server.sh
```

## Sharing your results

1. Run `./bench.sh --label "<chip> <RAM>GB"`
2. Commit the resulting `runs/*.json`
3. Open a PR titled e.g. `runs: M1 Max 64GB`

We deliberately keep raw run data, not just summaries, so per-run
variance can be examined later.

## Why this benchmark and not `llama-bench`?

`llama-bench` measures the *engine* in isolation. This bench measures
the *server pipeline* — request parsing, sampler, quantized KV cache,
chat-template rendering, the lot — which is what real applications
actually pay for. Numbers from the two tools won't match and that's
expected.

## Comparability rules

For a comparison to be meaningful across machines:

- same `server.sh` flags (no edits — note any deviation in `--label`)
- same model file (same SHA — `shasum -a 256 ~/models/*.gguf`)
- same prompt files (don't edit `prompts/*.txt`)
- same `bench.sh` runs count
- laptop on AC power, lid open, no other heavy work running
- run twice; if numbers move >10% between runs, your machine has
  thermal headroom issues — note that in the run label
