#!/usr/bin/env bash
# server.sh — reference llama-server launch for the benchmark.
#
# Cross-machine comparison: every contributor should run with these
# exact flags so the bench measures the same thing. Note any deviation
# in the bench --label.
#
# Two runtimes:
#   RUNTIME=native (default) — calls `llama-server` from your PATH
#                              (brew install llama.cpp on macOS)
#   RUNTIME=docker          — runs ghcr.io/ggml-org/llama.cpp:server-cuda
#                              with --gpus all (Linux + nvidia-container-toolkit)
#
# Defaults target 64 GB unified memory or a 24 GB+ NVIDIA GPU.

set -euo pipefail

MODEL="${MODEL:-$HOME/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf}"
PORT="${PORT:-8001}"
ALIAS="${ALIAS:-qwen3.6-35b-a3b}"
CTX="${CTX:-32768}"
RUNTIME="${RUNTIME:-native}"
DOCKER_IMAGE="${DOCKER_IMAGE:-ghcr.io/ggml-org/llama.cpp:server-cuda}"
GPUS="${GPUS:-all}"   # docker --gpus value: 'all' or 'device=0' / 'device=0,1' etc.

[[ -f "$MODEL" ]] || { echo "ERROR: model not found at $MODEL"; echo "Set MODEL=/path/to/file.gguf"; exit 1; }

echo "starting llama-server  ($RUNTIME)"
echo "  model: $MODEL"
echo "  port:  $PORT"
echo "  ctx:   $CTX"

LLAMA_FLAGS=(
  --port "$PORT"
  --alias "$ALIAS"
  -c "$CTX"
  -n 32768
  --no-context-shift
  -ngl 999
  -fa on
  -ctk q8_0 -ctv q8_0
  --temp 0.6 --top-p 0.95 --top-k 20
  --repeat-penalty 1.00 --presence-penalty 0.00
  --chat-template-kwargs '{"preserve_thinking": true}'
)

case "$RUNTIME" in
  native)
    command -v llama-server >/dev/null || {
      echo "ERROR: llama-server not on PATH (brew install llama.cpp)"; exit 1; }
    exec llama-server --model "$MODEL" "${LLAMA_FLAGS[@]}"
    ;;
  docker)
    command -v docker >/dev/null || { echo "ERROR: docker not found"; exit 1; }
    MODEL_DIR=$(dirname "$MODEL")
    MODEL_NAME=$(basename "$MODEL")
    exec docker run --gpus "$GPUS" --rm \
      -v "$MODEL_DIR":/models \
      -p "$PORT":"$PORT" \
      "$DOCKER_IMAGE" \
      --model "/models/$MODEL_NAME" \
      --host 0.0.0.0 \
      "${LLAMA_FLAGS[@]}"
    ;;
  *)
    echo "ERROR: RUNTIME must be 'native' or 'docker' (got: $RUNTIME)"; exit 1
    ;;
esac
