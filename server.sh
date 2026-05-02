#!/usr/bin/env bash
# server.sh — reference llama-server launch for the benchmark.
#
# IMPORTANT for cross-machine comparison: every contributor should
# run with these exact flags so the bench measures the same thing.
# If you need to deviate (e.g. less RAM), note the change in the
# --label of your bench run.
#
# Defaults below target a 64 GB Apple Silicon Mac. For 32 GB you
# will need a smaller model (e.g. Q3_K_M variant); the 35B-A3B
# Q4_K_XL + 32K context will not fit.

set -euo pipefail

MODEL="${MODEL:-$HOME/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf}"
PORT="${PORT:-8001}"
ALIAS="${ALIAS:-qwen3.6-35b-a3b}"
CTX="${CTX:-32768}"

[[ -f "$MODEL" ]] || { echo "ERROR: model not found at $MODEL"; echo "Set MODEL=/path/to/file.gguf"; exit 1; }
command -v llama-server >/dev/null || { echo "ERROR: llama-server not found (brew install llama.cpp)"; exit 1; }

echo "starting llama-server"
echo "  model: $MODEL"
echo "  port:  $PORT"
echo "  ctx:   $CTX"

exec llama-server \
  --model "$MODEL" \
  --port "$PORT" \
  --alias "$ALIAS" \
  -c "$CTX" \
  -n 32768 \
  --no-context-shift \
  -ngl 999 \
  -fa on \
  -ctk q8_0 -ctv q8_0 \
  --temp 0.6 --top-p 0.95 --top-k 20 \
  --repeat-penalty 1.00 --presence-penalty 0.00 \
  --chat-template-kwargs '{"preserve_thinking": true}'
