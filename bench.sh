#!/usr/bin/env bash
# bench.sh — comparable LLM throughput benchmark for Apple Silicon
#
# Runs three probes against a local llama-server with prompt caching
# DISABLED (so each run is an honest, comparable measurement):
#
#   P1  short prompt + 600-token generation   → pure generation tok/s
#   P2  ~6K-token prompt + 60-token generation → pre-fill tok/s
#   P3  thinking-mode reasoning + 2K-token gen → realistic mixed workload
#
# Each probe runs N times (default 5). We report min/median/max so
# variance and thermal effects are visible.
#
# Output: a JSON file in ./runs/ and a markdown summary on stdout.
#
# Usage:
#   ./bench.sh --label "M2 Max 64GB"
#   ./bench.sh --runs 3 --label "M1 Max 64GB"
#   ./bench.sh --quick                # 1 run each (~30s smoke test)
#
# Env overrides:
#   HOST=http://localhost:8001  MODEL=qwen3.6-35b-a3b
#
# Requires: curl, jq, a running llama-server with --alias matching MODEL.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

HOST="${HOST:-http://localhost:8001}"
MODEL="${MODEL:-qwen3.6-35b-a3b}"
RUNS="${RUNS:-5}"
LABEL=""
OUT_DIR="${OUT_DIR:-$SCRIPT_DIR/runs}"

while [[ $# -gt 0 ]]; do
  case $1 in
    --label) LABEL="$2"; shift 2;;
    --runs)  RUNS="$2"; shift 2;;
    --host)  HOST="$2"; shift 2;;
    --model) MODEL="$2"; shift 2;;
    --quick) RUNS=1; shift;;
    -h|--help) sed -n '2,25p' "$0"; exit 0;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done

# --- preflight ---------------------------------------------------------------
command -v jq >/dev/null  || { echo "ERROR: install jq (brew install jq)"; exit 1; }
command -v curl >/dev/null || { echo "ERROR: curl not found"; exit 1; }

if ! curl -sf "$HOST/v1/models" >/dev/null; then
  echo "ERROR: llama-server not reachable at $HOST"
  echo "Start it with ./server.sh, then re-run."
  exit 1
fi

for f in p1.txt p2.txt p3.txt; do
  [[ -f "$SCRIPT_DIR/prompts/$f" ]] || { echo "ERROR: missing prompts/$f"; exit 1; }
done

# --- environment fingerprint -------------------------------------------------
CHIP=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || uname -m)
MEM_GB=$(awk -v b="$(sysctl -n hw.memsize 2>/dev/null || echo 0)" 'BEGIN{print int(b/1073741824)}')
CORES=$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 0)
OS=$(uname -srm)
LLAMA_VER=$(llama-server --version 2>&1 | grep -m1 -E "version|build" | tr -s ' ' || echo "unknown")
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
TIMESTAMP_FILE=$(date -u +"%Y%m%dT%H%M%SZ")    # filesystem-safe (no colons)
mkdir -p "$OUT_DIR"

# --- helpers ----------------------------------------------------------------
mkbody() {
  local prompt_file="$1" max="$2" temp="$3"
  jq -nc --rawfile p "$prompt_file" --arg m "$MODEL" --argjson n "$max" --argjson t "$temp" '{
    model:$m,
    messages:[{role:"user",content:$p}],
    max_tokens:$n,
    temperature:$t,
    top_p:0.95, top_k:20,
    seed:42,
    cache_prompt:false,
    stream:false
  }'
}

# Run one request → TSV: prompt_tps gen_tps ttft_ms prompt_n predicted_n
run_request() {
  local body="$1"
  curl -sS "$HOST/v1/chat/completions" \
    -H 'Content-Type: application/json' -d "$body" \
  | jq -r '
      if .timings then
        [(.timings.prompt_per_second // 0),
         (.timings.predicted_per_second // 0),
         (.timings.prompt_ms // 0),
         (.timings.prompt_n // 0),
         (.timings.predicted_n // 0)] | @tsv
      else
        [0,0,0,(.usage.prompt_tokens // 0),(.usage.completion_tokens // 0)] | @tsv
      end'
}

# --- run probes -------------------------------------------------------------
P1_BODY=$(mkbody "$SCRIPT_DIR/prompts/p1.txt" 600 0)
P2_BODY=$(mkbody "$SCRIPT_DIR/prompts/p2.txt" 60  0)
P3_BODY=$(mkbody "$SCRIPT_DIR/prompts/p3.txt" 2000 0.6)

echo "running benchmark — $RUNS runs per probe, prompt caching disabled"
echo "this will take a few minutes; do not use the laptop for other heavy work"
echo

RESULTS_TSV=""
run_probe() {
  local name="$1" body="$2"
  printf '  %s ' "$name"
  for i in $(seq 1 "$RUNS"); do
    line=$(run_request "$body")
    RESULTS_TSV+="${name}\t${i}\t${line}\n"
    printf '.'
  done
  printf ' done\n'
}

run_probe "P1_gen_speed" "$P1_BODY"
run_probe "P2_prefill"   "$P2_BODY"
run_probe "P3_thinking"  "$P3_BODY"

RESULTS_TSV=$(printf "$RESULTS_TSV")

# --- aggregate (min/median/max) using external sort -------------------------
# Returns "min<TAB>median<TAB>max" for the given column of $RESULTS_TSV
# rows whose first field matches $probe.
stats() {
  local probe="$1" col="$2"
  echo "$RESULTS_TSV" \
    | awk -F'\t' -v p="$probe" -v c="$col" '$1==p{print $c}' \
    | sort -n \
    | awk '
        { a[NR]=$1 }
        END {
          if (NR==0) { print "0\t0\t0"; exit }
          mid = int((NR+1)/2)
          printf "%.1f\t%.1f\t%.1f", a[1], a[mid], a[NR]
        }
      '
}

# For ttft we only need the median.
median() {
  local probe="$1" col="$2"
  echo "$RESULTS_TSV" \
    | awk -F'\t' -v p="$probe" -v c="$col" '$1==p{print $c}' \
    | sort -n \
    | awk '{a[NR]=$1} END { if (NR==0) {print 0; exit} ; print a[int((NR+1)/2)] }'
}

# Probe order, preserved.
PROBES="P1_gen_speed P2_prefill P3_thinking"
SUMMARY_TSV=""
for probe in $PROBES; do
  pf=$(stats "$probe" 3)        # prompt_tps min/med/max
  gn_=$(stats "$probe" 4)       # gen_tps   min/med/max
  ttft_med=$(median "$probe" 5)
  pn=$(echo "$RESULTS_TSV" | awk -F'\t' -v p="$probe" '$1==p{print $6; exit}')
  gn=$(echo "$RESULTS_TSV" | awk -F'\t' -v p="$probe" '$1==p{print $7; exit}')
  printf -v line '%s\t%s\t%s\t%.0f\t%d\t%d' "$probe" "$pf" "$gn_" "$ttft_med" "$pn" "$gn"
  SUMMARY_TSV+="${line}"$'\n'
done
SUMMARY_TSV=${SUMMARY_TSV%$'\n'}

# --- pretty-print summary ---------------------------------------------------
echo
echo "================ RESULTS ================"
[[ -n "$LABEL" ]] && echo "label : $LABEL"
echo "host  : $HOST"
echo "model : $MODEL"
echo "chip  : $CHIP ($CORES cores, ${MEM_GB} GB)"
echo "os    : $OS"
echo "build : $LLAMA_VER"
echo "runs  : $RUNS per probe"
echo
{
  printf 'probe\tprefill_min\tprefill_med\tprefill_max\tgen_min\tgen_med\tgen_max\tttft_med_ms\tprompt_n\tgen_n\n'
  echo "$SUMMARY_TSV"
} | column -t -s $'\t'
echo "========================================="

# --- write JSON --------------------------------------------------------------
SAFE_LABEL=$(echo "${LABEL:-run}" | tr ' /' '__' | tr -cd '[:alnum:]_-')
OUT_FILE="$OUT_DIR/${TIMESTAMP_FILE}_${SAFE_LABEL}.json"

jq -n \
  --arg ts "$TIMESTAMP" \
  --arg lbl "$LABEL" \
  --arg host "$HOST" \
  --arg model "$MODEL" \
  --arg chip "$CHIP" \
  --argjson mem "${MEM_GB:-0}" \
  --argjson cores "${CORES:-0}" \
  --arg os "$OS" \
  --arg build "$LLAMA_VER" \
  --argjson runs "$RUNS" \
  --arg raw "$RESULTS_TSV" \
  --arg sum "$SUMMARY_TSV" '
{
  timestamp: $ts,
  "label": $lbl,
  env: {
    host: $host, model: $model, chip: $chip,
    memory_gb: $mem, cores: $cores, os: $os, llama_build: $build,
    runs_per_probe: $runs
  },
  raw: ($raw | split("\n") | map(select(length>0)) | map(split("\t")) |
        map({probe:.[0], run:(.[1]|tonumber),
             prefill_tps:(.[2]|tonumber), gen_tps:(.[3]|tonumber),
             ttft_ms:(.[4]|tonumber), prompt_n:(.[5]|tonumber), gen_n:(.[6]|tonumber)})),
  summary: ($sum | split("\n") | map(select(length>0)) | map(split("\t")) |
            map({probe:.[0],
                 prefill_min:(.[1]|tonumber), prefill_med:(.[2]|tonumber), prefill_max:(.[3]|tonumber),
                 gen_min:(.[4]|tonumber),     gen_med:(.[5]|tonumber),     gen_max:(.[6]|tonumber),
                 ttft_med_ms:(.[7]|tonumber), prompt_n:(.[8]|tonumber),    gen_n:(.[9]|tonumber)}))
}' > "$OUT_FILE"

echo
echo "saved: $OUT_FILE"
echo "to share: commit this file in runs/ and open a PR"
