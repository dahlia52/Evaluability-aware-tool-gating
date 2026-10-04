#!/usr/bin/env bash
# Usage: scripts/serve.sh <model-short-name> <gpu-list> <port>
set -euo pipefail
SHORT=${1:?model short name required}
GPUS=${2:?CUDA device list required}
PORT=${3:?port required}
PARSER=qwen3_coder
EXTRA=()
case "$SHORT" in
  kanana-1.3b) MODEL=kakaocorp/kanana-2-1.3b-instruct; EXTRA+=(--trust-remote-code);;
  kanana-3b)   MODEL=kakaocorp/kanana-2-3b-instruct; EXTRA+=(--trust-remote-code);;
  kanana-30b)  MODEL=kakaocorp/kanana-2-30b-a3b-instruct; PARSER=hermes; EXTRA+=(--trust-remote-code);;
  qwen35-2b)   MODEL=Qwen/Qwen3.5-2B;;
  qwen35-4b)   MODEL=Qwen/Qwen3.5-4B;;
  qwen35-9b)   MODEL=Qwen/Qwen3.5-9B;;
  *) echo "unknown model short name: $SHORT" >&2; exit 2;;
esac
MODEL=${MODEL_ID:-$MODEL}
TP=$(echo "$GPUS" | tr ',' '\n' | wc -l)
echo "serving $SHORT ($MODEL) on GPU $GPUS port $PORT tp=$TP"
CUDA_VISIBLE_DEVICES=$GPUS exec vllm serve "$MODEL" \
  --served-model-name "$SHORT" --enable-auto-tool-choice --tool-call-parser "$PARSER" \
  --port "$PORT" --max-model-len 32768 --gpu-memory-utilization 0.90 \
  --tensor-parallel-size "$TP" "${EXTRA[@]}"
