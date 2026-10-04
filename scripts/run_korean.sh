#!/usr/bin/env bash
# Compare the full 200-instance Korean base split with the English original.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
MODEL=${1:?served model name required}
PORT=${2:?server port required}
EXTRA_BODY=${3:-}
WORKERS=${WORKERS:-16}

scripts/wait_ready.sh "$PORT"

if [[ -z "$EXTRA_BODY" && "$MODEL" == qwen35-* ]]; then
  EXTRA_BODY='{"chat_template_kwargs":{"enable_thinking":false}}'
fi

ARGS=(--model "$MODEL" --base-url "http://localhost:$PORT/v1" --out out/runs_korean --tag korean --workers "$WORKERS"
   --conditions B0 B3 B3r M1a M1b M2 M2b M3 M3b --splits ko_full base
   --custom-split ko_full ko_subset/BFCL_v4_multi_turn_ko_full.json ko_subset/possible_answer_ko_full.json)
[[ -n "$EXTRA_BODY" ]] && ARGS+=(--extra-body "$EXTRA_BODY")
python scripts/run_experiment.py "${ARGS[@]}"
