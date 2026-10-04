<div align="center">

# Pre-Call Evaluability-Aware Tool Gating

**A deterministic safety layer for tool-using small language models**

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![Tests](https://img.shields.io/badge/tests-passing-2ea44f)
![Venue](https://img.shields.io/badge/HCLT-2026-7b2cbf)

</div>

Official implementation for **“Pre-Call Evaluability-Aware Tool Gating for
On-Device Small Language Models”** (HCLT 2026).

The paper shows that only 33 of 83 state-dependent preconditions in BFCL v4
multi-turn tools can be evaluated before call arguments are known. This
repository implements the paper's three-way tool exposure policy and its
deterministic gate immediately before execution:

- `AVAILABLE`: no violated state-only precondition;
- `CONDITIONAL`: a state-and-argument (`MIXED`) precondition must wait for the
  model to choose arguments;
- `BLOCKED`: a state-only (`STATE`) precondition is currently violated.

Once arguments are available, the gate evaluates both `STATE` and `MIXED`
preconditions. A rejected call is not executed, and an optional structured
repair message names tools that can modify the violated state.

## Key results

| Result | Value |
| --- | ---: |
| Extracted preconditions | 112 |
| State-dependent preconditions | 83 |
| Evaluable before arguments are known | 33 / 83 (39.8%) |
| Replayed ground-truth calls incorrectly blocked | 0 / 4,625 |
| Violating calls blocked in gate conditions | 16,986 |

The paper manuscript and submission sources are intentionally kept outside
version control.

## Repository contents

```text
artifacts/contracts/     frozen contracts used in the paper (112 preconditions)
configs/experiment.yaml paper models, conditions, and inference settings
ko_subset/               Korean translation of all 200 BFCL base instances
scripts/                 extraction, serving, evaluation, and analysis commands
src/precall_gating/      exposure control, gate, runner, data, and metrics
tests/                   dependency-free regression tests for the core policy
```

Raw model outputs and server logs are intentionally excluded because they are
generated artifacts (roughly 450 MiB in the original experiment workspace).

## Setup

Python 3.12, CUDA 12.8, and vLLM 0.19.0 were used for the paper. Create an
environment and install the Python dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Fetch the exact BFCL source revision used in the experiments:

```bash
scripts/setup_bfcl.sh
export BFCL_ROOT="$PWD/gorilla/berkeley-function-call-leaderboard"
```

The upstream source is cloned locally and is ignored by Git. The frozen commit
is `6ea57973c7a6097fd7c5915698c54c17c5b1b6c8`.

## Verify the implementation

The core policy tests do not require BFCL or a model server:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

Re-extract the contracts from BFCL and compare the reported inventory with the
paper (112 total: 33 `STATE`, 50 `MIXED`, 29 `ARG`):

```bash
FUNCTIONS="$BFCL_ROOT/bfcl_eval/eval_checker/multi_turn_eval/func_source_code"
python src/extract_preconditions.py "$FUNCTIONS" /tmp/preconditions.json
python src/extract_effects.py "$FUNCTIONS" /tmp/effects.json
```

The checked-in files under `artifacts/contracts/` are the exact inputs used by
the evaluation. The extractor intentionally matches the paper implementation:
it recognizes guards whose `if` body directly returns an error. Consequently,
it does not recover the 11 guards expressed through `else` branches, as noted
in the paper's limitations.

Before model experiments, replay BFCL ground-truth calls through the harness:

```bash
python scripts/validate_harness.py --split base --n 200 --condition M3
```

Repeat this for all four splits and gate conditions when validating a full
release. The paper's replay covered 4,625 unique ground-truth calls and blocked
none of them.

## Run an experiment

Start a model server. The helper selects the model ID and correct tool parser;
the server stays in the foreground so logs and lifecycle remain under the
caller's control.

```bash
scripts/serve.sh kanana-3b 0 8100
```

In another shell, wait for the server to become ready before starting the
experiment:

```bash
scripts/wait_ready.sh 8100
```

Run the 11 paper conditions on all four BFCL categories:

```bash
python scripts/run_experiment.py \
  --model kanana-3b \
  --base-url http://localhost:8100/v1 \
  --conditions B0 B1 B2 B3 B3r M1a M1b M2 M2b M3 M3b \
  --splits base miss_func miss_param long_context \
  --out out/runs \
  --workers 16
```

For Qwen3.5 models, disable thinking exactly as in the paper:

```bash
python scripts/run_experiment.py \
  --model qwen35-4b --base-url http://localhost:8102/v1 \
  --conditions B0 B1 B2 B3 B3r M1a M1b M2 M2b M3 M3b \
  --splits base miss_func miss_param long_context \
  --extra-body '{"chat_template_kwargs":{"enable_thinking":false}}' \
  --out out/runs
```

`kanana-2-30b-a3b-instruct` uses the `hermes` parser; the other five paper
models use `qwen3_coder`. `scripts/serve.sh` applies this distinction.
The published Kanana 1.3B configuration uses per-layer RoPE settings that
vLLM 0.19.0 does not understand directly; run `scripts/patch_vllm_qwen3.py`
once in the paper environment before serving that model. The patch is
version-specific, idempotent, and saves the original module as `qwen3.py.orig`.

## Korean evaluation

The paper evaluates all 200 `base` instances in Korean with nine conditions
(the standalone state-snapshot and precondition-document baselines are
excluded):

```bash
scripts/run_korean.sh kanana-3b 8100
```

The translated dataset was generated with
`gpt-4.1-mini-2025-04-14`, temperature 0, while keeping function names,
schemas, identifiers, and argument values in English. The released JSON files
allow evaluation without an OpenAI key. To regenerate the translation, set
`OPENAI_API_KEY`, run `scripts/ko_build_translation.py`, then apply the
generated text to the BFCL base data and validate it with
`scripts/ko_mech_check.py`. Translation regeneration may differ from the
released snapshot despite the dated model name.

## Aggregate and analyze

```bash
python scripts/result_tables.py --runs out/runs --tex out/tables/main.tex
python scripts/make_figures.py \
  --runs out/runs --out out/figures/remove_vs_tag_accuracy.pdf
```

`temperature=0` does not make vLLM continuous batching bitwise deterministic.
Small differences between repeated runs may therefore remain.

## Paper configuration

The complete model and condition matrix is recorded in
[`configs/experiment.yaml`](configs/experiment.yaml). The six evaluated models
are Kanana 2 1.3B, 3B, and 30B-A3B, plus Qwen3.5 2B, 4B, and 9B. Inference uses
a 32,768-token context, 1,024 generated tokens, and temperature 0.
