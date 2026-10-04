"""Load BFCL v4 multi-turn data and attach API ownership metadata."""
from __future__ import annotations

import json
import os
from pathlib import Path

from bfcl_eval.constants.executable_backend_config import (
    MULTI_TURN_FUNC_DOC_FILE_MAPPING,
)

SPLITS = ["base", "miss_func", "miss_param", "long_context"]


def bfcl_root() -> Path:
    p = os.environ.get("BFCL_ROOT")
    if p:
        return Path(p)
    return Path(__file__).resolve().parents[2] / "gorilla" / "berkeley-function-call-leaderboard"


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in open(path) if l.strip()]


_DOC_CACHE: dict[str, list[dict]] = {}


def class_docs(api_class: str) -> list[dict]:
    if api_class not in _DOC_CACHE:
        p = (bfcl_root() / "bfcl_eval" / "data" / "multi_turn_func_doc"
             / MULTI_TURN_FUNC_DOC_FILE_MAPPING[api_class])
        docs = _jsonl(p)
        for d in docs:
            d["_api_class"] = api_class
        _DOC_CACHE[api_class] = docs
    return [dict(d) for d in _DOC_CACHE[api_class]]


def load_split(split: str) -> tuple[list[dict], dict[str, list[list[str]]]]:
    root = bfcl_root() / "bfcl_eval" / "data"
    entries = _jsonl(root / f"BFCL_v4_multi_turn_{split}.json")
    gts = {g["id"]: g["ground_truth"]
           for g in _jsonl(root / "possible_answer" / f"BFCL_v4_multi_turn_{split}.json")}

    for e in entries:
        funcs: list[dict] = []
        for c in e["involved_classes"]:
            funcs.extend(class_docs(c))
        e["function"] = funcs
        # The miss_func split restores held-out functions at their target turn.
        if "missed_function" in e:
            for turn_index, names in list(e["missed_function"].items()):
                held = []
                for nm in names:
                    for i, fd in enumerate(e["function"]):
                        if fd["name"] == nm:
                            held.append(e["function"].pop(i))
                            break
                e["missed_function"][turn_index] = held
    return entries, gts


def precondition_relevant(entry: dict, gt: list[list[str]], store) -> bool:
    """Return whether the reference path includes an initially blocked tool."""
    return bool(blocked_gt_calls(entry, gt, store))


def blocked_gt_calls(entry: dict, gt: list[list[str]], store) -> list[str]:
    """Return reference calls whose STATE conditions fail initially."""
    from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import (
        execute_multi_turn_func_call,
    )
    tag = f"precheck_{entry['id']}"
    _, envs = execute_multi_turn_func_call(
        [], entry["initial_config"], entry["involved_classes"], tag, entry["id"],
        long_context="long_context" in entry["id"], is_evaL_run=False)
    owner = {d["name"]: d["_api_class"] for d in entry["function"]}
    for turn_docs in (entry.get("missed_function") or {}).values():
        for d in turn_docs:
            owner[d["name"]] = d["_api_class"]

    out = []
    for turn in gt:
        for call in turn:
            fn = call.split("(")[0].strip()
            cls = owner.get(fn)
            if cls and store.blocked_precall(cls, fn, envs.get(cls)) is not None:
                out.append(call)
    return out


CUSTOM_SPLITS: dict[str, tuple[str, str]] = {}


def register_custom(name: str, data_path: str, answer_path: str) -> None:
    CUSTOM_SPLITS[name] = (data_path, answer_path)


def load_split_any(split: str):
    """Load a registered custom split or fall back to the matching BFCL split."""
    if split not in CUSTOM_SPLITS:
        return load_split(split)
    dp, ap = CUSTOM_SPLITS[split]
    entries = _jsonl(Path(dp))
    gts = {g["id"]: g["ground_truth"] for g in _jsonl(Path(ap))}
    for e in entries:
        funcs = []
        for c in e["involved_classes"]:
            funcs.extend(class_docs(c))
        e["function"] = funcs
        if "missed_function" in e:
            for turn_index, names in list(e["missed_function"].items()):
                held = []
                for nm in names:
                    if isinstance(nm, dict):
                        held.append(nm); continue
                    for i, fd in enumerate(e["function"]):
                        if fd["name"] == nm:
                            held.append(e["function"].pop(i)); break
                e["missed_function"][turn_index] = held
    return entries, gts
