"""Combine the official BFCL checker with tool-gating metrics.

The additional metrics cover call-level precondition violations, Repair@n,
false masking, execution errors, and runtime cost.
"""
from __future__ import annotations

import collections
from typing import Any

from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker import multi_turn_checker


def score_entry(entry: dict, gt: list[list[str]], result: list[list[list[str]]],
                run_tag: str) -> dict:
    try:
        return multi_turn_checker(result, gt, entry, entry["id"].rsplit("_", 1)[0], run_tag)
    except Exception as e:
        return {"valid": False, "error_type": "checker_exception",
                "error_message": str(e)[:300]}


def _call_events(records: list[dict]):
    for r in records:
        for e in r["events"]:
            if e.get("type") == "api_error":
                continue
            yield r, e


def gating_metrics(records: list[dict], gt_by_id: dict[str, list[list[str]]]) -> dict:
    n_calls = 0
    n_violation = 0  # Attempted violations measured before intervention.
    n_violation_executed = 0  # Violations that reached the environment.
    n_gate_blocked = 0
    n_exec_err = 0
    by_kind = collections.Counter()
    tax = collections.Counter()

    # A false mask is a BLOCKED tool that appears in the reference path.
    fm_events = fm_false = 0
    for r in records:
        gt = gt_by_id.get(r["id"], [])
        gt_funcs = {c.split("(")[0].strip() for t in gt for c in t}
        seen_blocked = set()
        for e in r["events"]:
            if e.get("label") == "BLOCKED" and e["function"] not in seen_blocked:
                seen_blocked.add(e["function"])
        # Exposure logs retain only class counts, so this conservatively
        # approximates false masks from labels on tools the model called.
        for f in seen_blocked:
            fm_events += 1
            if f in gt_funcs:
                fm_false += 1

    for r, e in _call_events(records):
        n_calls += 1
        if e.get("pre_violation"):
            n_violation += 1
            by_kind[e.get("violation_kind")] += 1
            if e.get("outcome") != "gate_blocked":
                n_violation_executed += 1
        if e.get("outcome") == "gate_blocked":
            n_gate_blocked += 1
        elif e.get("outcome") == "executed_error":
            n_exec_err += 1

    out = {
        "n_instances": len(records),
        "n_calls": n_calls,
        "n_executed_calls": n_calls - n_gate_blocked,
        "n_violation_events": n_violation,
        "n_violation_executed": n_violation_executed,
        # Attempted violations capture model intent before gate intervention.
        "attempted_violation_rate": round(n_violation / max(n_calls, 1), 4),
        # Executed violations capture unsafe calls that reached the environment.
        "executed_violation_rate": round(n_violation_executed / max(n_calls, 1), 4),
        "precondition_violation_rate": round(n_violation / max(n_calls, 1), 4),
        "violations_by_kind": dict(by_kind),
        "n_gate_blocked": n_gate_blocked,
        "n_execution_errors": n_exec_err,
        # Gate-rejected calls are excluded because they were never executed.
        "execution_error_rate": round(n_exec_err / max(n_calls - n_gate_blocked, 1), 4),
        "false_mask_events": fm_events,
        "false_mask_rate": round(fm_false / max(fm_events, 1), 4),
    }
    for n in (1, 2, 3):
        out[f"repair@{n}"] = repair_at_n(records, n)
    out.update(failure_taxonomy(records))
    out.update(cost_metrics(records))
    return out


def repair_at_n(records: list[dict], n: int = 1) -> float:
    """Measure recovery with at most n intervening calls before a successful retry."""
    tot = rep = 0
    for r in records:
        ev = [e for e in r["events"] if e.get("type") != "api_error"]
        for i, e in enumerate(ev):
            if not e.get("pre_violation"):
                continue
            tot += 1
            for x in ev[i + 1: i + 1 + n + 1]:
                if x["function"] == e["function"] and x.get("outcome") == "executed_ok":
                    rep += 1
                    break
    return round(rep / max(tot, 1), 4)


def failure_taxonomy(records: list[dict]) -> dict:
    """Classify the first post-violation action and state-awareness failures."""
    c = collections.Counter()
    for r in records:
        ev = [e for e in r["events"] if e.get("type") != "api_error"]
        for i, e in enumerate(ev):
            if not e.get("pre_violation"):
                continue
            # T1: the model called a tool classified as BLOCKED.
            if e.get("label") == "BLOCKED":
                c["T1_state_unaware"] += 1
            after = ev[i + 1:]
            if not after:
                c["T3_no_repair"] += 1
                continue
            nxt = after[0]
            if nxt["function"] == e["function"]:
                c["T3_no_repair"] += 1
            elif e.get("enabling_tools") and nxt["function"] in e["enabling_tools"]:
                c["OK_repair"] += 1
            elif e.get("enabling_tools"):
                c["T4_wrong_repair"] += 1
            else:
                c["T_unknown"] += 1
    tot = sum(v for k, v in c.items() if k != "T1_state_unaware")
    out = {f"tax_{k}": v for k, v in c.items()}
    out["tax_total"] = tot
    for k in ("T3_no_repair", "T4_wrong_repair", "OK_repair"):
        out[f"tax_{k}_ratio"] = round(c[k] / max(tot, 1), 4)
    return out


def cost_metrics(records: list[dict]) -> dict:
    lat, ptok, steps = [], [], []
    for r in records:
        for t in r["turn_stats"]:
            lat.extend(t["latencies_ms"])
            ptok.append(t["input_tokens"])
            steps.append(t["steps"])
    lat.sort()
    return {
        "avg_prompt_tokens_per_turn": round(sum(ptok) / max(len(ptok), 1), 1),
        "p95_latency_ms": round(lat[int(0.95 * (len(lat) - 1))], 1) if lat else 0.0,
        "avg_steps_per_turn": round(sum(steps) / max(len(steps), 1), 2),
        "force_quit_rate": round(sum(r["force_quit"] for r in records) / max(len(records), 1), 4),
    }


def mcnemar(a: list[bool], b: list[bool]) -> dict:
    """Run an exact paired McNemar test on instance-level outcomes."""
    from math import comb
    n01 = sum(1 for x, y in zip(a, b) if not x and y)
    n10 = sum(1 for x, y in zip(a, b) if x and not y)
    n = n01 + n10
    if n == 0:
        return {"n01": 0, "n10": 0, "p_value": 1.0}
    k = min(n01, n10)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / (2 ** n))
    return {"n01": n01, "n10": n10, "p_value": round(p, 5)}
