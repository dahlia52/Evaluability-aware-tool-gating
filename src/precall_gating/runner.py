"""Pre-call gating experiment loop built on BFCL primitives.

The runner adapts BFCL execution with exposure-control, gating, and repair
hooks, while scoring remains delegated to the official BFCL checker.

Measurement is separated from intervention: every condition evaluates
precondition violations before execution or gate intervention, but only
conditions with ``precall_gate=True`` block the call. This keeps violation
rates comparable to the no-intervention baseline.
"""
from __future__ import annotations

import json
import time
from typing import Any

from bfcl_eval.constants.default_prompts import (
    DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_FC,
    MAXIMUM_STEP_LIMIT,
)
from bfcl_eval.constants.executable_backend_config import (
    MULTI_TURN_FUNC_DOC_FILE_MAPPING,
)
from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import (
    execute_multi_turn_func_call,
)

from .conditions import ExposureController

SYSTEM_PROMPT = (
    "You are an expert in composing functions. You are given a question and a set of "
    "possible functions. Based on the question, you will need to make one or more "
    "function/tool calls to achieve the purpose.\n"
    "If none of the functions can be used, point it out. If the given question lacks "
    "the parameters required by the function, also point it out.\n\n"
    "At each turn, you should try your best to complete the tasks requested by the user "
    "within the current turn. Continue to output functions to call until you have "
    "fulfilled the user's request to the best of your ability. Once you have no more "
    "functions to call, the system will consider the current turn complete and proceed "
    "to the next turn or task."
)


def render_call(name: str, args: dict) -> str:
    """Convert an OpenAI tool call to the Python-call syntax BFCL evaluates."""
    return f"{name}({', '.join(f'{k}={v!r}' for k, v in args.items())})"


class ToolGatingRunner:
    def __init__(self, client, model: str, store, cond, run_tag: str,
                 temperature: float = 0.0, max_tokens: int = 1024,
                 max_steps: int = MAXIMUM_STEP_LIMIT,
                 extra_body: dict | None = None):
        self.client, self.model = client, model
        self.store, self.cond = store, cond
        self.ctl = ExposureController(store, cond)
        self.run_tag = run_tag
        self.temperature, self.max_tokens = temperature, max_tokens
        self.max_steps = max_steps
        self.extra_body = extra_body or {}

    @staticmethod
    def _owner_map(func_docs: list[dict]) -> dict[str, str]:
        return {d["name"]: d["_api_class"] for d in func_docs}

    def _instances(self, entry, long_context):
        _, inst = execute_multi_turn_func_call(
            [], entry["initial_config"], entry["involved_classes"],
            self.run_tag, entry["id"], long_context=long_context, is_evaL_run=False)
        return inst

    def _execute(self, calls, entry, long_context):
        return execute_multi_turn_func_call(
            calls, entry["initial_config"], entry["involved_classes"],
            self.run_tag, entry["id"], long_context=long_context, is_evaL_run=False)

    def run_entry(self, entry: dict, func_docs: list[dict]) -> dict:
        eid = entry["id"]
        category = eid.rsplit("_", 1)[0]
        long_context = "long_context" in category

        holdout: dict = entry.get("missed_function", {}) or {}
        active_docs = list(func_docs)
        owner = self._owner_map(func_docs)

        envs = self._instances(entry, long_context)
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        all_turn_results: list[list[list[str]]] = []
        events: list[dict] = []
        turn_stats: list[dict] = []
        force_quit = False

        for turn_idx, turn_msgs in enumerate(entry["question"]):
            if str(turn_idx) in holdout:
                extra = holdout[str(turn_idx)]
                for d in extra:
                    d = dict(d)
                    active_docs.append(d)
                    owner[d["name"]] = d.get("_api_class")
                turn_msgs = [{"role": "user",
                              "content": DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_FC}]

            for m in turn_msgs:
                messages.append({"role": m["role"], "content": m["content"]})

            turn_results: list[list[str]] = []
            step = 0
            t_in = t_out = 0
            lat: list[float] = []

            while True:
                tools, labels = self.ctl.build_tools(active_docs, envs)
                state_block = self.ctl.build_state_block(envs)
                sys_msg = SYSTEM_PROMPT + state_block
                messages[0] = {"role": "system", "content": sys_msg}

                t0 = time.time()
                try:
                    resp = self.client.chat.completions.create(
                        model=self.model, messages=messages, tools=tools,
                        tool_choice="auto", temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        extra_body=self.extra_body or None)
                except Exception as e:
                    events.append({"turn": turn_idx, "step": step, "type": "api_error",
                                   "error": str(e)[:300]})
                    break
                lat.append((time.time() - t0) * 1000)
                if resp.usage:
                    t_in += resp.usage.prompt_tokens or 0
                    t_out += resp.usage.completion_tokens or 0

                msg = resp.choices[0].message
                tcs = msg.tool_calls or []
                messages.append({
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [{"id": c.id, "type": "function",
                                    "function": {"name": c.function.name,
                                                 "arguments": c.function.arguments}}
                                   for c in tcs] or None,
                })
                if not tcs:
                    messages[-1].pop("tool_calls", None)
                    turn_results.append([])
                    break

                executed: list[str] = []
                for c in tcs:
                    name = c.function.name
                    try:
                        args = json.loads(c.function.arguments or "{}")
                        if not isinstance(args, dict):
                            args = {}
                        args_ok = True
                    except Exception:
                        args, args_ok = {}, False

                    api_class = owner.get(name)
                    env = envs.get(api_class)
                    call_str = render_call(name, args)

                    # Measure before either execution or gate intervention.
                    pre_v = (self.store.gate(api_class, name, env, args)
                             if (env is not None and args_ok) else None)
                    ev = {
                        "turn": turn_idx, "step": step, "function": name,
                        "api_class": api_class, "args_ok": args_ok,
                        "call": call_str,
                        "label": ("BLOCKED" if name in labels["BLOCKED"] else
                                  "CONDITIONAL" if name in labels["CONDITIONAL"] else
                                  "AVAILABLE"),
                        "known_function": api_class is not None,
                        "pre_violation": bool(pre_v),
                        "violated_state": pre_v.state_vars if pre_v else [],
                        "violation_kind": pre_v.kind if pre_v else None,
                        "enabling_tools": self.store.repair(pre_v) if pre_v else [],
                    }

                    # Intervene only when the current condition enables the gate.
                    if pre_v is not None and self.cond.precall_gate:
                        ev["outcome"] = "gate_blocked"
                        events.append(ev)
                        messages.append({"role": "tool", "tool_call_id": c.id,
                                         "content": self.ctl.gate_message(pre_v)})
                        continue

                    res, envs = self._execute([call_str], entry, long_context)
                    out = res[0] if res else ""
                    is_err = out.startswith("Error during execution:") or (
                        isinstance(out, str) and '"error"' in out[:60])
                    ev["outcome"] = "executed_error" if is_err else "executed_ok"
                    ev["result_head"] = out[:200]
                    events.append(ev)
                    executed.append(call_str)

                    content = out
                    if is_err:
                        content = self.ctl.format_error(api_class, name, out, env, args)
                    messages.append({"role": "tool", "tool_call_id": c.id,
                                     "content": content})

                turn_results.append(executed)
                step += 1
                if step > self.max_steps:
                    force_quit = True
                    break

            all_turn_results.append(turn_results)
            turn_stats.append({"turn": turn_idx, "steps": step,
                               "input_tokens": t_in, "output_tokens": t_out,
                               "latencies_ms": lat,
                               "exposed": {k: len(v) for k, v in labels.items()}})
            if force_quit:
                break

        # Pad unprocessed turns after force-quit to preserve checker alignment.
        while len(all_turn_results) < len(entry["question"]):
            all_turn_results.append([[]])

        return {"id": eid, "result": all_turn_results, "events": events,
                "turn_stats": turn_stats, "force_quit": force_quit}
