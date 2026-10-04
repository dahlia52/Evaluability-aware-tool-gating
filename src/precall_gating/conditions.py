"""Experiment conditions and tool-exposure control.

B0 exposes the original tools. B1 adds state, and B2 adds natural-language
preconditions. B3/B3r use binary exposure control with annotation/removal;
M1a/M1b use three-way exposure control. M2/M2b add pre-call gating, while
M3/M3b also suggest candidate repair tools from the effect index. The
``r``/``b`` variants
remove BLOCKED tools instead of annotating them.

B3 and M1a differ only in whether CONDITIONAL tools are distinguished, which
isolates the contribution of the three-way classification.
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass

TYPE_MAP = {"dict": "object", "float": "number", "tuple": "array", "any": "string"}


def _norm_schema(s):
    if isinstance(s, dict):
        out = {}
        for k, v in s.items():
            if k == "type" and isinstance(v, str):
                out[k] = TYPE_MAP.get(v, v)
            else:
                out[k] = _norm_schema(v)
        if out.get("type") == "array" and "items" not in out:
            out["items"] = {"type": "string"}
        return out
    if isinstance(s, list):
        return [_norm_schema(v) for v in s]
    return s


@dataclass
class Condition:
    id: str
    inject_state: bool = False
    inject_precondition: bool = False
    exposure: str = "none"  # none | binary | tri
    expose_mode: str = "annotate"  # annotate | remove
    precall_gate: bool = False
    repair: str = "none"  # none | suggest
    structured_error: bool = False


CONDITIONS = {
    "B0":  Condition("B0"),
    "B1":  Condition("B1", inject_state=True),
    "B2":  Condition("B2", inject_state=True, inject_precondition=True),
    "B3":  Condition("B3", inject_state=True, inject_precondition=True,
                     exposure="binary", structured_error=True),
    # Exposure classes and BLOCKED-tool handling form a 2x2 design.
    "B3r": Condition("B3r", inject_state=True, inject_precondition=True,
                     exposure="binary", expose_mode="remove", structured_error=True),
    "M1a": Condition("M1a", inject_state=True, inject_precondition=True,
                     exposure="tri", structured_error=True),
    "M1b": Condition("M1b", inject_state=True, inject_precondition=True,
                     exposure="tri", expose_mode="remove", structured_error=True),
    "M2":  Condition("M2", inject_state=True, inject_precondition=True,
                     exposure="tri", precall_gate=True, structured_error=True),
    # Removal handles precall-decidable STATE cases; the gate handles MIXED
    # cases once arguments are bound.
    "M2b": Condition("M2b", inject_state=True, inject_precondition=True,
                     exposure="tri", expose_mode="remove", precall_gate=True,
                     structured_error=True),
    "M3b": Condition("M3b", inject_state=True, inject_precondition=True,
                     exposure="tri", expose_mode="remove", precall_gate=True,
                     repair="suggest", structured_error=True),
    "M3":  Condition("M3", inject_state=True, inject_precondition=True,
                     exposure="tri", precall_gate=True, repair="suggest",
                     structured_error=True),
}

PRIVATE_PREFIX = ("_",)
SKIP_STATE_KEYS = {"random_seed", "long_context"}
MAX_STATE_CHARS = 6000


def snapshot(env) -> dict:
    """Extract public environment state for prompt injection."""
    out = {}
    for k, v in vars(env).items():
        if k.startswith(PRIVATE_PREFIX) or k in SKIP_STATE_KEYS:
            continue
        if isinstance(v, (str, int, float, bool, type(None))):
            out[k] = v
        elif isinstance(v, (list, dict)):
            s = json.dumps(v, ensure_ascii=False, default=str)
            out[k] = v if len(s) <= 400 else f"<{type(v).__name__} with {len(v)} entries>"
    return out


class ExposureController:
    """Transform tool documents and track their exposure classifications."""

    def __init__(self, store, cond: Condition):
        self.store, self.cond = store, cond

    def build_tools(self, func_docs: list[dict], envs: dict) -> tuple[list[dict], dict]:
        """Return OpenAI tool schemas and names grouped by exposure class."""
        tools: list[dict] = []
        labels = {"BLOCKED": [], "CONDITIONAL": [], "AVAILABLE": []}

        for doc in func_docs:
            doc = copy.deepcopy(doc)
            api_class = doc.get("_api_class")
            env = envs.get(api_class)
            name = doc["name"]
            desc = doc.get("description", "").rstrip()

            if self.cond.inject_precondition and env is not None:
                pcs = self.store.nl_for(api_class, name)
                if pcs:
                    desc += "\n[Preconditions] " + " ".join(pcs)

            label = "AVAILABLE"
            if self.cond.exposure != "none" and env is not None:
                p = self.store.blocked_precall(api_class, name, env)
                if p is not None:
                    label = "BLOCKED"
                elif self.cond.exposure == "tri" and self.store.is_conditional(api_class, name):
                    label = "CONDITIONAL"

            if label == "BLOCKED":
                labels["BLOCKED"].append(name)
                if self.cond.expose_mode == "remove":
                    continue
                p = self.store.blocked_precall(api_class, name, env)
                desc += ("\n[UNAVAILABLE NOW] Blocked by current device state "
                         f"({', '.join(p.state_vars)}). Resolve that state first.")
            elif label == "CONDITIONAL":
                labels["CONDITIONAL"].append(name)
                cvars = self.store.conditional_vars(api_class, name)
                desc += ("\n[CONDITIONAL] Callable, but may fail depending on the "
                         f"arguments you choose, given current state ({', '.join(cvars)}).")
            else:
                labels["AVAILABLE"].append(name)

            tools.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": desc,
                    "parameters": _norm_schema(doc.get("parameters", {"type": "object",
                                                                      "properties": {}})),
                },
            })
        return tools, labels

    def build_state_block(self, envs: dict) -> str:
        if not self.cond.inject_state:
            return ""
        st = {c: snapshot(e) for c, e in envs.items()}
        body = json.dumps(st, ensure_ascii=False, indent=1, default=str)
        if len(body) > MAX_STATE_CHARS:
            body = body[:MAX_STATE_CHARS] + "\n... (truncated)"
        return ("\n\n# CURRENT DEVICE STATE\n" + body +
                "\nTools may fail if their state preconditions are unmet.\n")

    def gate(self, api_class: str, function: str, env, bound_args: dict):
        if not self.cond.precall_gate:
            return None
        return self.store.gate(api_class, function, env, bound_args)

    def gate_message(self, p) -> str:
        """Build feedback for a rejected call, including repair tools if enabled."""
        payload = {
            "status": "blocked_before_execution",
            "function": p.function,
            "violated_state": p.state_vars,
            "reason": p.to_nl(),
            "note": "The call was NOT executed, so device state is unchanged.",
        }
        if self.cond.repair == "suggest":
            tools = self.store.repair(p)
            if tools:
                payload["resolved_by"] = tools
                payload["instruction"] = (
                    "Call one of `resolved_by` first to fix the violated state, "
                    "then retry this call.")
            else:
                payload["instruction"] = "No enabling tool was found automatically."
        return json.dumps(payload, ensure_ascii=False)

    def format_error(self, api_class: str, function: str, raw: str, env,
                     bound_args: dict) -> str:
        """Format feedback for a call that failed during execution."""
        if not self.cond.structured_error:
            return raw
        p = self.store.gate(api_class, function, env, bound_args)
        payload = {"status": "error", "function": function, "raw": raw[:300]}
        if p is not None:
            payload["violated_state"] = p.state_vars
            if self.cond.repair == "suggest":
                t = self.store.repair(p)
                if t:
                    payload["resolved_by"] = t
        return json.dumps(payload, ensure_ascii=False)
