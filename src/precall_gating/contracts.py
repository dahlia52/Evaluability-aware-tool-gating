"""Runtime representation of extracted preconditions and effects.

Preconditions are classified as STATE, MIXED, or ARG. STATE conditions can be
evaluated before a call; MIXED conditions become evaluable after arguments are
bound. Effect records provide the reverse index used to suggest enabling tools.

Extracted predicates may reference module constants such as ``MAX_FUEL_LEVEL``,
so evaluation includes globals from the corresponding BFCL backend module.
"""
from __future__ import annotations

import importlib
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

BACKEND = "bfcl_eval.eval_checker.multi_turn_eval.func_source_code"

CLASS_MODULE = {
    "GorillaFileSystem": "gorilla_file_system",
    "MathAPI": "math_api",
    "MessageAPI": "message_api",
    "TwitterAPI": "posting_api",
    "TicketAPI": "ticket_api",
    "TradingBot": "trading_bot",
    "TravelAPI": "travel_booking",
    "VehicleControlAPI": "vehicle_control",
}

# Harness metadata is not a device precondition and must not affect gating.
HARNESS_VARS = {"long_context"}

_MODULE_GLOBALS: dict[str, dict] = {}


def module_globals(api_class: str) -> dict:
    if api_class not in _MODULE_GLOBALS:
        try:
            mod = importlib.import_module(f"{BACKEND}.{CLASS_MODULE[api_class]}")
            _MODULE_GLOBALS[api_class] = dict(vars(mod))
        except Exception:
            _MODULE_GLOBALS[api_class] = {}
    return _MODULE_GLOBALS[api_class]


@dataclass
class Precondition:
    api_class: str
    function: str
    kind: str  # STATE / MIXED / ARG
    state_vars: list[str]
    arg_vars: list[str]
    condition: str  # A true predicate indicates a violation.
    error_message: str
    curated_nl: str = ""  # Manually curated natural-language description.

    @property
    def is_state_dependent(self) -> bool:
        return self.kind in ("STATE", "MIXED")

    @property
    def is_harness_artifact(self) -> bool:
        return bool(self.state_vars) and set(self.state_vars) <= HARNESS_VARS

    @property
    def decidable_precall(self) -> bool:
        """Return whether the condition is decidable before arguments are known."""
        return self.kind == "STATE" and not self.is_harness_artifact

    def _eval(self, env, bound_args: dict[str, Any] | None) -> bool | None:
        """Evaluate the violation predicate, returning ``None`` on evaluation failure."""
        ns: dict[str, Any] = {"self": env}
        if bound_args:
            ns.update(bound_args)
        elif self.arg_vars:
            return None  # MIXED predicates require bound arguments.
        g = dict(module_globals(self.api_class))
        g["__builtins__"] = {
            "len": len, "str": str, "int": int, "float": float, "bool": bool,
            "abs": abs, "min": min, "max": max, "sum": sum, "any": any, "all": all,
            "isinstance": isinstance, "sorted": sorted, "round": round,
            "list": list, "dict": dict, "set": set, "tuple": tuple,
        }
        try:
            return bool(eval(self.condition, g, ns))  # noqa: S307
        except Exception:
            return None

    def violated_precall(self, env) -> bool:
        """Evaluate a STATE violation before call arguments are known."""
        if not self.decidable_precall:
            return False
        return self._eval(env, None) is True

    def violated_with_args(self, env, bound_args: dict[str, Any]) -> bool:
        """Evaluate STATE and MIXED violations after argument binding."""
        if not self.is_state_dependent or self.is_harness_artifact:
            return False
        return self._eval(env, bound_args) is True

    def to_nl(self) -> str:
        if self.curated_nl:
            return self.curated_nl.rstrip(".") + "."
        msg = re.sub(r"^f?['\"]|['\"]$", "", (self.error_message or "")).strip()
        msg = re.split(r"\+|Here (is|are)", msg)[0].strip()
        if msg:
            return f"Fails when: {msg.rstrip('.')}."
        return f"Depends on state: {', '.join(self.state_vars)}."


class EffectIndex:
    """Map state variables to functions that may update them."""

    def __init__(self, effects: dict[str, list[str]]):
        self.by_func = effects
        self.rev: dict[tuple[str, str], list[str]] = defaultdict(list)
        for key, vars_ in effects.items():
            cls, fn = key.split(".", 1)
            for v in vars_:
                self.rev[(cls, v)].append(fn)

    @classmethod
    def load(cls, path: str) -> "EffectIndex":
        return cls(json.load(open(path)))

    def enabling_tools(self, api_class: str, state_vars: list[str],
                       exclude: str | None = None) -> list[str]:
        out: list[str] = []
        for v in state_vars:
            for fn in self.rev.get((api_class, v), []):
                if fn != exclude and fn not in out:
                    out.append(fn)
        return out


class ContractStore:
    def __init__(self, records: list[dict], effects: dict[str, list[str]],
                 curated: dict[tuple[str, str, str], str] | None = None):
        curated = curated or {}
        self.by_func: dict[tuple[str, str], list[Precondition]] = defaultdict(list)
        self.all: list[Precondition] = []
        for r in records:
            key = (r["api_class"], r["function"], r["condition"])
            p = Precondition(
                api_class=r["api_class"], function=r["function"], kind=r["kind"],
                state_vars=r.get("state_vars", []), arg_vars=r.get("arg_vars", []),
                condition=r["condition"], error_message=r.get("error_message", ""),
                curated_nl=curated.get(key, ""),
            )
            self.by_func[(p.api_class, p.function)].append(p)
            self.all.append(p)
        self.effects = EffectIndex(effects)

    @classmethod
    def load(cls, pre_path: str, eff_path: str,
             curated_csv: str | None = None) -> "ContractStore":
        curated = {}
        if curated_csv:
            import csv, os
            if os.path.exists(curated_csv):
                with open(curated_csv) as f:
                    for row in csv.DictReader(f):
                        nl = (row.get("CURATED_NL") or "").strip()
                        if nl:
                            curated[(row["api_class"], row["function"],
                                     row["condition"])] = nl
        return cls(json.load(open(pre_path)), json.load(open(eff_path)), curated)

    def blocked_precall(self, api_class: str, function: str, env) -> Precondition | None:
        """Return a definitely violated STATE condition before the call."""
        for p in self.by_func.get((api_class, function), []):
            if p.violated_precall(env):
                return p
        return None

    def is_conditional(self, api_class: str, function: str) -> bool:
        """Return whether the function has a non-harness MIXED condition."""
        return any(p.kind == "MIXED" and not p.is_harness_artifact
                   for p in self.by_func.get((api_class, function), []))

    def conditional_vars(self, api_class: str, function: str) -> list[str]:
        out: list[str] = []
        for p in self.by_func.get((api_class, function), []):
            if p.kind == "MIXED" and not p.is_harness_artifact:
                for v in p.state_vars:
                    if v not in out:
                        out.append(v)
        return out

    def gate(self, api_class: str, function: str, env,
             bound_args: dict[str, Any]) -> Precondition | None:
        """Return the first violation found after argument binding."""
        for p in self.by_func.get((api_class, function), []):
            if p.violated_with_args(env, bound_args):
                return p
        return None

    def repair(self, p: Precondition) -> list[str]:
        return self.effects.enabling_tools(p.api_class, p.state_vars, exclude=p.function)

    def nl_for(self, api_class: str, function: str) -> list[str]:
        out, seen = [], set()
        for p in self.by_func.get((api_class, function), []):
            if not p.is_state_dependent or p.is_harness_artifact:
                continue
            s = p.to_nl()
            if s not in seen:
                seen.add(s)
                out.append(s)
        return out
