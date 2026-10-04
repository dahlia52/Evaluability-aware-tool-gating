#!/usr/bin/env python3
"""Check harness compatibility with the official BFCL scorer.

Ground-truth call paths are replayed as mock model responses. Under the default
B0 condition, any result below 100% indicates a harness/scorer mismatch that
should be investigated before running model experiments. Gated conditions may
intentionally reject a reference call. No model server is required.
"""
import argparse, json, re, sys, types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "gorilla" / "berkeley-function-call-leaderboard"))

from precall_gating.contracts import ContractStore
from precall_gating.conditions import CONDITIONS
from precall_gating.data import load_split
from precall_gating.runner import ToolGatingRunner
from precall_gating import score as S

CALL_RE = re.compile(r"^\s*([A-Za-z_]\w*)\s*\((.*)\)\s*$", re.S)


def parse_call(s: str):
    m = CALL_RE.match(s)
    if not m:
        return None
    name, argstr = m.group(1), m.group(2)
    try:
        node = __import__("ast").parse(f"f({argstr})", mode="eval").body
        args = {}
        for kw in node.keywords:
            args[kw.arg] = __import__("ast").literal_eval(kw.value)
        for i, pos in enumerate(node.args):
            args[f"__pos{i}"] = __import__("ast").literal_eval(pos)
        return name, args
    except Exception:
        return None


class MockClient:
    """Emit one reference path per turn, then stop producing calls.

    Reference paths may contain positional arguments such as ``sort('x')``.
    They are mapped to schema parameter names to match actual model output.
    """

    def __init__(self, gt, param_order=None):
        self.param_order = param_order or {}
        self.gt = gt
        self.turn = -1
        self.emitted = set()
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._create))
        self._nuser = 0

    def _create(self, **kw):
        msgs = kw["messages"]
        nuser = sum(1 for m in msgs if m["role"] == "user")
        turn = nuser - 1
        key = turn
        tcs = []
        if key not in self.emitted and 0 <= turn < len(self.gt):
            self.emitted.add(key)
            for i, c in enumerate(self.gt[turn]):
                p = parse_call(c)
                if not p:
                    continue
                name, args = p
                order = self.param_order.get(name, [])
                for k in [k for k in args if k.startswith("__pos")]:
                    idx = int(k[5:])
                    if idx < len(order):
                        args[order[idx]] = args.pop(k)
                tcs.append(types.SimpleNamespace(
                    id=f"c{turn}_{i}", type="function",
                    function=types.SimpleNamespace(
                        name=name, arguments=json.dumps(args, default=str))))
        msg = types.SimpleNamespace(content="", tool_calls=tcs or None)
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=msg)],
            usage=types.SimpleNamespace(prompt_tokens=0, completion_tokens=0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="base")
    ap.add_argument("--data", nargs=2, default=None, metavar=("DATA","ANS"))
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--condition", default="B0")
    ap.add_argument("--contracts", default=str(ROOT / "artifacts" / "contracts" / "preconditions.json"))
    ap.add_argument("--effects", default=str(ROOT / "artifacts" / "contracts" / "effects.json"))
    a = ap.parse_args()

    store = ContractStore.load(a.contracts, a.effects)
    cond = CONDITIONS[a.condition]
    if a.data:
        from precall_gating.data import register_custom, load_split_any
        register_custom("custom", a.data[0], a.data[1])
        entries, gts = load_split_any("custom")
    else:
        entries, gts = load_split(a.split)
    entries = entries[: a.n]

    ok = bad = 0
    fails = []
    for e in entries:
        gt = gts[e["id"]]
        order = {d["name"]: list((d.get("parameters") or {}).get("properties", {}).keys())
                 for d in e["function"]}
        for td in (e.get("missed_function") or {}).values():
            for d in td:
                order[d["name"]] = list((d.get("parameters") or {}).get("properties", {}).keys())
        runner = ToolGatingRunner(MockClient(gt, order), "mock", store, cond,
                            run_tag=f"val_{a.condition}_{e['id']}")
        r = runner.run_entry(e, e["function"])
        res = S.score_entry(e, gt, r["result"], f"val_{a.condition}_{e['id']}")
        if res.get("valid"):
            ok += 1
        else:
            bad += 1
            fails.append((e["id"], res.get("error_type"), str(res.get("error_message"))[:110]))

    print(f"\n정답 경로 재생 결과 [{a.split}, {a.condition}]: {ok}/{ok+bad} = {ok/(ok+bad)*100:.1f}%")
    if fails:
        print(f"\n실패 {len(fails)}건 (상위 12):")
        for i, (fid, et, em) in enumerate(fails[:12]):
            print(f"  {fid:<28} {et}\n      {em}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
