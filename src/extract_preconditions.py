"""Extract preconditions from BFCL backend source with the Python AST.

A branch is treated as a guard only when its own body returns an error.
Enclosing branch predicates are conjoined so nested guards retain both state
and argument dependencies. This intentionally matches the paper artifacts and
does not infer guards expressed only through an ``else`` branch.
"""
import ast, json, sys, pathlib, collections

SRC = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".")
OUT = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else "preconditions.json")

TARGETS = ["vehicle_control.py", "trading_bot.py", "travel_booking.py",
           "gorilla_file_system.py", "ticket_api.py", "message_api.py",
           "posting_api.py", "math_api.py"]

BLOCKS = (ast.If, ast.For, ast.While, ast.Try, ast.With, ast.AsyncFor, ast.AsyncWith)


def _is_error_return(node) -> bool:
    if not (isinstance(node, ast.Return) and isinstance(node.value, ast.Dict)):
        return False
    for k in node.value.keys:
        if isinstance(k, ast.Constant) and isinstance(k.value, str) \
           and "error" in k.value.lower():
            return True
    return False


def direct_error_return(body) -> ast.Return | None:
    """Find an error return outside the nested control-flow nodes in ``BLOCKS``."""
    for st in body:
        if _is_error_return(st):
            return st
        if isinstance(st, BLOCKS):
            continue  # Nested guards are handled by the recursive traversal.
        for sub in ast.walk(st):
            if _is_error_return(sub):
                return sub
    return None


def error_message(node) -> str:
    v = node.value
    for k, val in zip(v.keys, v.values):
        if isinstance(k, ast.Constant) and "error" in str(k.value).lower():
            if isinstance(val, ast.Constant):
                return str(val.value)
            try:
                return ast.unparse(val)[:240]
            except Exception:
                return "<dynamic>"
    return ""


def render(conds) -> str:
    parts = []
    for test, pos in conds:
        s = ast.unparse(test)
        parts.append(s if pos else f"not ({s})")
    return " and ".join(parts)


def classify(conds, params):
    state, args = set(), set()
    for test, _ in conds:
        for sub in ast.walk(test):
            if isinstance(sub, ast.Attribute) and isinstance(sub.value, ast.Name) \
               and sub.value.id == "self":
                state.add(sub.attr)
            elif isinstance(sub, ast.Name) and sub.id in params:
                args.add(sub.id)
    if state and args:
        kind = "MIXED"
    elif state:
        kind = "STATE"
    elif args:
        kind = "ARG"
    else:
        kind = "OTHER"
    return kind, sorted(state), sorted(args)


def visit(stmts, enclosing, params, ctx, out):
    for st in stmts:
        if isinstance(st, ast.If):
            pos = enclosing + [(st.test, True)]
            ret = direct_error_return(st.body)
            if ret is not None:
                kind, sv, av = classify(pos, params)
                if kind != "OTHER":
                    out.append({**ctx, "kind": kind, "state_vars": sv, "arg_vars": av,
                                "condition": render(pos)[:400],
                                "error_message": error_message(ret)[:240],
                                "depth": len(pos)})
            visit(st.body, pos, params, ctx, out)
            visit(st.orelse, enclosing + [(st.test, False)], params, ctx, out)
        elif isinstance(st, (ast.For, ast.While, ast.AsyncFor)):
            visit(st.body, enclosing, params, ctx, out)
            visit(st.orelse, enclosing, params, ctx, out)
        elif isinstance(st, ast.Try):
            visit(st.body, enclosing, params, ctx, out)
            for h in st.handlers:
                visit(h.body, enclosing, params, ctx, out)
            visit(st.orelse, enclosing, params, ctx, out)
            visit(st.finalbody, enclosing, params, ctx, out)
        elif isinstance(st, (ast.With, ast.AsyncWith)):
            visit(st.body, enclosing, params, ctx, out)


records = []
for fname in TARGETS:
    path = SRC / fname
    if not path.exists():
        continue
    tree = ast.parse(path.read_text())
    for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
        for fn in [n for n in cls.body if isinstance(n, ast.FunctionDef)]:
            if fn.name.startswith("_"):
                continue
            params = {a.arg for a in fn.args.args} - {"self"}
            ctx = {"file": fname, "api_class": cls.name, "function": fn.name}
            visit(fn.body, [], params, ctx, records)

OUT.write_text(json.dumps(records, ensure_ascii=False, indent=2))

by_class = collections.defaultdict(collections.Counter)
for r in records:
    by_class[r["api_class"]][r["kind"]] += 1
print(f"총 사전조건 {len(records)}개 추출 → {OUT}\n")
print(f"{'API 클래스':<26}{'STATE':>7}{'MIXED':>7}{'ARG':>7}{'합계':>7}")
print("-" * 54)
tot = collections.Counter()
for cls, c in sorted(by_class.items(), key=lambda x: -(x[1]['STATE'] + x[1]['MIXED'])):
    print(f"{cls:<26}{c['STATE']:>7}{c['MIXED']:>7}{c['ARG']:>7}{sum(c.values()):>7}")
    tot.update(c)
print("-" * 54)
print(f"{'전체':<26}{tot['STATE']:>7}{tot['MIXED']:>7}{tot['ARG']:>7}{sum(tot.values()):>7}")
sm = tot['STATE'] + tot['MIXED']
print(f"\n상태 의존 = {sm} ({sm/max(sum(tot.values()),1)*100:.1f}%)")
print(f"호출 전 판정 가능(STATE) = {tot['STATE']}/{sm} = {tot['STATE']/max(sm,1)*100:.1f}%")
