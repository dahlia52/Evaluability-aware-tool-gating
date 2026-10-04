"""Extract functions that write each state variable from BFCL source.

The resulting effect index maps violated state variables to candidate tools
that may enable a blocked call.
"""
import ast, json, sys, pathlib, collections

SRC = pathlib.Path(sys.argv[1]); OUT = pathlib.Path(sys.argv[2])
TARGETS = ["vehicle_control.py","trading_bot.py","travel_booking.py","gorilla_file_system.py",
           "ticket_api.py","message_api.py","posting_api.py","math_api.py"]

effects = collections.defaultdict(set)  # (class, function) -> written state variables
for fname in TARGETS:
    p = SRC/fname
    if not p.exists(): continue
    tree = ast.parse(p.read_text())
    for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
        for fn in [n for n in cls.body if isinstance(n, ast.FunctionDef)]:
            if fn.name.startswith("_"): continue
            for node in ast.walk(fn):
                tgts = []
                if isinstance(node, ast.Assign): tgts = node.targets
                elif isinstance(node, (ast.AugAssign, ast.AnnAssign)): tgts = [node.target]
                for t in tgts:
                    # Normalize both self.x and self.x[key] assignments to x.
                    base = t
                    while isinstance(base, ast.Subscript): base = base.value
                    if isinstance(base, ast.Attribute) and isinstance(base.value, ast.Name) \
                       and base.value.id == "self":
                        effects[(cls.name, fn.name)].add(base.attr)

out = {f"{c}.{f}": sorted(v) for (c,f),v in effects.items()}
OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2))
print(f"효과 추출: {len(out)}개 함수 → {OUT}")
