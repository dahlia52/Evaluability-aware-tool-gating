#!/usr/bin/env python3
"""Generate the paper's Section 6 table and console summary from run files.

Call-level metrics are recomputed from raw records. Pooling events instead of
averaging split-level rates preserves their correct weighting.
"""
import argparse, json, sys, collections
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src")); sys.path.insert(0, str(ROOT/"gorilla"/"berkeley-function-call-leaderboard"))
from precall_gating import score as S

ORDER = ["B0","B1","B2","B3","B3r","M1a","M1b","M2","M2b","M3","M3b"]
LABEL = {"B3r":"B3r 이진(제거)","M2b":"M2b 3분류(제거)+게이트","M3b":"M3b 3분류(제거)+게이트+복구","B0":"B0 개입 없음","B1":"B1 +상태","B2":"B2 +사전조건",
         "B3":"B3 +이진 노출제어","M1a":"M1a +3분류","M1b":"M1b 3분류(제거)",
         "M2":"M2 +실행직전 게이트","M3":"M3 +결정적 복구","M4":"M4 자동복구"}


def load(runs, only=None):
    """Load run files, optionally restricting them to a set of instance IDs."""
    g = collections.defaultdict(list)
    for f in sorted(Path(runs).glob("*.json")):
        try: d = json.loads(f.read_text())
        except Exception: continue
        m = dict(d["metrics"]); recs = d.get("records", [])
        if only is not None:
            recs = [r for r in recs if r["id"] in only]
            m["n_instances"] = len(recs)
            m["state_accuracy"] = (sum(bool(r.get("valid")) for r in recs) / len(recs)) if recs else 0.0
        g[(m["model"], m["condition"])].append((m, recs))
    return g


def agg(items):
    recs = [r for _, rs in items for r in rs]
    splits = sorted({m["split"] for m, _ in items})
    n_inst = sum(m["n_instances"] for m, _ in items)
    n_ok = sum(round(m["state_accuracy"]*m["n_instances"]) for m, _ in items)
    calls = viol = vexec = gate = eerr = 0
    for r in recs:
        for e in r["events"]:
            if e.get("type") == "api_error": continue
            calls += 1
            if e.get("pre_violation"):
                viol += 1
                if e.get("outcome") != "gate_blocked": vexec += 1
            if e.get("outcome") == "gate_blocked": gate += 1
            elif e.get("outcome") == "executed_error": eerr += 1
    out = {
        "n_inst": n_inst, "acc": n_ok/max(n_inst,1), "calls": calls,
        "viol": viol, "viol_rate": viol/max(calls,1),
        "vexec": vexec, "vexec_rate": vexec/max(calls,1),
        "gate": gate, "eerr_rate": eerr/max(calls-gate,1),
        "splits": splits, "n_splits": len(splits),
    }
    for n in (1,2,3): out[f"rep{n}"] = S.repair_at_n(recs, n)
    out.update(S.failure_taxonomy(recs))
    return out


def console(g, common_only=False):
    if common_only:
        # Keep splits shared by every condition to avoid partial-run bias.
        per_model = collections.defaultdict(list)
        for (mo, c), items in g.items():
            per_model[mo].append({m["split"] for m, _ in items})
        keep = {mo: set.intersection(*ss) if ss else set() for mo, ss in per_model.items()}
        g = {k: [(m, r) for m, r in v if m["split"] in keep[k[0]]] for k, v in g.items()}
        g = {k: v for k, v in g.items() if v}
        for mo, ks in keep.items():
            print(f"[{mo}] 공통 split만 사용: {sorted(ks)}")
    A = {k: agg(v) for k, v in g.items()}
    models = sorted({k[0] for k in A})
    print("="*118)
    print("Table 3  주 결과")
    print("="*118)
    print(f"{'model':<13}{'cond':<5}{'sp':>3}{'inst':>5}{'calls':>7}{'stateAcc':>10}"
          f"{'시도위반율':>11}{'실행위반율':>11}{'gate':>6}{'rep@1':>8}{'rep@3':>8}{'execErr':>9}")
    print("-"*118)
    for mo in models:
        for c in ORDER:
            a = A.get((mo,c))
            if not a: continue
            mark = "" if a["n_splits"] == 4 else "!"
            print(f"{mo:<13}{c+mark:<5}{a['n_splits']:>3}{a['n_inst']:>5}{a['calls']:>7}{a['acc']*100:>9.1f}%"
                  f"{a['viol_rate']*100:>10.2f}%{a['vexec_rate']*100:>10.2f}%{a['gate']:>6}"
                  f"{a['rep1']*100:>7.1f}%{a['rep3']*100:>7.1f}%{a['eerr_rate']*100:>8.1f}%")
        print()
    print("="*100)
    print("Table 1  사전조건 위반 후 행동 분포")
    print("="*100)
    print(f"{'model':<13}{'events':>8}{'재계획실패':>16}{'오복구':>13}{'복구성공':>12}{'미분류':>11}")
    print("-"*100)
    for mo in models:
        a = A.get((mo,"B0"))
        if not a: continue
        total = a.get("tax_total", 0)
        unknown = a.get("tax_T_unknown", 0)
        print(f"{mo:<13}{total:>8}"
              f"{a.get('tax_T3_no_repair',0):>9} ({a.get('tax_T3_no_repair_ratio',0)*100:>5.1f}%)"
              f"{a.get('tax_T4_wrong_repair',0):>6} ({a.get('tax_T4_wrong_repair_ratio',0)*100:>4.1f}%)"
              f"{a.get('tax_OK_repair',0):>5} ({a.get('tax_OK_repair_ratio',0)*100:>4.1f}%)"
              f"{unknown:>5} ({unknown/max(total,1)*100:>4.1f}%)")
    return A


def latex(A, dest):
    models = sorted({k[0] for k in A})
    L = ["\\begin{table*}[t]\\centering\\small",
         "\\caption{주 결과. 시도 위반율은 위반 술어가 참인 상태에서 모델이 시도한 호출의 비율이고, "
         "실행 위반율은 그 가운데 게이트에 반려되지 않고 환경에 실제로 실행된 호출의 비율이다. "
         "두 지표는 모두 전체 호출 수를 분모로 하므로 게이트가 없는 조건에서는 값이 일치한다. "
         "\\texttt{Repair@n}은 위반 후 최대 n개의 중간 호출을 거쳐 같은 함수를 "
         "성공적으로 재시도한 비율이다.}",
         "\\label{tab:main}",
         "\\begin{tabular}{llrrrrrr}", "\\toprule",
         "모델 & 조건 & State Acc. & 시도 위반율 & 실행 위반율 & \\texttt{Repair@1} & \\texttt{Repair@3} & 실행오류율 \\\\",
         "\\midrule"]
    for mo in models:
        first = True
        for c in ORDER:
            a = A.get((mo,c))
            if not a: continue
            name = mo if first else ""
            first = False
            L.append(f"{name} & {LABEL.get(c,c)} & {a['acc']*100:.1f}\\% & {a['viol_rate']*100:.2f}\\% & "
                     f"{a['vexec_rate']*100:.2f}\\% & {a['rep1']*100:.1f}\\% & {a['rep3']*100:.1f}\\% & "
                     f"{a['eerr_rate']*100:.1f}\\% \\\\")
        L.append("\\midrule")
    L[-1] = "\\bottomrule"
    L += ["\\end{tabular}", "\\end{table*}"]
    Path(dest).write_text("\n".join(L))
    print(f"\n저장: {dest}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(ROOT/"out"/"runs"))
    ap.add_argument("--tex", default=str(ROOT/"out"/"tables"/"main.tex"))
    ap.add_argument("--common-splits", action="store_true",
                    help="모든 조건에 공통인 split만 사용 (부분 완료 시 공정 비교)")
    ap.add_argument("--relevant-only", action="store_true",
                    help="사전조건 관련 인스턴스로 제한 (out/relevant_ids.json)")
    a = ap.parse_args()
    only = set(json.load(open(ROOT/"out"/"relevant_ids.json"))) if a.relevant_only else None
    g = load(a.runs, only)
    if not g: sys.exit("결과 없음: " + a.runs)
    A = console(g, common_only=a.common_splits)
    Path(a.tex).parent.mkdir(parents=True, exist_ok=True)
    latex(A, a.tex)
    aggregate = ROOT/"out"/"aggregate.json"
    aggregate.parent.mkdir(parents=True, exist_ok=True)
    aggregate.write_text(json.dumps({f"{k[0]}|{k[1]}": v for k, v in A.items()},
                                    ensure_ascii=False, indent=1, default=str))
