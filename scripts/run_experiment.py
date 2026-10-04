#!/usr/bin/env python3
"""Run the pre-call tool-gating experiment matrix.

  python scripts/run_experiment.py --model kanana-3b --base-url http://localhost:8100/v1 \
      --conditions B0 B1 B3 M1a M2 M3 --splits base miss_func miss_param long_context \
      --out out/runs

Instances run concurrently in threads. Each condition-split pair produces a
JSON file containing raw records and aggregate metrics.
"""
import argparse, json, os, sys, time, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "gorilla" / "berkeley-function-call-leaderboard"))

from precall_gating.contracts import ContractStore
from precall_gating.conditions import CONDITIONS
from precall_gating.data import load_split_any, blocked_gt_calls, register_custom
from precall_gating.runner import ToolGatingRunner
from precall_gating import score as S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="서빙 이름 (served-model-name)")
    ap.add_argument("--base-url", default="http://localhost:8100/v1")
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    ap.add_argument("--splits", nargs="+", default=["base", "miss_func", "miss_param", "long_context"])
    ap.add_argument("--contracts", default=str(ROOT / "artifacts" / "contracts" / "preconditions.json"))
    ap.add_argument("--effects", default=str(ROOT / "artifacts" / "contracts" / "effects.json"))
    ap.add_argument("--out", default=str(ROOT / "out" / "runs"))
    ap.add_argument("--limit", type=int, default=0, help="스플릿당 인스턴스 상한(디버그)")
    ap.add_argument("--relevant-only", action="store_true",
                    help="사전조건 관련 인스턴스만 실행")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--extra-body", default="",
                    help='요청에 실어 보낼 JSON. 예: \'{"chat_template_kwargs":{"enable_thinking":false}}\'')
    ap.add_argument("--custom-split", nargs=3, action="append", default=[],
                    metavar=("NAME", "DATA", "ANSWERS"),
                    help="사용자 정의 스플릿 등록 (한국어 서브셋 등)")
    a = ap.parse_args()

    for name, dp, apth in a.custom_split:
        register_custom(name, dp, apth)

    extra_body = json.loads(a.extra_body) if a.extra_body else {}

    from openai import OpenAI
    client = OpenAI(base_url=a.base_url, api_key=a.api_key, timeout=180, max_retries=3)

    store = ContractStore.load(a.contracts, a.effects)
    outdir = Path(a.out); outdir.mkdir(parents=True, exist_ok=True)

    data = {}
    for sp in a.splits:
        entries, gts = load_split_any(sp)
        if a.relevant_only:
            entries = [e for e in entries if blocked_gt_calls(e, gts[e["id"]], store)]
        if a.limit:
            entries = entries[: a.limit]
        data[sp] = (entries, gts)
        print(f"[data] {sp}: {len(entries)} instances")

    for cond_id in a.conditions:
        cond = CONDITIONS[cond_id]
        stamp = a.tag or time.strftime("%m%d%H%M")
        run_tag = f"{a.model}_{cond_id}_{stamp}".replace("/", "_").replace("-", "_").replace(".", "_")
        print(f"\n{'='*70}\n[run] model={a.model} condition={cond_id} tag={run_tag}\n{'='*70}")

        for sp in a.splits:
            entries, gts = data[sp]
            dest = outdir / f"{a.model}__{cond_id}__{sp}.json"
            if dest.exists():
                print(f"  [skip] {dest.name} 이미 존재")
                continue

            records, lock = [], threading.Lock()
            done = [0]

            def work(e):
                runner = ToolGatingRunner(client, a.model, store, cond,
                                    run_tag=f"{run_tag}_{sp}",
                                    temperature=a.temperature, max_tokens=a.max_tokens,
                                    extra_body=extra_body)
                try:
                    r = runner.run_entry(e, e["function"])
                except Exception as ex:
                    r = {"id": e["id"], "result": [[[]] for _ in e["question"]],
                         "events": [{"type": "api_error", "error": str(ex)[:300]}],
                         "turn_stats": [], "force_quit": True}
                r["valid"] = None
                return e, r

            t0 = time.time()
            with ThreadPoolExecutor(max_workers=a.workers) as ex:
                futs = [ex.submit(work, e) for e in entries]
                for f in as_completed(futs):
                    e, r = f.result()
                    with lock:
                        records.append((e, r))
                        done[0] += 1
                        if done[0] % 20 == 0:
                            print(f"    {done[0]}/{len(entries)}  ({time.time()-t0:.0f}s)")

            # Score sequentially because the BFCL checker uses global instances.
            recs = []
            for e, r in records:
                res = S.score_entry(e, gts[e["id"]], r["result"], f"{run_tag}_{sp}")
                r["valid"] = bool(res.get("valid"))
                r["error_type"] = res.get("error_type")
                recs.append(r)

            gt_by_id = {e["id"]: gts[e["id"]] for e, _ in records}
            m = S.gating_metrics(recs, gt_by_id)
            m["state_accuracy"] = round(sum(r["valid"] for r in recs) / max(len(recs), 1), 4)
            m.update({"model": a.model, "condition": cond_id, "split": sp,
                      "elapsed_s": round(time.time() - t0, 1)})

            dest.write_text(json.dumps({"metrics": m, "records": recs},
                                       ensure_ascii=False, default=str))
            print(f"  [{sp}] acc={m['state_accuracy']:.3f}  "
                  f"viol={m['precondition_violation_rate']:.3f} "
                  f"({m['n_violation_events']}/{m['n_calls']})  "
                  f"repair@1={m['repair@1']:.3f}  -> {dest.name}")


if __name__ == "__main__":
    main()
