#!/usr/bin/env python3
"""Run deterministic checks over the Korean translation.

C1 detects unexpected scripts, C2 missing answer identifiers, C3 missing
numeric tokens except the ignored values 1, 2, and 3, C4 unusually short
translations, and C5 empty or untranslated text.
"""
import argparse, json, re, sys, unicodedata, collections
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "gorilla" / "berkeley-function-call-leaderboard"))
from precall_gating.data import load_split

OK_SCRIPTS = {"HANGUL", "LATIN", "CJK", "DIGIT", "FULLWIDTH"}
IDENT_ARGS = {"folder", "dir_name", "file_name", "file_name1", "file_name2", "source",
              "destination", "path", "name", "user", "username", "receiver_id",
              "sender_id", "symbol", "stock_name"}
ARG_RE = re.compile(r"(\w+)\s*=\s*(?:'([^']*)'|\"([^\"]*)\")")
NUM_RE = re.compile(r"\d+(?:\.\d+)?")  # Korean units may attach directly to numbers.


def alien_chars(s):
    bad = collections.Counter()
    for ch in s:
        if ch.isspace() or not ch.isalpha():
            continue
        try:
            name = unicodedata.name(ch).split()[0]
        except ValueError:
            name = "UNKNOWN"
        if name not in OK_SCRIPTS:
            bad[name] += 1
    return bad


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--korean-data",
        default=str(ROOT / "ko_subset" / "BFCL_v4_multi_turn_ko_full.json"),
    )
    parser.add_argument("--out", default=str(ROOT / "out" / "ko_mech_findings.json"))
    args = parser.parse_args()

    ent, gts = load_split("base")
    ko_entries = {row["id"]: row for row in (
        json.loads(line) for line in Path(args.korean_data).read_text().splitlines() if line.strip()
    )}
    tr = {}
    for entry in ent:
        translated = ko_entries[entry["id"]]
        for ti, turn in enumerate(translated["question"]):
            for mi, message in enumerate(turn):
                if message["role"] == "user":
                    tr[f"{entry['id']}|{ti}|{mi}"] = message["content"]

    findings = collections.defaultdict(list)
    ratios = []
    n = 0
    for e in ent:
        gt = gts[e["id"]]
        idents = []
        for turn in gt:
            for call in turn:
                for m in ARG_RE.finditer(call):
                    a = m.group(1); v = m.group(2) or m.group(3) or ""
                    if a in IDENT_ARGS and len(v) >= 2 and not v.isdigit():
                        idents.append(v)
        for ti, t in enumerate(e["question"]):
            for mi, m in enumerate(t):
                if m["role"] != "user":
                    continue
                k = f"{e['id']}|{ti}|{mi}"
                en, ko = m["content"], tr.get(k, "")
                n += 1

                if not ko.strip():
                    findings["C5_빈번역"].append((k, "")); continue
                if ko.strip() == en.strip():
                    findings["C5_미번역"].append((k, ko[:60]))

                bad = alien_chars(ko)
                if bad:
                    findings["C1_문자오염"].append((k, dict(bad)))

                lost = [v for v in idents if v in en and v not in ko]
                if lost:
                    findings["C2_식별자유실"].append((k, sorted(set(lost))))

                # Remove thousands separators before comparing numeric tokens.
                en_nums = set(NUM_RE.findall(en.replace(",", ""))) - {"1", "2", "3"}
                ko_nums = set(NUM_RE.findall(ko.replace(",", "")))
                miss = sorted(x for x in en_nums
                              if x not in ko_nums and x.lstrip("0") not in ko_nums)
                if miss:
                    findings["C3_수치유실"].append((k, miss))

                if len(en) >= 60:
                    ratios.append((k, len(ko) / len(en), len(ko), len(en)))

    # Use a distribution-based threshold because Korean is typically shorter
    # than English and a fixed threshold creates many false positives.
    if ratios:
        vals = sorted(r for _, r, _, _ in ratios)
        import statistics
        med = statistics.median(vals)
        cut = vals[max(0, int(0.03 * len(vals)))]
        for k, r, lk, le in ratios:
            if r <= cut:
                findings["C4_길이이상"].append((k, f"{lk}/{le}={r:.2f} (중앙값 {med:.2f}, 하위3% 컷 {cut:.2f})"))
    print(f"검사 발화: {n}개")
    if ratios:
        print(f"한국어/영어 길이비 중앙값 {med:.2f} — 하위 3%({cut:.2f} 이하)만 이상치로 본다\n")
    order = ["C1_문자오염", "C2_식별자유실", "C3_수치유실", "C4_길이이상", "C5_미번역", "C5_빈번역"]
    total = 0
    for key in order:
        rows = findings.get(key, [])
        total += len(rows)
        print(f"{key:<14} {len(rows):3d}건")
    print(f"{'합계':<14} {total:3d}건\n")

    for key in order:
        rows = findings.get(key, [])
        if not rows:
            continue
        print(f"=== {key} ({len(rows)}건) ===")
        for k, info in rows[:20]:
            print(f"  {k:<28} {info}")
        if len(rows) > 20:
            print(f"  ... 외 {len(rows)-20}건")
        print()

    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps({k: v for k, v in findings.items()}, ensure_ascii=False, indent=1))
    print(f"저장: {destination}")
    # C4 produces review candidates and therefore does not affect the exit code.
    hard_failures = sum(
        len(findings.get(key, []))
        for key in ("C1_문자오염", "C2_식별자유실", "C3_수치유실", "C5_미번역", "C5_빈번역")
    )
    return 0 if hard_failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
