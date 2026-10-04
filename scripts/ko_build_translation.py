#!/usr/bin/env python3
"""Generate Korean utterance translations for a BFCL split through the OpenAI API."""
import argparse, json, os, re, sys, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "gorilla" / "berkeley-function-call-leaderboard"))

IDENT_ARGS = {"folder", "dir_name", "file_name", "file_name1", "file_name2", "source",
              "destination", "path", "name", "user", "username", "receiver_id",
              "sender_id", "symbol", "stock_name"}
ARG_RE = re.compile(r"(\w+)\s*=\s*(?:'([^']*)'|\"([^\"]*)\")")


def keep_literals(gt, en):
    """Return selected identifier arguments found verbatim in the source utterance."""
    out = []
    for turn in gt:
        for call in turn:
            for match in ARG_RE.finditer(call):
                argument = match.group(1)
                value = match.group(2) or match.group(3) or ""
                if (argument in IDENT_ARGS and len(value) >= 2 and not value.isdigit()
                        and value in en and value not in out):
                    out.append(value)
    return out


def batched(items, size):
    for start in range(0, len(items), size):
        yield items[start:start + size]

MODEL = "gpt-4.1-mini-2025-04-14"
TEMPERATURE = 0.0
BATCH = 15

SYS = """당신은 영어 사용자 발화를 한국어로 옮기는 번역가입니다.
이 발화들은 도구 호출 에이전트 벤치마크에서 사용자가 어시스턴트에게 내리는 요청입니다.

[문체]
- 반드시 해요체 존댓말로 통일합니다. 문장 끝은 "~해 주세요", "~할까요?", "~입니다" 형태입니다.
- 반말("~해줘", "~야")을 절대 쓰지 마세요. 한 데이터셋 안에서 문체가 섞이면 안 됩니다.

[충실성]
- 원문의 모든 요구사항·조건·수치·순서를 빠짐없이 옮깁니다.
- 요약하거나 생략하지 않습니다. 원문이 장황해도 내용을 줄이지 마세요.
- 원문에 없는 내용을 덧붙이지 않습니다.

[영문 유지]
- 각 항목의 keep_in_english 배열에 있는 문자열은 반드시 영문 그대로 남깁니다.
  이들은 정답 채점에 쓰이는 식별자(파일명·디렉터리명·사용자명 등)입니다.
  예: keep_in_english에 "document"가 있으면 "문서 디렉터리"가 아니라 "document 디렉터리".
- 중요: keep_in_english의 단어가 원문에서 형용사나 수식어처럼 보여도 식별자입니다.
  "my latest photography project" -> "최근 photography 프로젝트" (O) / "사진 프로젝트" (X)
  "our communal folder" -> "우리 communal 폴더" (O) / "공동 폴더" (X)
  "the existing shared directory" -> "기존 shared 디렉터리" (O) / "공유 디렉터리" (X)
- 토큰·카드ID·해시태그·따옴표로 감싼 리터럴 문자열도 영문을 유지합니다.
  예: 'final_report.pdf', USR001, #TravelGoals, securePass123
- 숫자·날짜·금액은 원문 표기를 보존합니다. $10,000은 "1만 달러"가 아니라 "10,000달러"입니다.
- 날짜가 계산식으로 서술된 경우 그 근거를 그대로 옮깁니다. 임의로 해석하거나 생략하지 마세요.
  "two days after close of the 14th day in the year 2026's eleventh month"
  -> "2026년 11월 14일이 끝나고 이틀 뒤" (O) / "2026년 11월" (X, 날짜 소실)

[한국어로 옮기는 것]
- 열거값에 해당하는 일반 명사는 자연스러운 한국어로 옮깁니다.
  예: business class -> 비즈니스 클래스, parking brake engaged -> 주차 브레이크를 걸다,
      Technology sector -> 기술 업종

입력의 각 항목에 대해 하나씩, 입력과 같은 개수의 결과를 돌려주세요."""

SCHEMA = {"type": "json_schema", "json_schema": {"name": "translations", "strict": True, "schema": {
    "type": "object",
    "properties": {"items": {"type": "array", "items": {
        "type": "object",
        "properties": {"id": {"type": "string"}, "ko": {"type": "string"}},
        "required": ["id", "ko"], "additionalProperties": False}}},
    "required": ["items"], "additionalProperties": False}}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--split", default="base")
    ap.add_argument("--out", default=str(ROOT / "out" / "ko_translations_api.json"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--n", type=int, default=0)
    a = ap.parse_args()

    from openai import OpenAI
    from precall_gating.data import load_split
    cli = OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=300)

    ent, gts = load_split(a.split)
    items = []
    for e in ent:
        for ti, t in enumerate(e["question"]):
            for mi, m in enumerate(t):
                if m["role"] == "user":
                    items.append({"id": f"{e['id']}|{ti}|{mi}", "en": m["content"],
                                  "keep": keep_literals(gts[e["id"]], m["content"])})
    if a.n:
        items = items[: a.n]
    print(f"발화 {len(items)}개 | 모델 {a.model} | temperature={TEMPERATURE} | 배치 {BATCH}",
          flush=True)

    out, tok = {}, [0, 0]

    def one(ch):
        for att in range(4):
            try:
                r = cli.chat.completions.create(
                    model=a.model, temperature=TEMPERATURE, max_tokens=12000,
                    messages=[{"role": "system", "content": SYS},
                              {"role": "user", "content": json.dumps(
                                  [{"id": x["id"], "en": x["en"],
                                    "keep_in_english": x["keep"]} for x in ch],
                                  ensure_ascii=False)}],
                    response_format=SCHEMA)
                d = json.loads(r.choices[0].message.content)["items"]
                if len(d) != len(ch):
                    raise ValueError(f"개수 불일치 {len(d)}!={len(ch)}")
                return d, (r.usage.prompt_tokens, r.usage.completion_tokens)
            except Exception as ex:
                if att == 3:
                    print(f"  실패: {str(ex)[:120]}", file=sys.stderr)
                    return [], (0, 0)
                time.sleep(2 * (att + 1))

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for d, u in ex.map(one, list(batched(items, BATCH))):
            tok[0] += u[0]; tok[1] += u[1]
            for r in d:
                out[r["id"]] = r["ko"]

    rows = [{"key": x["id"], "ko": out.get(x["id"], "")} for x in items]
    missing = sum(1 for r in rows if not r["ko"])
    Path(a.out).write_text(json.dumps(rows, ensure_ascii=False, indent=1))

    cost = tok[0] / 1e6 * 0.40 + tok[1] / 1e6 * 1.60
    print(f"\n완료 {len(rows)-missing}/{len(rows)}  (미생성 {missing})")
    print(f"토큰 in {tok[0]:,} / out {tok[1]:,} | 비용 ${cost:.3f} | {time.time()-t0:.0f}초")
    print(f"저장: {a.out}")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
