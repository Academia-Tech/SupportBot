#!/usr/bin/env python3
"""Evaluate bot response quality via /debug/answer endpoint.

Tests real questions from StabX groups and reports on:
- Case link relevance (are linked cases about the same topic?)
- Response accuracy (does the answer address the question?)
- Hallucination check (is information grounded in cases?)

Usage:
    python scripts/eval_debug_answer.py [--api-url URL] [--group-id GID]
"""
import argparse
import json
import re
import sys
import time
import urllib.request
import urllib.error

# StabX group IDs
STABX1_GID = "ftu7+VoMkMY4Krk8Jy5qWCT9QSfOUdc8Kjju/RpGkok="
STABX2_GID = "slBCQWykXU3IcfE9cd/nHAir1phGMDwminH2VGflpEE="

# Real questions from StabX chats (extracted from production DB)
TEST_CASES = [
    {
        "id": 1,
        "question": "бажаю здоровья.\nЧи коректно буде працювати система при такому встановленні та налаштуванні?",
        "group_id": STABX1_GID,
        "topic": "installation/mounting position",
        "expected_topics": ["mounting", "vibration", "camera FOV", "installation"],
    },
    {
        "id": 2,
        "question": "Бажаю здоров'я підкажіть чи є прошивка стабх на карму?",
        "group_id": STABX1_GID,
        "topic": "StabX firmware for GoPro Karma",
        "expected_topics": ["karma", "firmware", "ardupilot"],
    },
    {
        "id": 3,
        "question": "А як подружити стабх з р2д2 та працюючим gps. Є якась рекомендована параметрія?",
        "group_id": STABX1_GID,
        "topic": "StabX + R2D2 + GPS integration",
        "expected_topics": ["r2d2", "gps", "fuse gps", "parameters"],
    },
    {
        "id": 4,
        "question": "Підскажіть, будь-ласка, якщо направити якійсь потужний ліхтар в камери це дасть бій в оптичному утримуванні?",
        "group_id": STABX1_GID,
        "topic": "bright light affecting optical hold",
        "expected_topics": ["light", "camera", "optical", "position hold"],
    },
    {
        "id": 5,
        "question": "Вітаю. 10-12 хв польоту і видає помилку. І борт починає кружляти навколо своєї точки і не стабілізується. В чому може бути проблема?",
        "group_id": STABX1_GID,
        "topic": "loss of stability after 10-12 min flight",
        "expected_topics": ["overheating", "power", "raspberry pi", "stability"],
    },
    {
        "id": 6,
        "question": "Що може бути, все припаяно вірно, всі параметри залиті вірно, камера працює, немає фпс і мавлінк з вольтажем",
        "group_id": STABX1_GID,
        "topic": "MAVLINK down, no FPS/voltage",
        "expected_topics": ["mavlink", "uart", "connection", "tx/rx"],
    },
    {
        "id": 7,
        "question": "Підкажіть як зробити перемикання між камерами, на польотнику є лише один вихід на камеру поставив відеоселектор",
        "group_id": STABX1_GID,
        "topic": "camera switching via video selector",
        "expected_topics": ["camera", "selector", "rc passthrough", "servo"],
    },
    {
        "id": 8,
        "question": "Чи потрібно встановлювати 1 біт FuseAllVelocities у EK3_SRC_OPTIONS?",
        "group_id": STABX1_GID,
        "topic": "EK3_SRC_OPTIONS configuration",
        "expected_topics": ["ek3", "src_options", "velocity", "navigation"],
    },
    {
        "id": 9,
        "question": "Вітаю, підскажіть що робити якщо модуль підключається і майже одразу відєднюється від мережі?",
        "group_id": STABX1_GID,
        "topic": "module keeps disconnecting from WiFi",
        "expected_topics": ["wifi", "power", "disconnect", "reboot"],
    },
    {
        "id": 10,
        "question": "Скажіть будь ласка, при використанні r2d2 чи можна якось зробити так, щоби RTL працював тільки від r2d2? Тобто вимкнути його на Стаб-Х?",
        "group_id": STABX1_GID,
        "topic": "RTL with R2D2 vs StabX",
        "expected_topics": ["rtl", "r2d2", "failsafe", "stabx"],
    },
    {
        "id": 11,
        "question": "Підскажіть у кого є хороші піди фільтри на АрдуП для королеви шершнів?",
        "group_id": STABX1_GID,
        "topic": "PID/filter settings for specific drone",
        "expected_topics": ["pid", "filter", "autotune", "specific to frame"],
    },
    {
        "id": 12,
        "question": "Так як r2d2 і Stab-X використовують мавлінк, чи не могло це спричинити якийсь конфлікт в арду?",
        "group_id": STABX1_GID,
        "topic": "MAVLink conflict between R2D2 and StabX",
        "expected_topics": ["mavlink", "conflict", "r2d2", "stabx"],
    },
    {
        "id": 13,
        "question": "Вітаю! Скажіть будь ласка, як працює режим RTL? При втраті зв'язку дрон наче полетів на точку вильоту. Але після відновлення зв'язку, пілот так і не зміг взяти керування на себе.",
        "group_id": STABX1_GID,
        "topic": "RTL behavior and regaining control",
        "expected_topics": ["rtl", "failsafe", "mode switch", "control"],
    },
    {
        "id": 14,
        "question": "А на orange pi можна накатити?",
        "group_id": STABX1_GID,
        "topic": "StabX on Orange Pi compatibility",
        "expected_topics": ["orange pi", "raspberry pi", "compatibility"],
    },
    {
        "id": 15,
        "question": "Бажаю міцного - чи реально зробити дистанційне керування на волкснейл щоб наприклад станція стояла десь на висоті, а пілот за декілька км?",
        "group_id": STABX1_GID,
        "topic": "Walksnail relay/repeater setup",
        "expected_topics": ["relay", "repeater", "walksnail", "range"],
    },
    {
        "id": 16,
        "question": "Ще таке питання - дрон вирій 15 поставили денний стабх. Все працює. На дрон підключається дві батареї 6s, і на esc дає 50v. А на польотник 24v. Як виставити щоб на осд виводило коректний вольтаж по банках?",
        "group_id": STABX1_GID,
        "topic": "OSD voltage per cell with 12S (2x6S)",
        "expected_topics": ["osd", "voltage", "cell_count", "12s"],
    },
]


def call_debug_answer(api_url: str, group_id: str, question: str) -> dict:
    """Call the /debug/answer endpoint."""
    url = f"{api_url}/debug/answer"
    data = json.dumps({"group_id": group_id, "question": question, "lang": "uk"}).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}: {e.read().decode()[:200]}"}
    except Exception as e:
        return {"error": str(e)}


def extract_case_urls(text: str) -> list[str]:
    """Extract case URLs from response text."""
    return re.findall(r'supportbot\.info/case/([a-f0-9]+)', text or "")


def assess_result(test_case: dict, result: dict) -> dict:
    """Assess the quality of a single result."""
    response = result.get("response", "")
    case_ids = extract_case_urls(response)
    is_admin = result.get("is_admin_tag", False)
    has_cases = result.get("has_case_link", False)
    scrag_hits = result.get("ua_scrag_hits", 0)

    assessment = {
        "test_id": test_case["id"],
        "question_short": test_case["question"][:80],
        "topic": test_case["topic"],
        "scrag_hits": scrag_hits,
        "case_links": len(case_ids),
        "is_admin_tag": is_admin,
        "response_length": len(response),
        "response_preview": response[:200] if response else "",
    }

    # Check if response is SKIP or empty
    if response in ("SKIP", "[[TAG_ADMIN]]", ""):
        assessment["quality"] = "escalated" if is_admin else "skip"
    else:
        assessment["quality"] = "answered"

    return assessment


def main():
    parser = argparse.ArgumentParser(description="Eval bot responses via /debug/answer")
    parser.add_argument("--api-url", default="http://161.33.64.115:8000", help="Backend API URL")
    parser.add_argument("--ids", type=str, default="", help="Comma-separated test IDs to run (default: all)")
    parser.add_argument("--delay", type=float, default=2.0, help="Delay between requests (seconds)")
    args = parser.parse_args()

    test_ids = set()
    if args.ids:
        test_ids = {int(x.strip()) for x in args.ids.split(",")}

    cases_to_run = [tc for tc in TEST_CASES if not test_ids or tc["id"] in test_ids]
    print(f"Running {len(cases_to_run)} test cases against {args.api_url}")
    print("=" * 80)

    results = []
    for i, tc in enumerate(cases_to_run):
        print(f"\n[{i+1}/{len(cases_to_run)}] Test #{tc['id']}: {tc['topic']}")
        print(f"  Q: {tc['question'][:100]}...")

        t0 = time.time()
        raw = call_debug_answer(args.api_url, tc["group_id"], tc["question"])
        elapsed = time.time() - t0

        if "error" in raw:
            print(f"  ERROR: {raw['error']}")
            results.append({"test_id": tc["id"], "error": raw["error"]})
            continue

        assessment = assess_result(tc, raw)
        assessment["elapsed_s"] = round(elapsed, 1)
        results.append(assessment)

        # Print summary
        response = raw.get("response", "")
        case_ids = extract_case_urls(response)
        print(f"  SCRAG hits: {raw.get('ua_scrag_hits', 0)}")
        print(f"  Cases cited: {len(case_ids)}")
        print(f"  Admin tag: {raw.get('is_admin_tag', False)}")
        print(f"  Time: {elapsed:.1f}s")
        # Print response (first 300 chars)
        resp_preview = response[:300].replace("\n", "\n  | ")
        print(f"  Response:\n  | {resp_preview}")

        if i < len(cases_to_run) - 1:
            time.sleep(args.delay)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    answered = sum(1 for r in results if r.get("quality") == "answered")
    escalated = sum(1 for r in results if r.get("quality") == "escalated")
    skipped = sum(1 for r in results if r.get("quality") == "skip")
    errors = sum(1 for r in results if "error" in r)
    with_cases = sum(1 for r in results if r.get("case_links", 0) > 0)
    avg_cases = sum(r.get("case_links", 0) for r in results) / max(len(results), 1)

    print(f"Total: {len(results)}")
    print(f"Answered: {answered}")
    print(f"Escalated (TAG_ADMIN): {escalated}")
    print(f"Skipped: {skipped}")
    print(f"Errors: {errors}")
    print(f"With case links: {with_cases}")
    print(f"Avg case links: {avg_cases:.1f}")

    # Save detailed results
    out_path = "results/eval_debug_answer.json"
    with open(out_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nDetailed results saved to {out_path}")


if __name__ == "__main__":
    main()
