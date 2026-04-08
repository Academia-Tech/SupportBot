#!/usr/bin/env python3
"""Compare GPT-5.4 vs Gemini synthesizer quality on real chat data.

Connects to the production DB (via SSH tunnel or direct), loads messages from
the specified group, runs the batch gate to extract questions, then synthesizes
answers using BOTH GPT-5.4 and Gemini (with Google Search grounding).

Usage:
    # On the OCI VM (inside the container or with env vars set):
    python scripts/eval_synthesizer_comparison.py

    # Or via SSH tunnel (forward MySQL 3306 and set MYSQL_HOST=127.0.0.1):
    ssh -L 3306:localhost:3306 supportbot
    MYSQL_HOST=127.0.0.1 python scripts/eval_synthesizer_comparison.py

Output: JSON file with side-by-side responses for manual review.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Add signal-bot to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "signal-bot"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("eval_synth")

# ── Configuration ──────────────────────────────────────────────────────────

GROUP_NAME_PATTERN = "Академія"  # substring match (fallback)
# Direct group_id override (set to skip name search)
GROUP_ID = os.environ.get("EVAL_GROUP_ID", "")
LAST_N = 40                      # messages to treat as "unprocessed"
CONTEXT_N = 40                   # prior context messages
OUTPUT_FILE = "results/synth_comparison.json"


@dataclass
class QuestionResult:
    question: str
    message_ids: list[str]
    reply_to_message_id: str
    gpt54_response: str = ""
    gemini_response: str = ""
    gpt54_time_s: float = 0.0
    gemini_time_s: float = 0.0
    gpt54_sub_agents: dict = field(default_factory=dict)
    gemini_sub_agents: dict = field(default_factory=dict)


@dataclass
class EvalResult:
    group_id: str
    group_name: str
    total_messages: int
    context_messages: int
    unprocessed_messages: int
    questions_extracted: int
    questions: list[QuestionResult] = field(default_factory=list)
    skipped: list[dict] = field(default_factory=list)
    batch_gate_raw: dict | None = None


def find_group(db, pattern: str) -> tuple[str, str]:
    """Find group_id by name pattern in chat_groups table."""
    with db.connection() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT group_id, group_name FROM chat_groups WHERE group_name LIKE %s LIMIT 5",
            (f"%{pattern}%",),
        )
        rows = cur.fetchall()
    if not rows:
        # Fallback: search in raw_messages for distinct group_ids
        log.warning("No match in chat_groups, searching raw_messages...")
        with db.connection() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT DISTINCT group_id FROM raw_messages LIMIT 20",
            )
            gids = [r[0] for r in cur.fetchall()]
        if gids:
            log.info("Available group_ids: %s", gids)
        raise ValueError(f"No group matching '{pattern}' found")

    if len(rows) > 1:
        log.info("Multiple groups found:")
        for gid, gname in rows:
            log.info("  %s: %s", gid[:20], gname)

    gid, gname = rows[0]
    log.info("Using group: %s (%s)", gname, gid[:20])
    return gid, gname


def run_eval():
    from app.config import load_settings
    from app.db import get_last_messages_text, get_raw_message
    from app.db.queries_mysql import get_last_messages_meta
    from app.llm.client import LLMClient
    from app.agent.ultimate_agent import UltimateAgent
    from app.ingestion import hash_sender
    from app.jobs.worker import _load_images, _is_image_path

    settings = load_settings()
    from app.db.mysql import create_mysql
    db = create_mysql(settings)

    ultimate_agent = UltimateAgent()  # self-initializes with settings, llm, rag
    llm = ultimate_agent.llm

    bot_sender_hash = hash_sender(settings.signal_bot_e164)

    # Find group
    if GROUP_ID:
        group_id = GROUP_ID
        group_name = GROUP_ID[:20] + "..."
        log.info("Using group_id from env: %s", group_id[:20])
    else:
        group_id, group_name = find_group(db, GROUP_NAME_PATTERN)

    # Load messages
    unprocessed_msgs = get_last_messages_meta(db, group_id, n=LAST_N, bot_sender_hash=bot_sender_hash)
    all_context = get_last_messages_text(db, group_id, n=CONTEXT_N + LAST_N, bot_sender_hash=bot_sender_hash)

    # Split context: everything before the unprocessed window
    first_text = ""
    for mm in unprocessed_msgs:
        if (mm.get("content_text") or "").strip():
            first_text = mm["content_text"]
            break

    pre_window: list[str] = []
    if first_text:
        for ci, cl in enumerate(all_context):
            if first_text[:50] in cl:
                pre_window = all_context[:ci]
                break
        else:
            pre_window = list(all_context)
    else:
        pre_window = list(all_context)

    context_text = "\n".join(pre_window)

    log.info("Loaded %d unprocessed messages, %d context lines", len(unprocessed_msgs), len(pre_window))

    # Format unprocessed for batch gate
    unprocessed_lines: list[str] = []
    msg_map: dict[str, dict] = {}
    for mm in unprocessed_msgs:
        mid = mm["message_id"]
        sender = mm["sender_hash"]
        text = mm.get("content_text") or ""
        is_bot = mm.get("is_bot", False)
        has_img = False

        msg_obj = get_raw_message(db, message_id=mid)
        if msg_obj and msg_obj.image_paths:
            img_paths = [p for p in msg_obj.image_paths if _is_image_path(p)]
            has_img = bool(img_paths)

        label = "[BOT]" if is_bot else f"User{(sender or 'unknown')[:6]}"
        img_marker = " [IMG]" if has_img else ""
        line = f"[msg_id={mid}] [{label}]: {text}{img_marker}"
        unprocessed_lines.append(line)
        msg_map[mid] = {**mm, "has_images": has_img, "raw_message": msg_obj}

    unprocessed_text = "\n".join(unprocessed_lines)

    # ── Batch gate ─────────────────────────────────────────────────────
    log.info("Running batch gate...")
    gate_result = llm.batch_gate(unprocessed=unprocessed_text, context=context_text)
    questions = gate_result.questions
    log.info("Gate extracted %d questions", len(questions))

    eval_result = EvalResult(
        group_id=group_id,
        group_name=group_name,
        total_messages=len(all_context),
        context_messages=len(pre_window),
        unprocessed_messages=len(unprocessed_msgs),
        questions_extracted=len(questions),
        batch_gate_raw={"questions": [q.model_dump() for q in questions]},
    )

    # ── Per-question: run BOTH synthesizers ────────────────────────────
    for qi, q in enumerate(questions):
        log.info("━" * 60)
        log.info("Question %d/%d: %s", qi + 1, len(questions), q.question[:100])

        # Build context for this question
        question_context_lines = list(pre_window)
        for mm in unprocessed_msgs:
            mid = mm["message_id"]
            is_bot = mm.get("is_bot", False)
            text = mm.get("content_text") or ""
            sender = mm["sender_hash"]
            label = "[BOT]" if is_bot else f"User{(sender or 'unknown')[:6]}"
            question_context_lines.append(f"[{label}]: {text}")
            if mid in q.message_ids and mid == q.message_ids[-1]:
                break

        question_context = "\n".join(question_context_lines)

        # Load images
        question_images: list[tuple[bytes, str]] | None = None
        if q.has_images:
            all_loaded: list[tuple[bytes, str]] = []
            for mid in q.message_ids:
                meta = msg_map.get(mid)
                if not meta or not meta.get("has_images"):
                    continue
                msg_obj = meta.get("raw_message")
                if msg_obj and msg_obj.image_paths:
                    loaded = _load_images(
                        settings=settings,
                        image_paths=[p for p in msg_obj.image_paths if _is_image_path(p)],
                        max_images=2,
                        total_budget_bytes=settings.max_total_image_bytes,
                    )
                    all_loaded.extend(loaded)
            if all_loaded:
                question_images = all_loaded[:4]

        qr = QuestionResult(
            question=q.question,
            message_ids=q.message_ids,
            reply_to_message_id=q.reply_to_message_id,
        )

        # ── GPT-5.4 synthesis ──────────────────────────────────────────
        log.info("  → Running GPT-5.4 synthesizer...")
        t0 = time.monotonic()
        try:
            gpt_answer = ultimate_agent.answer(
                q.question,
                group_id=group_id,
                db=db,
                lang="uk",
                context=question_context,
                images=question_images,
                gate_tag="batch_question",
            )
            qr.gpt54_response = gpt_answer.text
            qr.gpt54_sub_agents = gpt_answer.sub_agent_results or {}
        except Exception as exc:
            log.warning("  GPT-5.4 failed: %s", exc)
            qr.gpt54_response = f"ERROR: {exc}"
        qr.gpt54_time_s = round(time.monotonic() - t0, 2)
        log.info("  GPT-5.4 (%0.1fs): %s", qr.gpt54_time_s, qr.gpt54_response[:150])

        # ── Gemini synthesis (swap synthesizer call) ───────────────────
        log.info("  → Running Gemini synthesizer...")
        t0 = time.monotonic()
        try:
            # Temporarily swap the synthesizer to Gemini
            original_method = llm.chat_openai_grounded
            llm.chat_openai_grounded = lambda **kwargs: llm.chat_grounded(
                prompt=kwargs.get("prompt", ""),
                timeout=kwargs.get("timeout", 45.0),
                images=kwargs.get("images"),
            )

            gemini_answer = ultimate_agent.answer(
                q.question,
                group_id=group_id,
                db=db,
                lang="uk",
                context=question_context,
                images=question_images,
                gate_tag="batch_question",
            )
            qr.gemini_response = gemini_answer.text
            qr.gemini_sub_agents = gemini_answer.sub_agent_results or {}

            # Restore original
            llm.chat_openai_grounded = original_method
        except Exception as exc:
            log.warning("  Gemini failed: %s", exc)
            qr.gemini_response = f"ERROR: {exc}"
            llm.chat_openai_grounded = original_method
        qr.gemini_time_s = round(time.monotonic() - t0, 2)
        log.info("  Gemini (%0.1fs): %s", qr.gemini_time_s, qr.gemini_response[:150])

        eval_result.questions.append(qr)
        time.sleep(0.5)  # rate limit

    # ── Save results ───────────────────────────────────────────────────
    output_path = Path(OUTPUT_FILE)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "group_id": eval_result.group_id,
        "group_name": eval_result.group_name,
        "total_messages": eval_result.total_messages,
        "context_messages": eval_result.context_messages,
        "unprocessed_messages": eval_result.unprocessed_messages,
        "questions_extracted": eval_result.questions_extracted,
        "batch_gate": eval_result.batch_gate_raw,
        "comparisons": [],
    }

    for qr in eval_result.questions:
        output["comparisons"].append({
            "question": qr.question,
            "message_ids": qr.message_ids,
            "reply_to": qr.reply_to_message_id,
            "gpt54": {
                "response": qr.gpt54_response,
                "time_s": qr.gpt54_time_s,
                "sub_agents": qr.gpt54_sub_agents,
            },
            "gemini": {
                "response": qr.gemini_response,
                "time_s": qr.gemini_time_s,
                "sub_agents": qr.gemini_sub_agents,
            },
        })

    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2))
    log.info("Results saved to %s", output_path)

    # ── Print summary ──────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print(f"SYNTHESIZER COMPARISON: {group_name}")
    print(f"Messages: {eval_result.unprocessed_messages} unprocessed, "
          f"{eval_result.context_messages} context")
    print(f"Questions extracted: {eval_result.questions_extracted}")
    print("=" * 80)

    for i, qr in enumerate(eval_result.questions):
        print(f"\n{'─' * 80}")
        print(f"Q{i+1}: {qr.question}")
        print(f"{'─' * 80}")
        print(f"\n[GPT-5.4] ({qr.gpt54_time_s}s):")
        print(qr.gpt54_response)
        print(f"\n[GEMINI] ({qr.gemini_time_s}s):")
        print(qr.gemini_response)

    # Cost estimates
    total_gpt_time = sum(qr.gpt54_time_s for qr in eval_result.questions)
    total_gemini_time = sum(qr.gemini_time_s for qr in eval_result.questions)
    print(f"\n{'=' * 80}")
    print(f"Total GPT-5.4 time: {total_gpt_time:.1f}s")
    print(f"Total Gemini time:  {total_gemini_time:.1f}s")
    print(f"Output saved to: {OUTPUT_FILE}")
    print("=" * 80)


if __name__ == "__main__":
    run_eval()
