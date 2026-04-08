#!/usr/bin/env python3
"""SupportBench evaluation — architecture-agnostic support bot benchmark.

Metrics:
  Score     — quality × recall (0–10). Headline metric: penalizes both bad
              answers and missed questions. A system that answers 30/36 questions
              at quality 9.4 gets Score = 9.4 × 0.83 = 7.8, not 9.4.
  Quality   — mean per-response score (0–10, dimensions: correctness,
              faithfulness, helpfulness, conciseness)
  Precision — (responded - redundant) / responded
  Recall    — (responded - redundant) / (responded - redundant + missed)

Four systems:
  supportbot           — SupportBot (ingest → KB → gate → RAG → respond)
  baseline             — LLM-Aggressive  (stuff 40K context, attempt all questions)
  baseline-conservative — LLM-Conservative (stuff 40K context, skip when unsure)
  chunked-rag          — Chunked-RAG (chunk history, embed, retrieve top-k, respond)

Usage:
    GOOGLE_API_KEY=<key> python3 scripts/eval_supportbench.py \\
        --dataset ua_ardupilot --split 100 --system supportbot

    GOOGLE_API_KEY=<key> python3 scripts/eval_supportbench.py \\
        --dataset ua_ardupilot --split 100 --system chunked-rag

    GOOGLE_API_KEY=<key> python3 scripts/eval_supportbench.py \\
        --dataset ua_ardupilot --split 100 --system baseline-conservative
"""
from __future__ import annotations

import argparse
import hashlib
import html as _html_mod
import json
import logging
import os
import re
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "signal-bot"))

import chromadb

from app.config import Settings, _env
from app.llm.client import LLMClient
from app.llm.schemas import UnifiedBufferResult
from app.rag.chroma import ChromaRag, DualRag
from app.agent import case_search_agent as _csa_mod
from app.agent.case_search_agent import CaseSearchAgent
# Relax RAG distance threshold for eval: more candidates reach the reranker,
# which filters irrelevant ones.  Production uses 0.45; eval uses 0.60.
_csa_mod.SCRAG_DISTANCE_THRESHOLD = 0.45  # production threshold; 0.60 tested but hurt quality

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = REPO_ROOT / "datasets"
GITHUB_PAGES_URL = "https://pavelshpagin.github.io/SupportBot"


def _case_id_hash(title: str, summary: str) -> str:
    """Deterministic 12-char hex case ID from content."""
    h = hashlib.sha256(f"{title}|{summary}".encode()).hexdigest()[:12]
    return h


def _generate_case_html(case_id: str, case: dict, dataset: str) -> str:
    """Generate a static HTML page matching production Next.js case UI exactly."""
    title = _html_mod.escape(case.get("problem_title", "Case " + case_id))
    status_raw = case.get("status", "recommendation")
    if status_raw == "solved":
        status_label, status_cls = "Вирішено", "status-solved"
        sol_cls, sol_title = "solution-section", "Рішення"
        sol_icon = '<polyline points="20 6 9 17 4 12"/>'
    else:
        status_label, status_cls = "Рекомендація", "status-recommendation"
        sol_cls = "solution-section recommendation-solution"
        sol_title = "Рекомендація (не підтверджено)"
        sol_icon = '<circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>'
    problem = _html_mod.escape(case.get("problem_summary", ""))
    solution = _html_mod.escape(case.get("solution_summary", ""))
    tags = case.get("tags", [])
    tags_html = "".join(f'<span class="tag">#{_html_mod.escape(t)}</span>' for t in tags)
    return f"""<!DOCTYPE html>
<html><head>
<title>{title} | SupportBot</title>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<meta property="og:title" content="{title}">
<meta property="og:description" content="{_html_mod.escape(case.get('solution_summary', '')[:200])}">
<link rel="stylesheet" href="https://rsms.me/inter/inter.css">
<style>
:root {{
  --signal-blue: #2c6bed;
  --page-bg: #f6f7f9;
  --card-bg: #ffffff;
  --text: #0d0d0d;
  --text-sec: #5c5c5c;
  --border: #d8d8d8;
  --radius: 12px;
  --green: #16a34a;
  --yellow: #ca8a04;
}}
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
  font-family: "Inter", -apple-system, BlinkMacSystemFont, system-ui, sans-serif;
  background: var(--page-bg); color: var(--text);
  min-height: 100vh; padding: 48px 20px;
  -webkit-font-smoothing: antialiased;
}}
@media (max-width: 520px) {{ body {{ padding: 24px 12px; }} }}
.shell {{ max-width: 640px; margin: 0 auto; }}
.card {{
  background: var(--card-bg); border: 1px solid var(--border);
  border-radius: var(--radius); overflow: hidden; margin-bottom: 16px;
}}
header {{
  display: flex; align-items: center; justify-content: space-between;
  padding: 14px 20px; border-bottom: 1px solid var(--border);
}}
.header-left {{ display: flex; align-items: center; gap: 10px; text-decoration: none; color: inherit; }}
.brand {{ font-size: 15px; font-weight: 600; letter-spacing: -0.02em; }}
.status-badge {{
  padding: 4px 10px; font-size: 11px; font-weight: 600;
  text-transform: uppercase; letter-spacing: 0.04em; border-radius: 4px;
}}
.status-solved {{ background: #dcfce7; color: var(--green); }}
.status-recommendation {{ background: #fef3c7; color: #b45309; }}
main {{ padding: 24px 20px; }}
h1 {{ font-size: 20px; font-weight: 700; letter-spacing: -0.025em; margin-bottom: 12px; line-height: 1.3; }}
.meta {{ font-size: 12px; color: var(--text-sec); margin-bottom: 16px; }}
.tags {{ display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 20px; }}
.tag {{
  background: rgba(44, 107, 237, 0.08); color: var(--signal-blue);
  padding: 4px 10px; border-radius: 4px; font-size: 12px; font-weight: 500;
}}
.section-title {{
  font-size: 11px; font-weight: 600; text-transform: uppercase;
  letter-spacing: 0.06em; color: var(--text-sec); margin-bottom: 10px;
  display: flex; align-items: center; gap: 6px;
}}
.section-title svg {{ width: 14px; height: 14px; }}
.section-content {{ font-size: 15px; line-height: 1.6; color: var(--text); white-space: pre-wrap; }}
.problem-section {{ padding-bottom: 20px; border-bottom: 1px solid var(--border); margin-bottom: 20px; }}
.solution-section {{
  background: #f0fdf4; margin: -24px -20px -24px -20px; padding: 20px;
  border-top: 1px solid #bbf7d0;
}}
.solution-section .section-title {{ color: var(--green); }}
.recommendation-solution {{ background: #fffbeb; border-top-color: #fde68a; }}
.recommendation-solution .section-title {{ color: #b45309; }}
.empty-chat {{
  padding: 32px 20px; text-align: center; color: var(--text-sec); font-size: 14px;
}}
footer {{
  padding: 14px 20px; border-top: 1px solid var(--border);
  color: var(--text-sec); font-size: 12px; text-align: center;
}}
@media (max-width: 520px) {{
  main {{ padding: 20px 16px; }}
  h1 {{ font-size: 18px; }}
  .solution-section {{ margin: -20px -16px -20px -16px; padding: 16px; }}
}}
</style>
</head><body>
<div class="shell">
  <div class="card">
    <header>
      <a href="{GITHUB_PAGES_URL}" class="header-left">
        <span class="brand">SupportBot</span>
      </a>
      <span class="{status_cls} status-badge">{status_label}</span>
    </header>
    <main>
      <h1>{title}</h1>
      <p class="meta">Case {case_id} &middot; {_html_mod.escape(dataset)}</p>
      {"<div class='tags'>" + tags_html + "</div>" if tags else ""}
      <div class="problem-section">
        <h2 class="section-title">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/>
            <line x1="12" y1="16" x2="12.01" y2="16"/>
          </svg>
          Проблема
        </h2>
        <p class="section-content">{problem}</p>
      </div>
      <div class="{sol_cls}">
        <h2 class="section-title">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">{sol_icon}</svg>
          {sol_title}
        </h2>
        <p class="section-content">{solution}</p>
      </div>
    </main>
  </div>
  <div class="card">
    <div class="empty-chat">Історія переписки доступна у продакшн-системі на <a href="https://supportbot.info">supportbot.info</a></div>
    <footer>Academia Tech &copy; 2026</footer>
  </div>
</div>
</body></html>"""


def _write_case_pages(cases: list[dict], case_ids: list[str], dataset: str) -> Path:
    """Write static HTML case pages to docs/case/ and return the directory."""
    case_dir = REPO_ROOT / "docs" / "case"
    case_dir.mkdir(parents=True, exist_ok=True)
    for cid, case in zip(case_ids, cases):
        html = _generate_case_html(cid, case, dataset)
        (case_dir / f"{cid}.html").write_text(html, encoding="utf-8")
    # Write a simple index
    index_lines = [f'<li><a href="{cid}.html">{_html_mod.escape(c.get("problem_title","?"))}</a> [{c.get("status","?")}]</li>'
                   for cid, c in zip(case_ids, cases)]
    index_html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>Cases — {_html_mod.escape(dataset)}</title>
<style>body{{font-family:system-ui,sans-serif;max-width:720px;margin:40px auto;padding:0 20px}}</style></head>
<body><h1>Cases: {_html_mod.escape(dataset)}</h1><ul>{"".join(index_lines)}</ul></body></html>"""
    (case_dir / "index.html").write_text(index_html, encoding="utf-8")
    return case_dir
JUDGE_MODEL = "gemini-2.5-pro"
BASELINE_CONTEXT_CHARS = 40_000
MAX_WORKERS = 4  # parallel API calls (AI Studio has 1000+ RPM)

# Eval cascades — match production (gemini-2.5-pro primary, flash fallback)
EVAL_SYNTH_CASCADE = ["gemini-2.5-pro", "gemini-2.5-flash"]
EVAL_SUBAGENT_CASCADE = ["gemini-2.5-pro", "gemini-2.5-flash"]

# ── Cost tracking ─────────────────────────────────────────────────────────────
# USD per 1M tokens (input, output). Source: cloud.google.com/vertex-ai/pricing
MODEL_PRICING: dict[str, tuple[float, float]] = {
    "gemini-2.5-flash":         (0.15,  0.60),
    "gemini-2.5-pro":           (1.25, 10.00),
    "gemini-3-flash-preview":   (0.15,  0.60),
    "gemini-3.1-pro-preview":   (1.25, 10.00),
}


class CostTracker:
    """Accumulates token usage and computes estimated USD cost."""

    def __init__(self):
        self.input_tokens = 0
        self.output_tokens = 0
        self._by_model: dict[str, tuple[int, int]] = {}  # model → (in, out)

    def add(self, model: str, input_tok: int, output_tok: int):
        self.input_tokens += input_tok
        self.output_tokens += output_tok
        prev = self._by_model.get(model, (0, 0))
        self._by_model[model] = (prev[0] + input_tok, prev[1] + output_tok)

    def add_from_genai_response(self, model: str, response):
        """Extract tokens from google-genai response.usage_metadata."""
        um = getattr(response, "usage_metadata", None)
        if um:
            self.add(model,
                     getattr(um, "prompt_token_count", 0) or 0,
                     getattr(um, "candidates_token_count", 0) or 0)

    def add_from_openai_response(self, model: str, response):
        """Extract tokens from OpenAI-compatible response.usage."""
        u = getattr(response, "usage", None)
        if u:
            self.add(model,
                     getattr(u, "prompt_tokens", 0) or 0,
                     getattr(u, "completion_tokens", 0) or 0)

    @property
    def cost_usd(self) -> float:
        total = 0.0
        for model, (inp, out) in self._by_model.items():
            # Find closest pricing match
            price = MODEL_PRICING.get(model)
            if price is None:
                for k, v in MODEL_PRICING.items():
                    if k in model or model in k:
                        price = v
                        break
            if price is None:
                price = (0.15, 0.60)  # default: flash pricing
            total += inp / 1_000_000 * price[0] + out / 1_000_000 * price[1]
        return total

    def summary(self) -> dict:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.input_tokens + self.output_tokens,
            "cost_usd": round(self.cost_usd, 4),
            "by_model": {m: {"input": i, "output": o}
                         for m, (i, o) in self._by_model.items()},
        }


_IMG_MARKER_RE = re.compile(r"\[\[IMG:(\d+)\]\]")


def _eval_chat_grounded(llm: LLMClient, *, prompt: str, cascade: list[str],
                         timeout: float = 60.0, temperature: float = 0.0,
                         images: list[tuple[bytes, str]] | None = None,
                         cost: CostTracker | None = None) -> str:
    """Local grounded-chat wrapper for eval. Calls google-genai directly with custom temperature.
    Interleaves images at [[IMG:N]] marker positions (matches production chat_grounded)."""
    try:
        from google import genai as _genai
        from google.genai import types as _gt
    except ImportError:
        return llm.chat_grounded(prompt=prompt, cascade=cascade, timeout=timeout, images=images)

    client = llm._genai_client
    if client is None:
        return llm.chat(prompt=prompt, cascade=cascade, timeout=timeout, images=images)

    models = list(cascade)
    last_err = None
    t0 = time.time()
    for m in models:
        remaining = timeout - (time.time() - t0)
        if remaining < 5:
            break
        try:
            contents: list = []
            if images:
                # Interleave images at [[IMG:N]] positions (synced from prod client.py:340-356)
                segments = _IMG_MARKER_RE.split(prompt)
                referenced: set[int] = set()
                for i, seg in enumerate(segments):
                    if i % 2 == 0:
                        if seg:
                            contents.append(seg)
                    else:
                        idx = int(seg)
                        referenced.add(idx)
                        if idx < len(images):
                            img_bytes, img_mime = images[idx]
                            contents.append(_gt.Part.from_bytes(data=img_bytes, mime_type=img_mime))
                # Append unreferenced images at the end
                for idx, (img_bytes, img_mime) in enumerate(images):
                    if idx not in referenced:
                        contents.append(_gt.Part.from_bytes(data=img_bytes, mime_type=img_mime))
                if not contents:
                    contents = [prompt]
            else:
                contents = [prompt]

            response = client.models.generate_content(
                model=m,
                contents=contents,
                config=_gt.GenerateContentConfig(
                    tools=[_gt.Tool(google_search=_gt.GoogleSearch())],
                    system_instruction=(
                        "Use Google Search to verify facts and enrich your answer with up-to-date information. "
                        "Search for key technical terms, parameter names, and product specifics mentioned in the prompt. "
                        "Combine search results with the context already provided to give the best answer."
                    ),
                    temperature=temperature,
                    http_options=_gt.HttpOptions(timeout=int(remaining * 1000)),
                ),
            )
            if cost:
                cost.add_from_genai_response(m, response)
            text = response.text or ""
            if text.strip():
                return text
        except Exception as e:
            last_err = e
            log.warning("Eval grounded cascade: %s failed (%s), trying next", m, e)
            continue
    if last_err:
        raise last_err
    return ""


def _retry_on_429(fn, *args, max_retries=6, **kwargs):
    """Call fn(*args, **kwargs) with exponential backoff on 429 rate-limit errors."""
    for attempt in range(max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if "429" in str(e) and attempt < max_retries:
                wait = 2 ** attempt * 10  # 10, 20, 40, 80, 160, 320s
                log.info("Rate limited, retry %d/%d in %ds", attempt + 1, max_retries, wait)
                time.sleep(wait)
            else:
                raise


# ═══════════════════════════════════════════════════════════════════════════════
#  Buffer formatting (replicates worker.py:111-176)
# ═══════════════════════════════════════════════════════════════════════════════

_HEADER_RE = re.compile(
    r"^[^\n]*\sts=\d+(?:\s+msg_id=\S+)?(?:\s+reply_to=\S+)?(?:\s+reactions=\d+)?\n",
    re.MULTILINE,
)
_MSG_ID_RE = re.compile(r"msg_id=(\S+)")


def _format_buffer_line(msg: dict) -> str:
    reply = f" reply_to={msg['reply_to_id']}" if msg.get("reply_to_id") else ""
    reactions_raw = msg.get("reactions")
    pos = sum(reactions_raw.values()) if reactions_raw and isinstance(reactions_raw, dict) else 0
    reactions = f" reactions={pos}" if pos > 0 else ""
    body = msg.get("body") or ""
    return f"{msg['sender']} ts={msg['ts']} msg_id={msg['id']}{reply}{reactions}\n{body}\n\n"


def _format_numbered_buffer(messages: list[dict]) -> str:
    raw_text = "".join(_format_buffer_line(m) for m in messages)
    headers = list(_HEADER_RE.finditer(raw_text))
    blocks = []
    for i, m in enumerate(headers):
        start = m.start()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(raw_text)
        raw = raw_text[start:end]
        start_line = raw_text.count("\n", 0, start) + 1
        end_line = start_line + raw.count("\n")
        header_line = raw.split("\n")[0] if raw else ""
        mid = _MSG_ID_RE.search(header_line)
        blocks.append({
            "idx": i, "start_line": start_line, "end_line": end_line,
            "raw_text": raw, "message_id": mid.group(1) if mid else "",
        })
    out = []
    for b in blocks:
        out.append(f"### MSG idx={b['idx']} lines={b['start_line']}-{b['end_line']}")
        out.append(b["raw_text"].rstrip("\n"))
        out.append("### END")
        out.append("")
    return "\n".join(out).strip()


# ═══════════════════════════════════════════════════════════════════════════════
#  System 1: SupportBot (matches production: batch gate → CaseSearch → synthesizer)
# ═══════════════════════════════════════════════════════════════════════════════

# Patterns from ultimate_agent.py for post-processing synthesizer output
_ATTACH_PATTERN = re.compile(r"\[\[ATTACH:(.*?)\]\]")
_CITE_PATTERN = re.compile(r"\[cite:\s*([a-f0-9]{32})\]")
_CITE_BROAD_PATTERN = re.compile(r"\[cite:[^\]]*\]")
# Bracket blocks containing hex case IDs (synced from ultimate_agent.py)
_INLINE_CASE_BRACKET = re.compile(r"\[([a-f0-9]{8,32}(?:\s*,\s*[^\]]+)*)\]")
_HEX_CASE_ID = re.compile(r"[a-f0-9]{12,32}")
# Bare hex32 case IDs not already part of a /case/ URL
_BARE_HEX32 = re.compile(r"(?<!/case/)(?<![a-fA-F0-9])([a-fA-F0-9]{32})(?![a-fA-F0-9])")
_REPLY_TO_PATTERN = re.compile(r"\[\[REPLY_TO:(\d+)\]\]")


def _detect_lang(text: str) -> str:
    if re.search(r"[а-яіїєґА-ЯІЇЄҐ]", text):
        return "uk"
    if re.search(r"[áéíóúñüÁÉÍÓÚÑÜ¿¡]", text):
        return "es"
    return "en"


def _lang_instruction(lang: str) -> str:
    """Return LLM language instruction from ISO code. Auto-generates for any language."""
    _KNOWN = {"uk": "Ukrainian (українська)", "en": "English", "es": "Spanish (español)",
              "de": "German (Deutsch)", "fr": "French (français)", "pt": "Portuguese (português)",
              "it": "Italian (italiano)", "pl": "Polish (polski)", "nl": "Dutch (Nederlands)",
              "ja": "Japanese (日本語)", "zh": "Chinese (中文)", "ko": "Korean (한국어)"}
    return _KNOWN.get(lang, f"the same language as the user's message")


def _load_image(msg: dict, dataset_dir: Path) -> list[tuple[bytes, str]] | None:
    """Load image from message's media_path if it's a photo/image."""
    media_path = msg.get("media_path")
    media_type = msg.get("media_type", "")
    if not media_path or media_type not in ("photo", "image"):
        return None
    full_path = dataset_dir / media_path
    if not full_path.exists():
        return None
    try:
        data = full_path.read_bytes()
        mime = "image/jpeg" if full_path.suffix.lower() in (".jpg", ".jpeg") else "image/png"
        return [(data, mime)]
    except Exception:
        return None


# Eval gate prompt — unified English prompt for all languages.
# Based on production P_DECISION_SYSTEM but without the "active expert discussion"
# override that suppresses questions in human discussions. Eval measures capability
# (can the bot answer?), not deployment policy (should the bot interrupt?).
P_DECISION_EVAL = """Determine if a message in a technical support group chat warrants a bot response.
Return ONLY JSON with keys:
- consider: boolean
- tag: string (new_question | ongoing_discussion | noise | statement)

IMPORTANT: CONTEXT contains ONLY unresolved discussions (solved cases already removed).

Tags:
- new_question: New support question, unrelated to CONTEXT
- ongoing_discussion: Continuation of a discussion from CONTEXT
- statement: Summary, conclusion, statement of fact (NOT a question)
- noise: Greeting, "ok", thanks, emoji-only, off-topic

QUESTIONS (consider=true):
- Starts with question words (any language): "How?", "Why?", "What?", "Where?", "Is/Are?", "Як?", "Чому?", "¿Cómo?", "¿Por qué?"
- Contains a question mark "?" and is SELF-CONTAINED (understandable without CONTEXT)
- Asks for advice or a solution ("help me", "how to", "does anyone know")
- Describes a PROBLEM that needs solving: "doesn't work", "can't find", "error when..."
- Explicit help request: "need help", "can someone"
- Message with image + request ("look at this", "what's wrong")
- Even WITHOUT "?" — if the message describes a PROBLEM (something broken, can't do X) → consider=true

STATEMENTS (consider=false, tag=statement):
- "In summary...", "So...", "The conclusion is..."
- Statement of facts without asking for help
- "I did X, now Y works" (report of SUCCESSFUL result)
- Conclusions that do NOT ask for confirmation
- A SINGLE WORD or SHORT NAME (e.g. device model, brand name) — this is an answer to someone else's question
- OBSERVATION without a request: "this issue happens in both simulator and real tests"

consider=false if:
- Greetings, "ok", "thanks", emoji-only, off-topic
- Statements WITHOUT asking for help (tag=statement)
- ANSWER or ADVICE: "use X", "try Y", "check Z", "reboot it" — even if technical content. Imperative instructions directed at another user ("check your serial port", "specify drone or plane", "make a curve") = advice, NOT questions
- HELPER ASKING USER TO CLARIFY: "уточніть на чому", "тре уточнювати", "specify which one" — these are INSTRUCTIONS from a helper, not questions for the bot (tag=noise)
- Message reports problem is solved: "solved", "it worked", "thanks, that helped"
- Statement of fact or conclusion without a request
- A helper CONFIRMING/PARAPHRASING settings: "в портах мсп дісплейпорт, в осд тип hd" = helper providing answer (tag=statement)

CRITICAL — QUESTION MARK "?":
If the message contains "?" it MAY be a question, but you MUST distinguish between:
1. A user asking for HELP → consider=true
2. A helper asking the user a DIAGNOSTIC question → consider=false (tag=noise)
3. A helper speculating/suggesting → consider=false (tag=statement)

HELPER → USER DIAGNOSTIC QUESTIONS (consider=false, tag=noise):
These are questions from a HELPER directed at the ORIGINAL POSTER, asking about THEIR specific setup.
The bot should NOT answer these — they require the USER's answer, not the bot's.
Examples: "is X enabled?", "what does Y show?", "did you try Z?", "which firmware?",
"mission complete не каже в месседжах?", "а airspeed use стоїть?", "чи тільки enable?",
"скидує в нуль?", "Що hud показує?"
How to detect: CONTEXT shows the sender is NOT the person who originally asked for help —
they are a helper asking a diagnostic/clarifying question.

HELPER SPECULATION/SUGGESTION (consider=false, tag=statement):
A helper suggesting a possible cause or agreeing with another helper.
Examples: "Може TECS?", "I suspect EKF", "Try rebooting", "use version 3"

USER ASKING FOR HELP (consider=true):
The ORIGINAL POSTER (person who described the problem) asks a follow-up, provides new info,
or requests clarification. Also: any NEW support question from anyone.
IMPORTANT: "I tried X, here's what happened" / "Спробував X, ось що дало" from the original poster = IMPLICIT request for help interpreting results → consider=true, tag=ongoing_discussion.

FEEDBACK AFTER [BOT] RESPONSE (consider=true, tag=ongoing_discussion):
- If CONTEXT contains a [BOT] response, and the user writes "doesn't help", "doesn't work", "what else can I try?" → this is a REQUEST FOR MORE HELP
- Even a short message after a [BOT] response = follow-up to bot, consider=true

ACTIVE HUMAN DISCUSSION:
If CONTEXT shows users actively helping each other (→ replies, back-and-forth advice):
- Diagnostic questions from helpers to the poster → consider=false (tag=noise)
- Pure observations, speculation, suggestions → consider=false (tag=statement)
- BUT: any message from the ORIGINAL POSTER that DESCRIBES A PROBLEM, ASKS FOR HELP, or provides new diagnostic info → consider=true
- NEW questions on a DIFFERENT topic from anyone → consider=true

Tag logic:
- If CONTEXT is empty or different topic → new_question
- If CONTEXT contains a similar discussion → ongoing_discussion
- If statement/fact without a question → statement
- If not a question and not a discussion → noise

If there are images (screenshots, photos, diagrams), consider their content.
Messages like "look at this" or "what's wrong in this screenshot" with an image
often mean a request for help (new_question).
"""


EVAL_DOCS_SYSTEM_PROMPT = """You are a technical support automation system. Your goal is to strictly filter and answer questions based ONLY on the provided documentation.

INPUT CLASSIFICATION & BEHAVIOR:

1. ANALYZE: Is the user input a QUESTION or a REQUEST FOR HELP?
   - NO (Greetings, gratitude, statements, random phrases): Output exactly: "SKIP"
   - YES: Proceed to step 2.

2. EVALUATE: Do you have the information in the provided CONTEXT to answer it?
   - NO (The topic is not covered, or you cannot perform the requested action): Output exactly: "INSUFFICIENT_INFO"
   - YES: Provide a clear, technical answer. Cite the source URL. If the answer comes from a specific section, use: URL (Section: section name).

3. LANGUAGE: Respond in the same language as the question.

NEVER invent information not present in the provided documents."""


class EvalDocsAgent:
    """Eval-local DocsAgent: fetches public web pages, caches text, answers from docs.
    Mirrors production DocsAgent but works with HTTP URLs instead of Google Drive."""

    _cache: dict[str, tuple[str, float]] = {}  # url → (text, fetch_time)
    _CACHE_TTL = 3600  # 1 hour for eval (pages don't change during a run)
    _MAX_DOC_CHARS = 30_000  # max chars per doc to avoid prompt overflow

    def __init__(self, llm: LLMClient, docs_urls: list[str]):
        self.llm = llm
        self.docs_urls = docs_urls
        self._docs_text: str | None = None

    def _fetch_url(self, url: str) -> str:
        """Fetch a URL and extract text. Uses cache."""
        now = time.time()
        if url in self._cache:
            text, fetched_at = self._cache[url]
            if now - fetched_at < self._CACHE_TTL:
                return text

        try:
            import urllib.request
            req = urllib.request.Request(url, headers={"User-Agent": "SupportBench/1.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                html = resp.read().decode("utf-8", errors="replace")
        except Exception as e:
            log.warning("DocsAgent: failed to fetch %s: %s", url, e)
            return ""

        # Extract text from HTML
        try:
            from html.parser import HTMLParser
            class _TextExtractor(HTMLParser):
                def __init__(self):
                    super().__init__()
                    self.parts = []
                    self._skip = False
                def handle_starttag(self, tag, attrs):
                    if tag in ("script", "style", "nav", "footer", "header"):
                        self._skip = True
                def handle_endtag(self, tag):
                    if tag in ("script", "style", "nav", "footer", "header"):
                        self._skip = False
                    if tag in ("p", "div", "li", "h1", "h2", "h3", "h4", "tr", "br"):
                        self.parts.append("\n")
                def handle_data(self, data):
                    if not self._skip:
                        self.parts.append(data)

            ext = _TextExtractor()
            ext.feed(html)
            text = "".join(ext.parts)
        except Exception:
            # Fallback: strip tags with regex
            text = re.sub(r'<[^>]+>', ' ', html)

        # Clean up whitespace
        text = re.sub(r'\n{3,}', '\n\n', text).strip()
        text = text[:self._MAX_DOC_CHARS]

        self._cache[url] = (text, now)
        return text

    def _get_all_docs(self) -> str:
        """Fetch and concatenate all docs. Cached after first call."""
        if self._docs_text is not None:
            return self._docs_text

        parts = []
        total_chars = 0
        max_total = 120_000  # total budget across all docs

        for url in self.docs_urls:
            if total_chars >= max_total:
                break
            text = self._fetch_url(url)
            if text:
                remaining = max_total - total_chars
                chunk = text[:remaining]
                parts.append(f"--- Source: {url} ---\n{chunk}\n")
                total_chars += len(chunk)
                log.info("DocsAgent: fetched %s (%d chars)", url, len(chunk))

        self._docs_text = "\n".join(parts)
        print(f"    DocsAgent: fetched {len(parts)}/{len(self.docs_urls)} docs ({total_chars:,} chars)")
        return self._docs_text

    def answer(self, question: str, context: str = "",
               images: list[tuple[bytes, str]] | None = None) -> str:
        """Answer a question from docs. Returns answer, 'INSUFFICIENT_INFO', or 'SKIP'."""
        docs_text = self._get_all_docs()
        if not docs_text:
            return "NO_DOCS"

        ctx_block = f"\n\nCHAT CONTEXT:\n{context}" if context.strip() else ""
        prompt = (
            f"{EVAL_DOCS_SYSTEM_PROMPT}\n\n"
            f"DOCUMENTATION:\n{docs_text}\n"
            f"{ctx_block}\n\n"
            f"QUESTION: {question}"
        )

        try:
            return self.llm.chat(
                prompt=prompt,
                cascade=EVAL_SUBAGENT_CASCADE,
                timeout=30.0,
                images=images,
            )
        except Exception as e:
            log.warning("DocsAgent LLM failed: %s", e)
            return f"INSUFFICIENT_INFO"


class SupportBotSystem:
    """Replicates production pipeline: batch_gate → CaseSearchAgent → DocsAgent → synthesizer."""

    def __init__(self, settings: Settings, group_id: str, group_description: str = "",
                 dataset_dir: Path | None = None, lang: str = "en",
                 docs_urls: list[str] | None = None):
        self.settings = settings
        self.group_id = group_id
        self.group_description = group_description
        self.dataset_dir = dataset_dir  # for loading images
        self.docs_urls = docs_urls or []
        self.lang = lang  # dataset language (kept for potential future use)
        self.cost = CostTracker()
        self.llm = LLMClient(settings)
        self.docs_agent = EvalDocsAgent(self.llm, self.docs_urls) if self.docs_urls else None
        self.public_url = settings.public_url.rstrip("/")
        self.chroma_client = chromadb.EphemeralClient()
        self.rag = DualRag(
            scrag=ChromaRag(collection_name="eval_scrag", client=self.chroma_client),
            rcrag=ChromaRag(collection_name="eval_rcrag", client=self.chroma_client),
        )
        self.case_agent = CaseSearchAgent(
            rag=self.rag, llm=self.llm, public_url=settings.public_url,
        )
        self.num_cases = 0
        self._cases: list[dict] = []  # for keyword search
        self._history_msgs: list[dict] = []  # for keyword search

    def _enrich_messages_with_ocr(self, messages: list[dict]) -> list[dict]:
        """Pre-process: run OCR/description on images, append to message body.
        Matches production ingestion.py behavior."""
        enriched = 0
        for m in messages:
            if not self.dataset_dir:
                continue
            img = _load_image(m, self.dataset_dir)
            if not img:
                continue
            try:
                j = self.llm.image_to_text_json(
                    image_bytes=img[0][0],
                    context_text=(m.get("body") or "")[:200],
                )
                ocr_parts = []
                if j.extracted_text:
                    ocr_parts.append(f"Текст на зображенні: {j.extracted_text}")
                if j.observations:
                    ocr_parts.append(f"Елементи на зображенні: {', '.join(j.observations)}")
                if ocr_parts:
                    m["body"] = (m.get("body") or "") + "\n\n[Зображення: " + " | ".join(ocr_parts) + "]"
                    enriched += 1
            except Exception as e:
                log.debug("OCR failed for %s: %s", m.get("id"), e)
                m["body"] = (m.get("body") or "") + "\n\n[Зображення]"
        if enriched:
            print(f"    OCR enrichment: {enriched} messages with image descriptions")
        return messages

    def ingest(self, messages: list[dict], cases_cache: str | None = None) -> None:
        """Parallel ingestion: split into chunks, extract cases concurrently.

        If cases_cache is a path to an existing JSON file, load cases from it
        instead of calling the LLM.  If the file doesn't exist, extract and
        save there for future runs.
        """
        # Note: history messages are NOT OCR-enriched here because the extraction
        # LLM call already sees images multimodally via [[IMG:N]] markers.
        # OCR enrichment is applied to live messages in process_batch() instead,
        # matching production where ingestion.py enriches messages at arrival time.

        all_cases: list[dict] = []

        if cases_cache and Path(cases_cache).exists():
            with open(cases_cache) as f:
                all_cases = json.load(f)
            print(f"    Loaded {len(all_cases)} cached cases from {cases_cache} (skipping extraction)")

        # Generate deterministic case IDs
        self._case_ids = [
            _case_id_hash(c["problem_title"], c.get("problem_summary", ""))
            for c in all_cases
        ]

        if all_cases:
            docs = [
                f"Проблема: {c['problem_title']}\n{c['problem_summary']}\nРішення: {c['solution_summary']}"
                for c in all_cases
            ]
            print(f"    Embedding {len(docs)} cases...")
            try:
                embeddings = self.llm.embed_batch(texts=docs)
                for cid, case, doc, emb in zip(self._case_ids, all_cases, docs, embeddings):
                    self.rag.upsert_case(
                        case_id=cid, document=doc, embedding=emb,
                        metadata={"group_id": self.group_id, "status": case["status"],
                                  "problem_title": case["problem_title"]},
                        status=case["status"],
                    )
            except Exception as e:
                print(f"    Batch embed failed: {e}")

            # Write static case HTML pages for GitHub Pages
            case_dir = _write_case_pages(all_cases, self._case_ids, self.group_id)
            print(f"    Generated {len(all_cases)} case pages in {case_dir}")

        self.num_cases = len(all_cases)
        self._cases = all_cases
        print(f"    Ingested {len(all_cases)} cases")

    @staticmethod
    def _arrow_label(msg: dict, id_to_sender: dict[str, str]) -> str:
        """Format sender label with → arrow for replies, including msg_id for precise reply tracing."""
        sender = f"User{msg['sender'][:6]}"
        mid = msg.get("id", "")
        reply_id = msg.get("reply_to_id")
        if reply_id and reply_id in id_to_sender:
            target = f"User{id_to_sender[reply_id][:6]}"
            return f"[{sender} msg_id={mid} → {target} msg_id={reply_id}]"
        return f"[{sender} msg_id={mid}]"

    def process_batch(self, live_msgs: list[dict], history_msgs: list[dict]) -> list[dict]:
        """Per-message gate with override for explicit questions, then parallel synthesis.

        Uses production decide_consider() gate, but overrides noise/statement to
        consider=true when the message contains "?" — ensuring explicit questions
        always get a response attempt. Synthesizer context uses → arrows.
        """
        self._history_msgs = history_msgs  # for keyword search

        # Enrich live messages with OCR (matches production ingestion.py)
        live_msgs = self._enrich_messages_with_ocr(list(live_msgs))

        # Build msg_id → sender AND msg_id → reply_to_id lookups
        id_to_sender: dict[str, str] = {}
        id_to_reply: dict[str, str] = {}
        for m in history_msgs:
            id_to_sender[m["id"]] = m["sender"]
            if m.get("reply_to_id"):
                id_to_reply[m["id"]] = m["reply_to_id"]
        for m in live_msgs:
            id_to_sender[m["id"]] = m["sender"]
            if m.get("reply_to_id"):
                id_to_reply[m["id"]] = m["reply_to_id"]

        # Build context lines with → arrows (gate + synth both see reply chains)
        gate_base_lines = []
        synth_base_lines = []
        for m in history_msgs[-40:]:
            body = m.get("body") or ""
            label = self._arrow_label(m, id_to_sender)
            gate_base_lines.append(f"{label}: {body}")
            synth_base_lines.append(f"{label}: {body}")
        gate_live_lines = []
        synth_live_lines = []
        for m in live_msgs:
            body = m.get("body") or ""
            label = self._arrow_label(m, id_to_sender)
            gate_live_lines.append(f"{label}: {body}")
            synth_live_lines.append(f"{label}: {body}")

        actions = [{"action": "skip", "text": ""} for _ in live_msgs]

        # ── Phase A: Per-message gate with ? override ────────────────────
        print(f"    Gate: classifying {len(live_msgs)} messages...")
        gate_results: dict[int, tuple] = {}

        def _gate_one(i: int):
            msg = live_msgs[i]
            msg_text = msg.get("body") or ""
            if not msg_text.strip():
                return i, False, "empty", None
            ctx = "\n".join(gate_base_lines + gate_live_lines[:i])
            images = _load_image(msg, self.dataset_dir) if self.dataset_dir else None
            try:
                gate = _retry_on_429(self._gate, msg_text, ctx, images)
                consider = gate.consider
                tag = gate.tag or ""
                # No override needed — eval gate prompt is already permissive for "?"
                return i, consider, tag, images
            except Exception as e:
                log.warning("Gate failed for msg[%d]: %s", i, e)
                return i, True, "new_question", images

        t_gate = time.time()
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            for result in pool.map(_gate_one, range(len(live_msgs))):
                i, consider, tag, images = result
                gate_results[i] = (consider, tag, images)

        passed_indices = []
        for i in range(len(live_msgs)):
            consider, tag, images = gate_results.get(i, (False, "empty", None))
            msg_text = (live_msgs[i].get("body") or "")[:50]
            img_flag = " 📷" if images else ""
            if not consider:
                print(f"    → msg[{i}] . gate={tag}{img_flag} {msg_text}")
                continue
            passed_indices.append(i)
            print(f"    → msg[{i}] ? gate={tag}{img_flag} {msg_text}")

        gate_time = time.time() - t_gate
        print(f"    Gate done: {len(passed_indices)} passed / {len(live_msgs)} total ({gate_time:.1f}s)")

        # ── Phase B: Sequential search + synthesize (bot sees own prior responses) ─
        print(f"    Synthesizing {len(passed_indices)} responses sequentially (with context injection)...")

        # Bot response history — injected into context for subsequent messages
        # Maps live_msgs index → bot response text (like prod buffer)
        bot_responses: dict[int, str] = {}

        def _build_synth_context(i: int) -> str:
            """Build synthesis context with bot responses injected (matches prod)."""
            ctx_start = max(0, i - 20)
            lines = list(synth_base_lines)
            for j in range(ctx_start, i):
                lines.append(synth_live_lines[j])
                if j in bot_responses:
                    lines.append(f"[BOT]: {bot_responses[j]}")
            return "\n".join(lines)

        t_synth = time.time()
        for i in passed_indices:
            msg = live_msgs[i]
            msg_text = msg.get("body") or ""
            _, tag, images = gate_results[i]
            ctx = _build_synth_context(i)

            # Parallel sub-agent search (case + keyword + docs simultaneously)
            # Production pipeline: answer_raw → reranker → format_cases
            case_ans = "No relevant cases found."
            keyword_ans = "No keyword matches."
            docs_ans = "NO_DOCS"
            case_raw: list[dict] = []
            keyword_raw: dict = {"cases": [], "negative_notes": []}
            with ThreadPoolExecutor(max_workers=3) as search_pool:
                case_f = search_pool.submit(
                    _retry_on_429, lambda: self.case_agent.answer_raw(msg_text, group_id=self.group_id, db=None))
                kw_f = search_pool.submit(
                    _retry_on_429, lambda q=msg_text, c=ctx, im=images: self._keyword_search_raw(q, c, im))
                docs_f = search_pool.submit(
                    _retry_on_429, lambda q=msg_text, c=ctx, im=images: self.docs_agent.answer(q, context=c, images=im)
                ) if self.docs_agent else None
                try:
                    case_raw = case_f.result(timeout=120)
                except Exception as e:
                    log.warning("CaseSearch failed for msg[%d]: %s", i, e)
                try:
                    keyword_raw = kw_f.result(timeout=120)
                except Exception as e:
                    log.warning("KeywordSearch failed for msg[%d]: %s", i, e)
                if docs_f:
                    try:
                        docs_ans = docs_f.result(timeout=120)
                    except Exception as e:
                        log.warning("DocsAgent failed for msg[%d]: %s", i, e)

            # Merge & deduplicate raw cases (matches prod ultimate_agent.py:145-157)
            all_candidates: list[dict] = []
            seen_ids: set = set()
            for c in case_raw:
                cid = c.get("case_id", "")
                if cid not in seen_ids:
                    seen_ids.add(cid)
                    all_candidates.append(c)
            for c in keyword_raw.get("cases", []):
                cid = c.get("case_id", "")
                if cid not in seen_ids:
                    seen_ids.add(cid)
                    all_candidates.append(c)

            # LLM reranker: strict filter + contextual synthesis (prod f48bfd1)
            reranker_synthesis = ""
            if all_candidates:
                try:
                    rerank = self.llm.rerank_cases(question=msg_text, candidates=all_candidates)
                    relevant_ids = {r.case_id for r in rerank.relevant}
                    filtered = [c for c in all_candidates if c["case_id"] in relevant_ids]
                    relevance_map = {r.case_id: r.relevance for r in rerank.relevant}
                    filtered.sort(key=lambda c: (0 if relevance_map.get(c["case_id"]) == "direct" else 1))
                    reranker_synthesis = getattr(rerank, "synthesis", "") or ""
                    if filtered:
                        case_ans = self.case_agent.format_cases(filtered)
                    # If reranker says 0 relevant — trust it. Don't force-feed irrelevant cases.
                except Exception as exc:
                    log.warning("Reranker failed (%s), falling back to unfiltered top 5", exc)
                    case_ans = self.case_agent.format_cases(all_candidates[:5])

            # Build keyword_ans from negative evidence only (cases already merged above)
            negative_notes = keyword_raw.get("negative_notes", [])
            if negative_notes:
                keyword_ans = "\n".join(negative_notes)

            try:
                resp_text = self._synthesize(
                    question=msg_text, case_ans=case_ans, context=ctx,
                    lang=_detect_lang(msg_text), keyword_ans=keyword_ans,
                    gate_tag=tag.replace("→override", ""), images=images,
                    docs_ans=docs_ans, reranker_synthesis=reranker_synthesis,
                )
            except Exception as e:
                log.warning("Synthesizer failed for msg[%d]: %s", i, e)
                resp_text = "[[TAG_ADMIN]]"

            msg_text_short = msg_text[:60]
            if not resp_text or resp_text == "SKIP":
                actions[i] = {"action": "skip", "text": ""}
            elif "[[TAG_ADMIN]]" in resp_text:
                actions[i] = {"action": "escalate", "text": ""}
            else:
                actions[i] = {"action": "respond", "text": resp_text}
                bot_responses[i] = resp_text[:300]  # inject into context for next msgs
                print(f"    → msg[{i}] R {msg_text_short}")

        synth_time = time.time() - t_synth
        responded = sum(1 for a in actions if a["action"] == "respond")
        print(f"    Synth done: {responded} responses ({synth_time:.1f}s)")

        return actions

    def _gate(self, message: str, context: str,
              images: list[tuple[bytes, str]] | None = None):
        """Gate call — uses eval-adapted prompt that measures capability, not deployment policy.

        Production gate aggressively skips questions in active human discussions
        (correct for deployment: don't interrupt). For eval, we want to measure
        whether the bot CAN answer correctly, so we keep the core gate logic
        (filter noise/statements/advice) but remove the "active discussion" override.
        The synthesizer's own SKIP rules handle cases where it can't add value.
        """
        from app.llm.schemas import DecisionResult
        user = f"MESSAGE:\n{message}\n\nCONTEXT (unresolved discussions from buffer):\n{context}"
        return self.llm._json_call(
            model=self.settings.model_decision,
            system=P_DECISION_EVAL,
            user=user,
            schema=DecisionResult,
            images=images,
            cascade=["gemini-2.5-flash", "gemini-2.5-pro"],
        )

    def _keyword_search(self, question: str, context: str = "",
                         images: list[tuple[bytes, str]] | None = None) -> str:
        """KeywordAgent: keyword search over cases + synthesis LLM call.

        Replicates production KeywordAgent (keyword_agent.py):
        1. LLM extracts keywords from question
        2. Search cases + history messages for keyword matches
        3. LLM #2 synthesizes sub-answer from matched cases (with links)
        4. Negative evidence appended
        """
        try:
            kw = self.llm.extract_keywords(message=question)
        except Exception:
            return "No keyword matches."

        terms = list(dict.fromkeys(kw.keywords))[:5]
        if not terms:
            return "No keyword matches."

        # Search cases by keyword (replaces DB LIKE search + case_evidence JOIN)
        matched_cases = []
        for case in self._cases:
            text = f"{case.get('problem_title','')} {case.get('problem_summary','')} {case.get('solution_summary','')}".lower()
            if any(t.lower() in text for t in terms):
                matched_cases.append(case)

        # Negative evidence (replaces count_term_in_messages)
        negative_notes = []
        all_history_text = " ".join((m.get("body") or "") for m in self._history_msgs).lower()
        for term in terms:
            if term.lower() not in all_history_text:
                negative_notes.append(f"NOTE: '{term}' has ZERO mentions across community message history.")

        if not matched_cases and not negative_notes:
            return "No keyword matches."

        # Step 4: LLM #2 synthesizes sub-answer (matches production keyword_agent.py:98-121)
        sub_answer = ""
        if matched_cases:
            cases_text = self._format_keyword_cases(matched_cases[:10])
            synth_prompt = (
                "You are a keyword search sub-agent. Given the user's question and cases found "
                "by keyword search, prepare a concise sub-answer for the main synthesizer.\n\n"
                "RULES:\n"
                "1. Use ONLY information from the provided cases. Do NOT invent.\n"
                "2. Preserve ALL case links (URLs) — the synthesizer needs them for citations.\n"
                "3. If no case is relevant to the question — say so.\n"
                "4. Be concise: 2-5 sentences max.\n"
                "5. Focus on info that ADDS VALUE (exact models, specific configs, verified solutions).\n"
                "6. Respond in the same language as the question.\n"
                "7. No markdown formatting.\n\n"
                f"User question: \"{question}\"\n\n"
            )
            if context.strip():
                synth_prompt += f"Chat context:\n{context}\n\n"
            synth_prompt += f"Cases found by keyword search:\n{cases_text}\n"

            try:
                sub_answer = self.llm.chat(
                    prompt=synth_prompt,
                    cascade=EVAL_SUBAGENT_CASCADE,
                    timeout=30.0,
                    images=images,
                )
            except Exception:
                log.warning("Keyword synthesis LLM failed, using raw cases")
                sub_answer = cases_text  # fallback: raw case list

        parts = []
        if sub_answer:
            parts.append(sub_answer)
        if negative_notes:
            parts.append("\n".join(negative_notes))
        return "\n\n".join(parts) if parts else "No keyword matches."

    def _keyword_search_raw(self, question: str, context: str = "",
                            images: list[tuple[bytes, str]] | None = None) -> dict:
        """Return raw keyword search results: cases + negative_notes (for reranker merge).

        Matches production keyword_agent.py pipeline:
        1. LLM extracts keywords
        2. Search history messages for keyword matches → find cases containing those messages
        3. Also search cases directly by keyword (supplementary)
        4. Negative evidence for keywords with 0 mentions
        """
        try:
            kw = self.llm.extract_keywords(message=question)
        except Exception:
            return {"cases": [], "negative_notes": []}

        terms = list(dict.fromkeys(kw.keywords))[:5]
        if not terms:
            return {"cases": [], "negative_notes": []}

        # Step 2: Search history messages by keyword (matches prod SQL LIKE search)
        # Build message_id → case mapping from evidence_ids
        msg_to_cases: dict[str, list[int]] = {}
        for ci, case in enumerate(self._cases):
            for eid in (case.get("evidence_ids") or []):
                eid_str = str(eid)
                msg_to_cases.setdefault(eid_str, []).append(ci)

        matched_case_indices: set[int] = set()
        for m in self._history_msgs:
            body = (m.get("body") or "").lower()
            if any(t.lower() in body for t in terms):
                mid = m.get("id", "")
                # Check if this message is evidence for any case
                for ci in msg_to_cases.get(mid, []):
                    matched_case_indices.add(ci)

        # Step 3: Also search cases directly by keyword (supplementary)
        for ci, case in enumerate(self._cases):
            text = f"{case.get('problem_title','')} {case.get('problem_summary','')} {case.get('solution_summary','')}".lower()
            if any(t.lower() in text for t in terms):
                matched_case_indices.add(ci)

        # Normalize to common format (matches prod keyword_agent.py:118-128)
        matched_cases = []
        for ci in list(matched_case_indices)[:15]:
            case = self._cases[ci]
            matched_cases.append({
                "case_id": self._case_ids[ci],
                "source": "keyword",
                "status": case.get("status", "recommendation"),
                "problem": f"{case.get('problem_title', '')} — {case.get('problem_summary', '')[:200]}",
                "solution": case.get("solution_summary", "")[:300],
            })

        # Step 4: Negative evidence (matches prod count_term_in_messages)
        negative_notes = []
        all_history_text = " ".join((m.get("body") or "") for m in self._history_msgs).lower()
        for term in terms:
            if term.lower() not in all_history_text:
                negative_notes.append(f"NOTE: '{term}' has ZERO mentions across community message history.")

        return {"cases": matched_cases, "negative_notes": negative_notes}

    def _format_keyword_cases(self, cases: list[dict]) -> str:
        """Format cases with links (matches production KeywordAgent._format_cases)."""
        lines = []
        for i, c in enumerate(cases):
            status = c.get("status", "recommendation")
            prefix = "[Solved]" if status == "solved" else "[Recommendation]"
            link = f"{self.public_url}/case/{c.get('case_id', f'eval_{i}')}"
            lines.append(
                f"- {prefix} {c.get('problem_title', '???')}\n"
                f"  Problem: {c.get('problem_summary', '')[:200]}\n"
                f"  Solution: {c.get('solution_summary', '')[:300]}\n"
                f"  Link: {link}"
            )
        return "\n".join(lines)

    def _synthesize(self, question: str, case_ans: str, context: str, lang: str,
                    keyword_ans: str = "", gate_tag: str = "",
                    images: list[tuple[bytes, str]] | None = None,
                    docs_ans: str = "NO_DOCS",
                    reranker_synthesis: str = "") -> str:
        """Replicate UltimateAgent._synthesize — synced with production prompt."""
        case_has_results = case_ans and "No relevant cases" not in case_ans
        keyword_has_results = keyword_ans and keyword_ans != "No keyword matches."
        docs_has_results = docs_ans and docs_ans not in ("NO_DOCS", "SKIP", "INSUFFICIENT_INFO") and not docs_ans.startswith("INSUFFICIENT_INFO")
        has_reranker = bool(reranker_synthesis and reranker_synthesis.strip())

        # Match prod: no agent output → escalate or skip
        if not case_has_results and not keyword_has_results and not docs_has_results and not has_reranker:
            if gate_tag == "statement":
                log.info("No results for statement — skipping (no admin escalation)")
                return "SKIP"
            # For ongoing_discussion and new_question: attempt via Google Search
            # ongoing_discussion may still be a real question (e.g. follow-up "how does X work?")
            log.info("No case/keyword/docs results for %s — synthesizer will use Google Search", gate_tag)

        lang_instruction = _lang_instruction(lang)
        context_block = f"\nRecent chat context (for reference):\n{context}\n" if context.strip() else ""

        case_block = ""
        if case_has_results:
            if case_ans.startswith("B1_ONLY:"):
                case_block = f"\nCASE AGENT (recommendation cases — unconfirmed):\n{case_ans[len('B1_ONLY:'):]}"
            else:
                case_block = f"\nCASE AGENT (solved cases):\n{case_ans}"

        reranker_block = ""
        if reranker_synthesis:
            reranker_block = f"\nRERANKER ANALYSIS (expert pre-analysis of the cases):\n{reranker_synthesis}"

        keyword_block = ""
        if keyword_has_results:
            keyword_block = f"\nKEYWORD AGENT (cases found by keyword search in message history):\n{keyword_ans}"

        docs_block = ""
        if docs_has_results:
            docs_block = f"\nDOCS AGENT (from documentation):\n{docs_ans}"
        elif self.docs_urls:
            urls_list = "\n".join(f"- {u}" for u in self.docs_urls)
            docs_block = f"\nDOCUMENTATION (authoritative sources — search these FIRST for parameter values, configs, and technical details):\n{urls_list}\n"

        # Extract case URLs for explicit citation block (matches prod ultimate_agent.py)
        cite_urls_block = ""
        if case_has_results:
            case_urls = re.findall(
                rf'{re.escape(self.public_url)}/case/[a-f0-9]+', case_ans
            )
            if case_urls:
                seen = set()
                unique = []
                for u in case_urls:
                    if u not in seen:
                        seen.add(u)
                        unique.append(u)
                cite_urls_block = "\nCASE URLS — copy these at the end of your answer:\n" + "\n".join(unique)

        # Embed image markers in question text (matches prod ultimate_agent.py:319-323)
        question_with_images = question
        if images:
            markers = " ".join(f"[[IMG:{j}]]" for j in range(len(images)))
            question_with_images = f"{question}\n{markers}"

        # ── Production synthesizer prompt (exact copy from ultimate_agent.py:325-365) ──
        prompt = f"""You are a support bot in a group chat. Answer like a knowledgeable colleague — short, direct, no fluff.
{context_block}
Question: "{question_with_images}"
{case_block}
{reranker_block}
{keyword_block}
{docs_block}
RULES:

BREVITY (HIGHEST PRIORITY):
You are in a CHAT, not writing documentation. 2-4 sentences is ideal. No numbered lists, no step-by-step tutorials, no paragraphs of explanation. Merge info into flowing text. If you can say it in one sentence, do. When in doubt, cut it shorter.

CASES ARE YOUR PRIMARY SOURCE (CRITICAL):
Cases above are REAL solutions from this community — they are MORE RELIABLE than Google Search results. ALWAYS prefer case information over Google Search when cases are relevant. Google Search is for SUPPLEMENTING cases, not replacing them. If a case directly answers the question, your response MUST be based on that case.

RELEVANCE:
Cases have been PRE-FILTERED by an expert reranker. Use them if they match the question. The RERANKER ANALYSIS (if present) is an expert pre-analysis — use it as a guide. If no cases are provided or none match → [[TAG_ADMIN]]. Never mix fragments from unrelated cases. Check exact product/system names — similar-sounding ≠ same.

TOPIC COHERENCE:
Multiple threads may be interleaved in context. Identify which thread the current question belongs to by checking reply chains and topic keywords. NEVER mix advice from one thread into another — e.g. if the question is about ESC telemetry, do not give gimbal advice even if gimbal messages appear nearby.
Even pre-filtered cases may not match the specific sub-topic. Only use cases that address the user's actual question.

WHEN TO SKIP (use msg_id and → arrows to trace who is talking to whom):
- If a HUMAN (not [BOT]) already answered this question in context → output "SKIP"
- If YOUR previous response already covered this (same advice/info) → output "SKIP". Do NOT repeat yourself.
- If the question is directed at a specific person (→ arrow points to them) → output "SKIP"
- If a helper just shared a link, file, or direct answer and you would just echo/paraphrase it → output "SKIP". Do NOT "confirm" or "agree with" a helper's answer — that's noise.
- If a helper is asking the USER a diagnostic question ("is X enabled?", "what does Y show?", "скидує в нуль?", "mission complete каже?") → output "SKIP". HOW TO DETECT: the message has → pointing to the person who originally asked for help, and it contains "?". These need the USER's answer, not yours.
- If helpers are speculating/discussing among themselves and you have no NEW information to add → output "SKIP"
- If a helper states a conclusion ("значить проблемний юніт", "тоді треба міняти") → output "SKIP". This is their assessment, not a question for you.

ADMIN ESCALATION:
If answer needs admin help → add [[TAG_ADMIN]] at the END. Never say "contact admins" without [[TAG_ADMIN]].

CITATIONS (MANDATORY when using cases):
If CASE URLS are listed below, you MUST copy them at the END of your response. This is NON-NEGOTIABLE — every response that uses case information must end with the case URLs. Just copy them as-is from the CASE URLS block.

EXACT VALUES — CRITICAL:
NEVER output specific parameter values (RC_OPTION=X, SERIAL_PROTOCOL=X, OSD_TYPE=X, etc.) UNLESS they come from one of the CASE AGENT cases above. Cases are verified community knowledge. Google Search results and your training data are UNRELIABLE for specific parameter numbers — they may be outdated, for a different firmware version, or for a different product. If a case provides the exact value, use it and cite it. If NO case has the value, do NOT guess — say "точне значення параметра потрібно перевірити в документації" and add [[TAG_ADMIN]]. A wrong config value can brick hardware.

USER'S PLATFORM:
Pay attention to what firmware/hardware the user is ACTUALLY running. If context says "DJI" do not give analog OSD advice. If hardware model is unclear, ASK — do not assume.

FORMAT:
- Plain text only — no markdown (**bold**, *italic*, `code`, #headers). Signal doesn't render it.
- No greeting, no preamble, no "Based on...", no "According to..."
- Respond in {lang_instruction}
- Never invent information
- If user disagrees with your previous answer, ask for clarification — don't fabricate alternatives
- NEGATIVE EVIDENCE: if KEYWORD AGENT notes ZERO mentions of a product, state it
{cite_urls_block}
Answer:"""

        # Hard timeout wrapper with retry on 504 — chat_grounded with AFC can hang
        from concurrent.futures import ThreadPoolExecutor as _TP, TimeoutError as _TE
        raw_text = None
        for attempt in range(2):  # retry once on 504
            with _TP(max_workers=1) as _p:
                fut = _p.submit(_eval_chat_grounded, self.llm, prompt=prompt, timeout=90.0,
                                 cascade=EVAL_SYNTH_CASCADE, images=images, temperature=0.0,
                                 cost=self.cost)
                try:
                    raw_text = fut.result(timeout=200)
                    break
                except (_TE, Exception) as e:
                    if attempt == 0 and "504" in str(e):
                        log.info("Synthesizer 504, retrying (attempt 2)...")
                        time.sleep(2)
                        continue
                    log.warning("Synthesizer hard timeout: %s", e)
                    return "[[TAG_ADMIN]]"
        if not raw_text:
            return "[[TAG_ADMIN]]"
        # Post-process (synced from ultimate_agent.py:369-397)
        clean_text = _ATTACH_PATTERN.sub("", raw_text).strip()
        # Parse reply-to target
        reply_match = _REPLY_TO_PATTERN.search(clean_text)
        if reply_match:
            clean_text = _REPLY_TO_PATTERN.sub("", clean_text).strip()
        # Fix [cite: case_id] → proper URL
        clean_text = _CITE_PATTERN.sub(
            lambda m: f"{self.public_url}/case/{m.group(1)}", clean_text
        )
        clean_text = _CITE_BROAD_PATTERN.sub("", clean_text)
        # Strip inline bracket citations (truncated IDs create broken URLs)
        clean_text = _INLINE_CASE_BRACKET.sub("", clean_text)
        # Safety net: bare hex32 IDs → URLs
        clean_text = _BARE_HEX32.sub(
            lambda m: f"{self.public_url}/case/{m.group(1).lower()}", clean_text
        )
        # Strip markdown formatting
        clean_text = re.sub(r'\*\*(.+?)\*\*', r'\1', clean_text)
        clean_text = re.sub(r'\*(.+?)\*', r'\1', clean_text)
        clean_text = re.sub(r'`(.+?)`', r'\1', clean_text)
        clean_text = re.sub(r'^#{1,6}\s+', '', clean_text, flags=re.MULTILINE)
        # Fix lines broken at hyphens (matches prod)
        clean_text = re.sub(r'-\n(\S)', r'-\1', clean_text)
        # Ensure case links end with .html for GitHub Pages compatibility
        clean_text = re.sub(r'(/case/[a-f0-9]{12})(?!\.html)(\b|$|\)|\s)', r'\1.html\2', clean_text)
        # Guard: detect leaked chain-of-thought (English reasoning in non-English group)
        if self.lang in ("uk", "es") and clean_text and len(clean_text) > 50:
            detected = _detect_lang(clean_text[:200])
            if detected == "en":
                log.warning("Synthesizer leaked English CoT for %s group, skipping", self.lang)
                return "SKIP"
        return clean_text

    # Keep old per-message interface for compatibility with run_eval
    def process(self, message: dict, prior_live: list[dict]) -> dict:
        """Legacy per-message interface — not used when process_batch is available."""
        raise NotImplementedError("Use process_batch instead")


# ═══════════════════════════════════════════════════════════════════════════════
#  System 2: Baselines (LLM-Conservative, LLM-Aggressive, Chunked-RAG)
# ═══════════════════════════════════════════════════════════════════════════════

BASELINE_PROMPT_CONSERVATIVE = """You are a technical support bot in a group chat.

For each new message, decide:
- If it's a support question you can answer from the chat history: respond concisely (1-3 sentences). Cite specific earlier messages or solutions mentioned in history.
- If it's NOT a question (greeting, acknowledgment, statement, off-topic): output exactly "SKIP"
- If it's a question but you lack information to answer: output exactly "SKIP"

No greetings, no preamble, no restating the question. Direct answer only."""

BASELINE_PROMPT_AGGRESSIVE = """You are a technical support bot in a group chat.

For each new message, decide:
- If it's a support question: respond concisely (1-3 sentences) using the chat history as context. If you lack information, give your best answer based on general knowledge.
- If it's NOT a question (greeting, acknowledgment, statement, off-topic): output exactly "SKIP"

No greetings, no preamble, no restating the question. Direct answer only."""

# All baselines use the same model as SupportBot synthesizer for fair comparison
BASELINE_MODEL = "gemini-2.5-pro"

# Chunked-RAG settings
CHUNKED_RAG_CHUNK_SIZE = 10  # messages per chunk
CHUNKED_RAG_TOP_K = 5        # chunks retrieved per query


class BaselineSystem:
    """Context-stuffing LLM baseline. Two variants:
    - conservative: skips questions it can't answer from context (high quality, low recall)
    - aggressive: attempts all questions (lower quality, high recall)
    """
    def __init__(self, settings: Settings, dataset_dir: Path | None = None,
                 variant: str = "aggressive"):
        self.llm = LLMClient(settings)
        self.settings = settings
        self.dataset_dir = dataset_dir
        self._history: list[dict] = []
        self.num_cases = 0
        self.cost = CostTracker()
        self.variant = variant
        self._prompt = (BASELINE_PROMPT_CONSERVATIVE if variant == "conservative"
                        else BASELINE_PROMPT_AGGRESSIVE)

    def ingest(self, messages: list[dict]) -> None:
        self._history = list(messages)
        print(f"    Stored {len(messages)} messages (no processing)")

    def process(self, message: dict, prior_live: list[dict]) -> dict:
        from google.genai import types as _gt

        msg_text = message.get("body") or ""
        all_context = self._history + prior_live
        lines = []
        chars = 0
        for m in reversed(all_context):
            line = f"{m['sender']}: {m.get('body', '')}"
            chars += len(line)
            if chars > BASELINE_CONTEXT_CHARS:
                break
            lines.insert(0, line)

        user = f"CHAT HISTORY:\n" + "\n".join(lines) + f"\n\nNEW MESSAGE:\n{msg_text}\n\nYour response:"

        # Load images from current message (multimodal, same as SupportBot)
        images: list[tuple[bytes, str]] = []
        if self.dataset_dir:
            loaded = _load_image(message, self.dataset_dir)
            if loaded:
                images = loaded

        try:
            client = self.llm._genai_client
            contents = [user]
            if images:
                for img_bytes, img_mime in images[:3]:
                    contents.append(_gt.Part.from_bytes(data=img_bytes, mime_type=img_mime))
            response = client.models.generate_content(
                model=BASELINE_MODEL,
                contents=contents,
                config=_gt.GenerateContentConfig(
                    system_instruction=self._prompt,
                    temperature=0.0,
                    http_options=_gt.HttpOptions(timeout=60_000),
                ),
            )
            self.cost.add_from_genai_response(BASELINE_MODEL, response)
            text = (response.text or "").strip()
        except Exception as e:
            log.warning("Baseline LLM failed: %s", e)
            return {"action": "skip", "text": ""}

        if not text or text.upper() == "SKIP":
            return {"action": "skip", "text": ""}
        return {"action": "respond", "text": text}


CHUNKED_RAG_PROMPT = """You are a technical support bot in a group chat.

You have access to retrieved context chunks from earlier conversations that may be relevant.
Each chunk has a reference link — cite the most relevant chunk at the end of your response.

For each new message, decide:
- If it's a support question: respond concisely (1-3 sentences) using the retrieved context. If the context doesn't help, give your best answer based on general knowledge. End with the link to the most relevant chunk.
- If it's NOT a question (greeting, acknowledgment, statement, off-topic): output exactly "SKIP"

No greetings, no preamble, no restating the question. Direct answer only."""


class ChunkedRAGSystem:
    """Standard chunked-RAG baseline: chunk history into fixed-size blocks,
    embed into ChromaDB, retrieve top-k per query, stuff into LLM context.
    Each chunk gets a reference URL for citation."""

    def __init__(self, settings: Settings, dataset_dir: Path | None = None):
        self.llm = LLMClient(settings)
        self.settings = settings
        self.dataset_dir = dataset_dir
        self.num_cases = 0
        self.cost = CostTracker()
        self._chroma = chromadb.EphemeralClient()
        self._collection = self._chroma.get_or_create_collection(
            name="chunked_rag_baseline")
        self._chunk_map: dict[str, str] = {}  # chunk_id → text (for serving)

    @staticmethod
    def _msg_label(m: dict, id_to_sender: dict) -> str:
        sender = f"User{m['sender'][:6]}"
        mid = m.get("id", "")
        reply_id = m.get("reply_to_id")
        if reply_id and reply_id in id_to_sender:
            target = f"User{id_to_sender[reply_id][:6]}"
            return f"[{sender} msg_id={mid} → {target} msg_id={reply_id}]"
        return f"[{sender} msg_id={mid}]"

    def ingest(self, messages: list[dict]) -> None:
        """Chunk messages into fixed-size blocks, embed, index."""
        self._id_to_sender = {m["id"]: m["sender"] for m in messages}
        chunks = []
        for i in range(0, len(messages), CHUNKED_RAG_CHUNK_SIZE):
            block = messages[i:i + CHUNKED_RAG_CHUNK_SIZE]
            text = "\n".join(
                f"{self._msg_label(m, self._id_to_sender)}: {m.get('body', '')}" for m in block
            )
            cid = f"chunk_{i}"
            chunks.append({"id": cid, "text": text})
            self._chunk_map[cid] = text

        print(f"    Chunking: {len(messages)} msgs → {len(chunks)} chunks "
              f"({CHUNKED_RAG_CHUNK_SIZE} msgs/chunk)")

        # Batch embed
        texts = [c["text"] for c in chunks]
        embeddings = self.llm.embed_batch(texts=texts)

        self._collection.add(
            ids=[c["id"] for c in chunks],
            documents=texts,
            embeddings=embeddings,
        )
        print(f"    Indexed {len(chunks)} chunks into ChromaDB")

    def process(self, message: dict, prior_live: list[dict]) -> dict:
        from google.genai import types as _gt

        msg_text = message.get("body") or ""
        if not msg_text.strip():
            return {"action": "skip", "text": ""}

        # Retrieve top-k chunks
        query_emb = self.llm.embed(text=msg_text)
        results = self._collection.query(
            query_embeddings=[query_emb],
            n_results=CHUNKED_RAG_TOP_K,
            include=["documents"],
        )
        retrieved_docs = (results.get("documents") or [[]])[0]
        retrieved_ids = (results.get("ids") or [[]])[0]

        # Format with chunk links
        chunk_sections = []
        for cid, doc in zip(retrieved_ids, retrieved_docs):
            link = f"{os.environ.get('EVAL_PUBLIC_URL', 'https://pavelshpagin.github.io/SupportBot')}/chunk/{cid}.html"
            chunk_sections.append(f"[{cid}] ({link}):\n{doc}")
        context = "\n---\n".join(chunk_sections)

        # Also include recent live messages for conversational context
        recent = prior_live[-10:] if prior_live else []
        recent_text = "\n".join(
            f"{self._msg_label(m, self._id_to_sender)}: {m.get('body', '')}" for m in recent
        )

        user = (f"RETRIEVED CONTEXT:\n{context}\n\n"
                f"RECENT MESSAGES:\n{recent_text}\n\n"
                f"NEW MESSAGE:\n{msg_text}\n\nYour response:")

        images: list[tuple[bytes, str]] = []
        if self.dataset_dir:
            loaded = _load_image(message, self.dataset_dir)
            if loaded:
                images = loaded

        try:
            client = self.llm._genai_client
            contents = [user]
            if images:
                for img_bytes, img_mime in images[:3]:
                    contents.append(_gt.Part.from_bytes(data=img_bytes, mime_type=img_mime))
            response = client.models.generate_content(
                model=BASELINE_MODEL,
                contents=contents,
                config=_gt.GenerateContentConfig(
                    system_instruction=CHUNKED_RAG_PROMPT,
                    temperature=0.0,
                    http_options=_gt.HttpOptions(timeout=60_000),
                ),
            )
            self.cost.add_from_genai_response(BASELINE_MODEL, response)
            text = (response.text or "").strip()
        except Exception as e:
            log.warning("ChunkedRAG LLM failed: %s", e)
            return {"action": "skip", "text": ""}

        if not text or text.upper() == "SKIP":
            return {"action": "skip", "text": ""}
        return {"action": "respond", "text": text}


# ═══════════════════════════════════════════════════════════════════════════════
#  Judge — architecture-agnostic, parallelized
# ═══════════════════════════════════════════════════════════════════════════════

JUDGE_QUALITY_SYSTEM = """You are evaluating a support bot's response in a technical group chat.
You are an impartial external observer — like an admin reviewing the bot's work.
You see the full chat history, the bot's response, and what happened after.

IMPORTANT: Use Google Search to fact-check the bot's technical claims. Search for specific product names, settings, commands, and technical details mentioned in the bot's response. If the bot states something that contradicts search results, mark correctness LOW.

Rate the bot's response on five dimensions (each 0-10):

- correctness: Factually correct AND verifiable for the SPECIFIC product/setup? Search for exact combination. Unverified claims → ≤7. Hallucinated = 0.
- helpfulness: Resolves the user's SPECIFIC problem? Check AFTER context. Generic advice → ≤6.
- specificity: Addresses THIS user's exact product/setup (9-10) vs generic advice (5-7) vs wrong product (0-3)?
- necessity: Should a bot have responded? Direct question → 9-10. Helper advice/statement → 0-3.
- sourcing: Does the response cite verifiable sources (URLs, documentation links, community case links)? Cited and relevant → 9-10. Cited but generic/irrelevant → 5-7. No sources at all → 3-4. Made-up/broken URLs → 0.

Respond ONLY with JSON:
{"correctness": N, "helpfulness": N, "specificity": N, "necessity": N, "sourcing": N, "reasoning": "brief"}"""

JUDGE_COVERAGE_SYSTEM = """You are analyzing a segment of a technical support group chat to identify support questions that an AI bot should answer.

Count as support questions:
- Direct questions asking for help ("How do I...?", "Why doesn't...?", "What is...?")
- Problem descriptions that need solving ("doesn't work", "can't find X", "error when...") — even without "?"
- Follow-ups where the user is still stuck ("updated, nothing helped", "still broken")
- Configuration/parameter questions ("what should ARSPD_USE be?", "which firmware version?")
- Requests for recommendations, documentation, or product suggestions
- Technical hypotheses phrased as questions ("Maybe TECS?", "Could it be X?")
- Short follow-up questions with enough technical content to answer ("which ESC?", "on drone or plane?")
- Messages describing a need or requirement ("I need X and Y to work simultaneously", "need to set up 2 channels on one switch")
- Messages describing setup behavior with an IMPLICIT question ("in positions 1-2 it works as X, but position 3..." — the user is asking about configuration)
- Wondering about fixes or configuration ("was thinking if this can be fixed with parameters")

Do NOT count:
- Rhetorical questions, greetings, jokes, memes, off-topic
- Pure acknowledgments ("ok", "thanks", "got it")
- ADVICE or ANSWERS from helpers: instructions telling someone what to do ("use X", "try Y", "check Z", "make a curve") — these are answers, not questions
- Diagnostic questions from helpers asking for USER-SPECIFIC info ("what does YOUR screen show?", "paste your log", "is X enabled?" directed at another user)
- Pure observations without any implicit request for help ("this issue is in simulator too", "I suspect it's EKF")
- Experience sharing without asking for help ("I also had this issue", "for me it was the same")
- Very short context-dependent follow-ups (≤5 words) that are only meaningful in the specific thread ("really?", "and then?", "which one?")

For each question found, note the message ID and a short excerpt.

Respond with JSON:
{"questions": [{"msg_id": "...", "text": "first 100 chars..."}]}"""


def _judge_quality_one(llm: LLMClient, message: dict, bot_text: str,
                       history_before: list[dict], messages_after: list[dict],
                       images: list[tuple[bytes, str]] | None = None) -> dict:
    context = history_before[-50:]
    history_text = "\n".join(
        f"[{m['sender']} ts={m['ts']}"
        + (f" reply_to={m['reply_to_id']}" if m.get('reply_to_id') else "")
        + f"] {m.get('body', '')}"
        for m in context
    )
    after_text = "\n".join(
        f"[{m['sender']} ts={m['ts']}"
        + (f" reply_to={m['reply_to_id']}" if m.get('reply_to_id') else "")
        + f"] {m.get('body', '')}"
        for m in messages_after[:20]
    )
    img_note = "\n\n(The target message included an image — shown below)" if images else ""
    prompt = (
        f"{JUDGE_QUALITY_SYSTEM}\n\n"
        f"CHAT HISTORY (last {len(context)} messages):\n{history_text}\n\n"
        f"TARGET MESSAGE:\n[{message['sender']} ts={message['ts']}] {message.get('body', '')}{img_note}\n\n"
        f"BOT RESPONSE:\n{bot_text}\n\n"
        f"SUBSEQUENT MESSAGES:\n{after_text}"
    )
    # Use chat_grounded so the judge can Google-search to verify bot claims
    raw = _retry_on_429(
        _eval_chat_grounded, llm,
        prompt=prompt,
        timeout=60.0,
        cascade=[JUDGE_MODEL],
        images=images,
        temperature=0.0,
    )
    # Extract JSON from response (may have markdown fences, control chars, etc.)
    def _safe_parse(text: str) -> dict | None:
        if not text or not text.strip():
            return None
        # Strip control characters that break JSON parsing
        text = re.sub(r'[\x00-\x1f\x7f]', ' ', text)
        # Try regex extraction first
        json_match = re.search(r'\{[^{}]*"correctness"[^{}]*\}', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        # Try stripping markdown fences
        cleaned = re.sub(r'```json\s*', '', text)
        cleaned = re.sub(r'```\s*', '', cleaned).strip()
        if cleaned:
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                pass
        return None

    result = _safe_parse(raw)
    if result:
        return result

    # Fallback: non-grounded call with JSON mode
    log.info("Grounded judge returned empty/unparseable, falling back to non-grounded")
    resp = _retry_on_429(
        llm.client.chat.completions.create,
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": JUDGE_QUALITY_SYSTEM},
            {"role": "user", "content": prompt.replace(JUDGE_QUALITY_SYSTEM + "\n\n", "")},
        ],
        response_format={"type": "json_object"},
        timeout=45.0,
    )
    fallback_raw = resp.choices[0].message.content or "{}"
    result = _safe_parse(fallback_raw)
    return result or json.loads(fallback_raw)


def _judge_coverage_chunk(llm: LLMClient, chunk: list[dict]) -> list[dict]:
    """Identify ground-truth questions in a RAW message chunk (no bot responses)."""
    lines = []
    for m in chunk:
        reply = f" reply_to={m['reply_to_id']}" if m.get("reply_to_id") else ""
        lines.append(f"[{m['sender']} ts={m['ts']} msg_id={m['id']}{reply}] {m.get('body', '')}")
    user = "CHAT SEGMENT:\n" + "\n".join(lines)
    resp = _retry_on_429(
        llm.client.chat.completions.create,
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": JUDGE_COVERAGE_SYSTEM},
            {"role": "user", "content": user},
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
        timeout=45.0,
    )
    result = json.loads(resp.choices[0].message.content or '{"questions": []}')
    return result.get("questions", [])


# ═══════════════════════════════════════════════════════════════════════════════
#  Settings
# ═══════════════════════════════════════════════════════════════════════════════

def _make_settings() -> Settings:
    from app.config import _detect_vertexai
    gak = os.environ["GOOGLE_API_KEY"]
    vp, vl = _detect_vertexai(gak)
    return Settings(
        db_backend="mysql",
        mysql_host="", mysql_port=3306, mysql_user="", mysql_password="", mysql_database="",
        oracle_user="", oracle_password="", oracle_dsn="", oracle_wallet_dir="",
        openai_api_key=gak,
        openai_key=_env("OPENAI_KEY", default=""),
        vertexai_project=vp, vertexai_location=vl,
        model_img=_env("MODEL_IMG", default="gemini-2.5-pro"),
        model_decision=_env("MODEL_DECISION", default="gemini-2.5-flash"),
        model_extract=_env("MODEL_EXTRACT", default="gemini-2.5-pro"),
        model_case=_env("MODEL_CASE", default="gemini-2.5-pro"),
        model_respond=_env("MODEL_RESPOND", default="gemini-2.5-pro"),
        model_blocks=_env("MODEL_BLOCKS", default="gemini-2.5-pro"),
        embedding_model=_env("EMBEDDING_MODEL", default="gemini-embedding-001"),
        chroma_url="http://localhost:9999",
        chroma_collection="eval",
        signal_bot_e164="+10000000000",
        signal_bot_storage="", signal_ingest_storage="", signal_cli="",
        bot_mention_strings=[], signal_listener_enabled=False,
        signal_link_timeout_seconds=60,
        use_signal_desktop=False, signal_desktop_url="",
        telegram_bot_token="", telegram_bot_username="", telegram_listener_enabled=False,
        log_level="INFO", context_last_n=40, retrieve_top_k=5,
        worker_poll_seconds=1.0, worker_enabled=False, history_token_ttl_minutes=240,
        admin_session_stale_minutes=30, http_debug_endpoints_enabled=False,
        buffer_max_age_hours=168, buffer_max_messages=300,
        max_images_per_gate=3, max_images_per_respond=5,
        max_kb_images_per_case=2, max_image_size_bytes=5_000_000,
        max_total_image_bytes=20_000_000,
        public_url=os.environ.get("EVAL_PUBLIC_URL", "https://pavelshpagin.github.io/SupportBot"),
        admin_whitelist=[], superadmin_list=[],
        whatsapp_bridge_url="", whatsapp_enabled=False,
        stripe_secret_key="", stripe_webhook_secret="",
        stripe_starter_price_id="", stripe_unlimited_price_id="",
        paddle_api_key="", paddle_webhook_secret="", paddle_client_token="",
        paddle_starter_price_id="", paddle_unlimited_price_id="",
        paddle_environment="sandbox",
        discord_bot_token="", discord_enabled=False,
        slack_bot_token="", slack_app_token="", slack_signing_secret="",
        slack_enabled=False, slack_socket_mode=False,
    )


def _load_dataset(name: str) -> dict:
    path = DATASETS_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    with open(path) as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════════════════════
#  Main evaluation — parallelized
# ═══════════════════════════════════════════════════════════════════════════════

def run_eval(system_name: str, system, judge_llm: LLMClient,
             history_msgs: list[dict], live_msgs: list[dict],
             cases_cache: str | None = None,
             coverage_cache: str | None = None,
             dataset_dir: Path | None = None) -> dict:

    # ── Phase 1: Ingest ──────────────────────────────────────────────────
    print("  Phase 1: Ingest")
    t0 = time.time()
    if hasattr(system, 'ingest'):
        import inspect
        if 'cases_cache' in inspect.signature(system.ingest).parameters:
            system.ingest(history_msgs, cases_cache=cases_cache)
        else:
            system.ingest(history_msgs)
    ingest_time = time.time() - t0
    print(f"    Done ({ingest_time:.1f}s)")

    # ── Phase 2: Process live messages ───────────────────────────────────
    print(f"  Phase 2: Processing {len(live_msgs)} live messages...")
    actions: list[dict] = []
    t1 = time.time()

    if system_name in ("baseline", "baseline-conservative", "chunked-rag"):
        # Baselines can be fully parallelized — each msg gets full history context
        def _process_one(i_msg):
            i, msg = i_msg
            return i, system.process(msg, live_msgs[:i])

        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(_process_one, (i, msg)) for i, msg in enumerate(live_msgs)]
            results_map = {}
            for f in as_completed(futures):
                try:
                    i, result = f.result()
                    results_map[i] = result
                except Exception as e:
                    pass
        for i, msg in enumerate(live_msgs):
            result = results_map.get(i, {"action": "skip", "text": ""})
            actions.append({"msg": msg, **result})
    else:
        # SupportBot: batch gate → case search → synthesizer (matches production)
        batch_actions = system.process_batch(live_msgs, history_msgs)
        for i, (msg, result) in enumerate(zip(live_msgs, batch_actions)):
            actions.append({"msg": msg, **result})

    process_time = time.time() - t1
    responded = [a for a in actions if a["action"] == "respond"]
    escalated = [a for a in actions if a["action"] == "escalate"]
    skipped_n = len(live_msgs) - len(responded) - len(escalated)
    print(f"    {len(responded)}R {len(escalated)}E {skipped_n}S ({process_time:.1f}s)")

    # ── Phase 3: Judge quality + coverage IN PARALLEL ────────────────────
    print(f"  Phase 3: Judging ({len(responded)} quality + coverage)...")
    t2 = time.time()

    all_msgs = history_msgs + live_msgs
    quality_results: list[dict] = [None] * len(responded)  # type: ignore

    # Coverage: load from cache or judge fresh (stabilizes precision/recall across runs)
    cached_coverage = None
    if coverage_cache and Path(coverage_cache).exists():
        with open(coverage_cache) as f:
            cached_coverage = json.load(f)
        print(f"    Loaded {len(cached_coverage)} cached questions from {coverage_cache}")

    coverage_chunk_size = 50
    coverage_chunks = []
    if cached_coverage is None:
        for start in range(0, len(live_msgs), coverage_chunk_size):
            coverage_chunks.append(live_msgs[start:start + coverage_chunk_size])

    all_questions: list[list[dict]] = [None] * len(coverage_chunks)  # type: ignore

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        # Quality futures
        quality_futures = {}
        for i, a in enumerate(responded):
            msg = a["msg"]
            msg_idx = next(j for j, m in enumerate(live_msgs) if m["id"] == msg["id"])
            global_idx = len(history_msgs) + msg_idx
            before = all_msgs[:global_idx]
            after = all_msgs[global_idx + 1:global_idx + 21]
            # Load image for multimodal judge (if available)
            judge_images = _load_image(msg, dataset_dir) if dataset_dir else None
            f = pool.submit(_judge_quality_one, judge_llm, msg, a["text"], before, after,
                            images=judge_images)
            quality_futures[f] = i

        # Coverage futures — on raw stream (skip if cached)
        coverage_futures = {}
        if cached_coverage is None:
            for ci, chunk in enumerate(coverage_chunks):
                f = pool.submit(_judge_coverage_chunk, judge_llm, chunk)
                coverage_futures[f] = ci

        # Collect quality results
        for f in as_completed(quality_futures):
            i = quality_futures[f]
            try:
                j = f.result()
            except Exception as e:
                log.warning("Quality judge %d failed: %s", i, e)
                j = {}
            scores = {k: j.get(k) for k in ("correctness", "helpfulness", "specificity", "necessity")}
            vals = [v for v in scores.values() if v is not None]
            mean_q = statistics.mean(vals) if vals else 0.0
            msg = responded[i]["msg"]
            quality_results[i] = {
                "msg_id": msg["id"],
                "msg_text": (msg.get("body") or "")[:200],
                "bot_text": responded[i]["text"][:2000],
                **scores,
                "quality": round(mean_q, 2),
                "reasoning": j.get("reasoning", ""),
            }
            print(f"    Quality [{i+1}/{len(responded)}] q={mean_q:.1f}")

        # Collect coverage results
        for f in as_completed(coverage_futures):
            ci = coverage_futures[f]
            try:
                all_questions[ci] = f.result()
            except Exception as e:
                log.warning("Coverage chunk %d failed: %s", ci, e)
                all_questions[ci] = []

    judge_time = time.time() - t2

    # Flatten ground-truth questions
    if cached_coverage is not None:
        questions_flat = cached_coverage
    else:
        questions_flat = []
        for qs in all_questions:
            if qs:
                questions_flat.extend(qs)
        # Save coverage cache
        if coverage_cache and questions_flat:
            Path(coverage_cache).parent.mkdir(parents=True, exist_ok=True)
            with open(coverage_cache, "w") as f:
                json.dump(questions_flat, f, ensure_ascii=False, indent=2)
            print(f"    Saved {len(questions_flat)} questions to {coverage_cache}")

    # ── Compute metrics ──────────────────────────────────────────────────
    valid_quality = [r for r in quality_results if r is not None]
    all_q = [r["quality"] for r in valid_quality]
    quality = statistics.mean(all_q) if all_q else 0.0

    # Build set of ground-truth question msg_ids (coverage judge only, no augmentation)
    live_ids = {m["id"] for m in live_msgs}
    question_ids = {q["msg_id"] for q in questions_flat if q["msg_id"] in live_ids}

    # Build set of msg_ids the system acted on
    responded_ids = {a["msg"]["id"] for a in actions if a["action"] == "respond"}
    escalated_ids = {a["msg"]["id"] for a in actions if a["action"] == "escalate"}
    handled_ids = responded_ids | escalated_ids

    # Precision/recall against coverage judge ground truth only
    tp_responses = len(responded_ids & question_ids)
    fp_responses = len(responded_ids - question_ids)
    covered_questions = len(question_ids & handled_ids)
    missed_questions = len(question_ids - handled_ids)

    total_questions = len(question_ids)
    precision = tp_responses / len(responded_ids) if responded_ids else 1.0
    recall = covered_questions / total_questions if total_questions > 0 else 1.0

    score = quality * recall  # headline metric: quality × recall

    print(f"    Questions: {total_questions} | TP={tp_responses} FP={fp_responses} Missed={missed_questions}")
    print(f"    Precision={precision:.1%} Recall={recall:.1%}")
    print(f"    Quality={quality:.2f} Score={score:.2f} ({judge_time:.1f}s)")

    total_time = ingest_time + process_time + judge_time

    return {
        "score": round(score, 2),
        "quality": round(quality, 2),
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "quality_breakdown": {
            k: round(statistics.mean([r[k] for r in valid_quality if r.get(k) is not None]), 2)
            for k in ("correctness", "helpfulness", "specificity", "necessity")
            if any(r.get(k) is not None for r in valid_quality)
        },
        "counts": {
            "history_msgs": len(history_msgs),
            "live_msgs": len(live_msgs),
            "cases_indexed": getattr(system, "num_cases", 0),
            "responded": len(responded),
            "escalated": len(escalated),
            "skipped": skipped_n,
            "questions_found": total_questions,
            "true_positives": tp_responses,
            "false_positives": fp_responses,
            "missed": missed_questions,
        },
        "timing": {
            "ingest_s": round(ingest_time, 1),
            "process_s": round(process_time, 1),
            "judge_s": round(judge_time, 1),
            "total_s": round(total_time, 1),
        },
        "cost": system.cost.summary() if hasattr(system, "cost") else {},
        "responses": valid_quality,
        "questions": questions_flat,
    }


## ═════════════════════════════════════════════════════════════════════════════
##  V2 Judge — unified chunked evaluation
## ═════════════════════════════════════════════════════════════════════════════
#
# 1. Bot processes messages sequentially (sees its own prior responses)
# 2. Live messages split into non-overlapping chunks (~50 msgs each)
# 3. ONE judge per chunk does EVERYTHING:
#    - Scores each bot response (correctness/faithfulness/helpfulness/conciseness)
#    - Identifies missed questions
#    - Identifies redundant bot responses
#    Each chunk gets context from previous/next chunks for continuity.
# 4. Results merged across chunks, deduplicated by msg_id
#
# Precision = (responded - redundant) / responded
# Recall    = (responded - redundant) / (responded - redundant + missed)

JUDGE_V2_CHUNK_SYSTEM = """You are a RIGOROUS evaluator of a support bot in a technical group chat.
You see a segment of the conversation with bot responses ([BOT]) interleaved.
Context from before and after the segment is provided for continuity.

Use Google Search EXTENSIVELY to fact-check. You are an expert auditor — do not give benefit of the doubt.

Do THREE things for the EVALUATE section only (ignore CONTEXT sections):

1. SCORE each [BOT] response on five dimensions (0-10):
   - correctness: Factually correct AND verifiable for the SPECIFIC product/setup?
     * For each technical claim, search for the EXACT combination (e.g., "Hobbywing 150A v5 ArduPilot telemetry", not just "ArduPilot ESC telemetry").
     * If the bot says "X works with Y" but you cannot find direct confirmation for that SPECIFIC combination → score ≤6.
     * Confident assertions without caveats about unverified specifics → score ≤7.
     * Generic advice that is technically true but does not address the user's specific hardware/firmware → score ≤7.
     * Hallucinated facts, wrong product/protocol, or unverifiable claims = 0-3.
   - helpfulness: Does the response resolve the user's SPECIFIC problem?
     * CRITICAL: check CONTEXT AFTER — if users report the advice didn't work, or experts correct the bot → reduce to ≤5.
     * Generic advice ("check your settings", "see documentation") when user asked a specific question → score ≤6.
     * Addresses the exact situation with actionable steps → 9-10.
   - specificity: How well does the response address THIS user's EXACT situation?
     * Bot names the exact product, firmware version, parameter values verified for that product → 9-10.
     * Bot gives correct general advice that MIGHT apply but doesn't confirm for this specific case → 5-7.
     * Bot gives advice for the wrong product/protocol entirely → 0-3.
   - necessity: Should a support bot have responded to this message AT ALL?
     * Direct support question → 9-10.
     * Implicit question or problem description → 7-9.
     * Follow-up in a thread the bot is already helping → 6-8.
     * Short follow-up (≤5 words) that humans are handling → 3-5.
     * Message from a HELPER giving advice, statement, confirmation, or observation → 0-3.
     A good support bot is SELECTIVE — responding to everything is spammy.
   - sourcing: Does the response cite verifiable sources (URLs, documentation links, community case links)?
     * Cited and relevant → 9-10. Cited but generic/irrelevant → 5-7. No sources at all → 3-4. Made-up/broken URLs → 0.

2. MISSED — support questions where the bot did NOT respond (regardless of human answers):
   - User asks for help, describes a problem, or requests information
   - No [BOT] response follows addressing their question
   - Substantial question (not just "ok?" or "really?")
   - Count as MISSED even if another human answered — we are measuring the BOT's coverage

   NOT missed: messages answered by [BOT], greetings/thanks,
   advice FROM helpers ("try X"), diagnostic questions from helpers ("is X enabled?"),
   observations/experience sharing, very short (≤3 words) ambiguous follow-ups,
   messages that are clearly part of an ongoing thread already addressed by [BOT].

3. REDUNDANT — [BOT] responses that should NOT have been sent:
   - Bot responded to a statement, advice from a helper, or observation — not a question
   - Bot repeated information it already gave
   - Human already gave a complete correct answer and bot adds nothing new
   - Bot responded to helper-to-helper exchange
   - Bot responded to noise (greetings, acknowledgments, short confirmations)
   - Bot responded to a ≤5 word follow-up in a thread humans are actively handling

   NOT redundant: answering genuine NEW support questions, adding NEW information beyond humans,
   answering implicit questions (problem descriptions), correcting incomplete human answers.

Respond with JSON:
{"scores": [{"msg_id": "...", "correctness": N, "helpfulness": N, "specificity": N, "necessity": N, "sourcing": N, "reasoning": "brief — what you searched, whether confirmed for specific product, whether bot should have responded"}],
 "missed": [{"msg_id": "...", "text": "first 100 chars...", "reason": "..."}],
 "redundant": [{"msg_id": "...", "reason": "..."}]}

scores array must have one entry per [BOT] response in the EVALUATE section.
msg_id must match the msg_id shown in the conversation."""

JUDGE_V2_CHUNK_SIZE = 50  # messages per chunk
JUDGE_V2_CONTEXT_SIZE = 20  # context messages before/after chunk


def _format_msg_line(msg: dict, actions: list[dict] | None = None,
                      action_idx: int | None = None,
                      include_msg_id: bool = False,
                      dataset_dir: Path | None = None) -> tuple[str, list[tuple[bytes, str]]]:
    """Format a message line with optional bot response. Returns (text, images)."""
    reply = f" reply_to={msg['reply_to_id']}" if msg.get("reply_to_id") else ""
    mid = f" msg_id={msg['id']}" if include_msg_id else ""
    images = []
    img_marker = ""
    if dataset_dir:
        loaded = _load_image(msg, dataset_dir)
        if loaded:
            images = loaded
            img_marker = " [[IMG]]"
    line = f"[{msg['sender']} ts={msg['ts']}{mid}{reply}] {msg.get('body', '')}{img_marker}"

    # Append bot response if applicable
    if actions is not None and action_idx is not None and action_idx < len(actions):
        action = actions[action_idx]
        if action.get("action") == "respond":
            line += f"\n[BOT responding to {msg['id']}] {action['text'][:2000]}"

    return line, images


def _judge_v2_chunk(llm: LLMClient, context_before: str, evaluate_text: str,
                     context_after: str,
                     images: list[tuple[bytes, str]] | None = None) -> dict:
    """Unified chunk judge — scores bot responses + finds missed + redundant.

    Two-stage approach for reliability:
    1. Try grounded (Google Search) — can verify facts but output is free-form text
    2. If parse fails, retry without grounding using response_mime_type=json

    Multimodal: images from the chunk are passed inline.
    """
    from google.genai import types as _gt

    user_content = (
        f"CONTEXT BEFORE (for reference only, do NOT evaluate):\n{context_before}\n\n"
        f"--- EVALUATE THIS SECTION ---\n{evaluate_text}\n"
        f"--- END EVALUATE ---\n\n"
        f"CONTEXT AFTER (for reference only, do NOT evaluate):\n{context_after}"
    )

    contents = [user_content]
    if images:
        for img_bytes, img_mime in images[:8]:
            contents.append(_gt.Part.from_bytes(data=img_bytes, mime_type=img_mime))

    client = llm._genai_client
    empty = {"scores": [], "missed": [], "redundant": []}

    def _parse_judge_json(text: str) -> dict | None:
        if not text or not text.strip():
            return None
        text = re.sub(r'[\x00-\x1f\x7f]', ' ', text)
        # Try stripping markdown fences first
        cleaned = re.sub(r'```json\s*', '', text)
        cleaned = re.sub(r'```\s*', '', cleaned).strip()
        if cleaned:
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                pass
        # Try finding JSON with scores key
        match = re.search(r'\{.*"scores"\s*:\s*\[.*\].*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return None

    # Stage 1: Grounded (Google Search) — best quality, free-form output
    for attempt in range(2):
        try:
            response = client.models.generate_content(
                model=JUDGE_MODEL,
                contents=contents,
                config=_gt.GenerateContentConfig(
                    system_instruction=JUDGE_V2_CHUNK_SYSTEM,
                    tools=[_gt.Tool(google_search=_gt.GoogleSearch())],
                    temperature=0.0,
                    http_options=_gt.HttpOptions(timeout=180_000),
                ),
            )
            raw = (response.text or "").strip()
            result = _parse_judge_json(raw)
            if result and result.get("scores"):
                return {
                    "scores": result.get("scores", []),
                    "missed": result.get("missed", []),
                    "redundant": result.get("redundant", []),
                }
            log.warning("V2 grounded judge attempt %d: parse failed (%d chars): %s",
                        attempt + 1, len(raw), raw[:150])
        except Exception as e:
            if "429" in str(e):
                time.sleep(2 ** attempt * 5)
            else:
                log.warning("V2 grounded judge attempt %d: %s", attempt + 1, e)

    # Stage 2: Non-grounded with guaranteed JSON output
    log.info("V2 chunk judge: falling back to non-grounded JSON mode")
    for attempt in range(2):
        try:
            response = client.models.generate_content(
                model=JUDGE_MODEL,
                contents=contents,
                config=_gt.GenerateContentConfig(
                    system_instruction=JUDGE_V2_CHUNK_SYSTEM,
                    temperature=0.0,
                    response_mime_type="application/json",
                    http_options=_gt.HttpOptions(timeout=180_000),
                ),
            )
            raw = (response.text or "").strip()
            if raw:
                result = json.loads(raw)
                return {
                    "scores": result.get("scores", []),
                    "missed": result.get("missed", []),
                    "redundant": result.get("redundant", []),
                }
        except Exception as e:
            if "429" in str(e):
                time.sleep(2 ** attempt * 5)
            else:
                log.warning("V2 JSON judge attempt %d: %s", attempt + 1, e)

    log.error("V2 chunk judge failed after all attempts")
    return empty


def run_eval_v2(system_name: str, system, judge_llm: LLMClient,
                history_msgs: list[dict], live_msgs: list[dict],
                cases_cache: str | None = None,
                dataset_dir: Path | None = None) -> dict:
    """V2 evaluation — unified chunked judge for quality + precision/recall."""

    # ── Phase 1: Ingest ──────────────────────────────────────────────────
    print("  Phase 1: Ingest")
    t0 = time.time()
    if hasattr(system, 'ingest'):
        import inspect
        if 'cases_cache' in inspect.signature(system.ingest).parameters:
            system.ingest(history_msgs, cases_cache=cases_cache)
        else:
            system.ingest(history_msgs)
    ingest_time = time.time() - t0
    print(f"    Done ({ingest_time:.1f}s)")

    # ── Phase 2: Process live messages (sequential with context injection) ─
    print(f"  Phase 2: Processing {len(live_msgs)} live messages...")
    t1 = time.time()
    actions: list[dict] = []

    if system_name in ("baseline", "baseline-conservative", "chunked-rag"):
        # Sequential: baseline sees its own prior responses (like prod)
        bot_responses: dict[int, str] = {}
        for i, msg in enumerate(live_msgs):
            # Inject bot responses into prior_live context
            prior_with_bot = []
            for j in range(i):
                prior_with_bot.append(live_msgs[j])
                if j in bot_responses:
                    prior_with_bot.append({"sender": "[BOT]", "body": bot_responses[j],
                                           "ts": live_msgs[j]["ts"], "id": f"bot_{live_msgs[j]['id']}"})
            result = system.process(msg, prior_with_bot)
            actions.append({"msg": msg, **result})
            if result.get("action") == "respond":
                bot_responses[i] = result["text"][:300]
            if result.get("action") == "respond":
                print(f"    → msg[{i}] R {(msg.get('body') or '')[:50]}")
    else:
        batch_actions = system.process_batch(live_msgs, history_msgs)
        for i, (msg, result) in enumerate(zip(live_msgs, batch_actions)):
            actions.append({"msg": msg, **result})

    process_time = time.time() - t1
    responded = [a for a in actions if a["action"] == "respond"]
    escalated = [a for a in actions if a["action"] == "escalate"]
    skipped_n = len(live_msgs) - len(responded) - len(escalated)
    print(f"    {len(responded)}R {len(escalated)}E {skipped_n}S ({process_time:.1f}s)")

    # ── Phase 3: Unified chunked judge ───────────────────────────────────
    chunk_size = JUDGE_V2_CHUNK_SIZE
    ctx_size = JUDGE_V2_CONTEXT_SIZE
    n_chunks = max(1, (len(live_msgs) + chunk_size - 1) // chunk_size)
    print(f"  Phase 3: V2 Chunked Judge ({n_chunks} chunks of ~{chunk_size} msgs)...")
    t2 = time.time()

    # Pre-build all live message lines with bot responses and images
    live_lines: list[str] = []
    live_images: list[list[tuple[bytes, str]]] = []
    for i, msg in enumerate(live_msgs):
        line, imgs = _format_msg_line(msg, actions=actions, action_idx=i,
                                       include_msg_id=True, dataset_dir=dataset_dir)
        live_lines.append(line)
        live_images.append(imgs)

    # History context lines (no msg_id, no images)
    history_lines = []
    for m in history_msgs[-50:]:
        reply = f" reply_to={m['reply_to_id']}" if m.get("reply_to_id") else ""
        history_lines.append(f"[{m['sender']} ts={m['ts']}{reply}] {m.get('body', '')}")

    # Build and submit chunks in parallel
    chunk_results: list[dict | None] = [None] * n_chunks
    chunk_ranges: list[tuple[int, int]] = []

    def _build_chunk(ci: int) -> tuple[str, str, str, list[tuple[bytes, str]]]:
        start = ci * chunk_size
        end = min(start + chunk_size, len(live_msgs))
        chunk_ranges.append((start, end))

        # Context before: history tail + previous live messages
        before_lines = list(history_lines[-ctx_size:]) if ci == 0 else []
        ctx_start = max(0, start - ctx_size)
        before_lines.extend(live_lines[ctx_start:start])
        context_before = "\n".join(before_lines[-ctx_size:]) if before_lines else "(start of conversation)"

        # Evaluate section
        evaluate_text = "\n".join(live_lines[start:end])

        # Context after: next live messages
        ctx_end = min(end + ctx_size, len(live_msgs))
        after_lines = live_lines[end:ctx_end]
        context_after = "\n".join(after_lines) if after_lines else "(end of conversation)"

        # Collect images from the evaluate section
        chunk_imgs: list[tuple[bytes, str]] = []
        for j in range(start, end):
            chunk_imgs.extend(live_images[j])

        return context_before, evaluate_text, context_after, chunk_imgs

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        chunk_futures = {}
        for ci in range(n_chunks):
            ctx_before, eval_text, ctx_after, imgs = _build_chunk(ci)
            f = pool.submit(_judge_v2_chunk, judge_llm, ctx_before, eval_text, ctx_after,
                            images=imgs if imgs else None)
            chunk_futures[f] = ci

        for f in as_completed(chunk_futures):
            ci = chunk_futures[f]
            try:
                chunk_results[ci] = f.result()
            except Exception as e:
                log.warning("V2 chunk judge %d failed: %s", ci, e)
                chunk_results[ci] = {"scores": [], "missed": [], "redundant": []}
            cr = chunk_results[ci]
            n_scores = len(cr["scores"])
            n_missed = len(cr["missed"])
            n_redun = len(cr["redundant"])
            print(f"    Chunk [{ci+1}/{n_chunks}]: {n_scores} scored, {n_missed} missed, {n_redun} redundant")

    judge_time = time.time() - t2

    # ── Merge results across chunks ──────────────────────────────────────
    live_ids = {m["id"] for m in live_msgs}
    responded_ids = {a["msg"]["id"] for a in actions if a["action"] == "respond"}
    handled_ids = responded_ids | {a["msg"]["id"] for a in actions if a["action"] == "escalate"}

    # Merge scores — match by msg_id to responded messages
    # Build numeric suffix index for robust matching (judge may return partial ids)
    _rid_by_suffix: dict[str, str] = {}
    for rid in responded_ids:
        # Extract trailing digits: "tg_ua_ardupilot_7442" → "7442"
        suffix = re.sub(r'^.*?(\d+)$', r'\1', rid)
        _rid_by_suffix[suffix] = rid
        _rid_by_suffix[rid] = rid  # exact match too

    all_scores: dict[str, dict] = {}
    _unmatched_global = []
    for ci, cr in enumerate(chunk_results):
        if not cr:
            continue
        # Collect bot response IDs in this chunk (in order) for positional fallback
        start = ci * chunk_size
        end = min(start + chunk_size, len(live_msgs))
        chunk_bot_ids = [actions[j]["msg"]["id"] for j in range(start, end)
                         if j < len(actions) and actions[j].get("action") == "respond"]

        _chunk_unmatched_scores = []
        for s in cr["scores"]:
            mid = s.get("msg_id", "")
            # Try exact match first
            if mid in responded_ids:
                all_scores[mid] = s
                continue
            # Try numeric suffix match
            suffix = re.sub(r'^.*?(\d+)$', r'\1', mid)
            if suffix in _rid_by_suffix:
                all_scores[_rid_by_suffix[suffix]] = s
                continue
            # Try substring containment
            matched = False
            for rid in responded_ids:
                if mid in rid or rid in mid:
                    all_scores[rid] = s
                    matched = True
                    break
            if not matched:
                _chunk_unmatched_scores.append(s)

        # Positional fallback: if unmatched scores remain, align by position
        # within this chunk's bot responses
        if _chunk_unmatched_scores:
            chunk_unscored = [bid for bid in chunk_bot_ids if bid not in all_scores]
            if len(cr["scores"]) == len(chunk_bot_ids):
                # Perfect count match — re-do full positional alignment for this chunk
                for pos, (bid, s) in enumerate(zip(chunk_bot_ids, cr["scores"])):
                    if bid not in all_scores:
                        all_scores[bid] = s
                        log.info("V2 judge: positional match %s → %s (chunk %d pos %d)",
                                 s.get("msg_id", "?"), bid, ci, pos)
                _chunk_unmatched_scores = []
            elif len(_chunk_unmatched_scores) == len(chunk_unscored):
                # Same number of unmatched scores and unscored responses — pair them
                for bid, s in zip(chunk_unscored, _chunk_unmatched_scores):
                    all_scores[bid] = s
                    log.info("V2 judge: positional fallback %s → %s (chunk %d)",
                             s.get("msg_id", "?"), bid, ci)
                _chunk_unmatched_scores = []

            _unmatched_global.extend(s.get("msg_id", "?") for s in _chunk_unmatched_scores)

    if _unmatched_global:
        log.warning("V2 judge: %d scores could not be matched: %s",
                    len(_unmatched_global), _unmatched_global[:5])

    # Build quality_results in responded order
    quality_results = []
    for a in responded:
        mid = a["msg"]["id"]
        s = all_scores.get(mid, {})
        scores = {k: s.get(k) for k in ("correctness", "helpfulness", "specificity", "necessity")}
        vals = [v for v in scores.values() if v is not None]
        mean_q = statistics.mean(vals) if vals else 0.0
        quality_results.append({
            "msg_id": mid,
            "msg_text": (a["msg"].get("body") or "")[:200],
            "bot_text": a["text"][:2000],
            **scores,
            "quality": round(mean_q, 2),
            "reasoning": s.get("reasoning", ""),
        })

    # Merge missed — deduplicate by msg_id, only live messages not handled
    seen_missed = set()
    all_missed = []
    for cr in chunk_results:
        if not cr:
            continue
        for m in cr["missed"]:
            mid = m.get("msg_id", "")
            if mid and mid not in seen_missed and mid in live_ids and mid not in handled_ids:
                seen_missed.add(mid)
                all_missed.append(m)

    # Merge redundant — deduplicate by msg_id, only responded messages
    redundant_ids = set()
    all_redundant = []
    for cr in chunk_results:
        if not cr:
            continue
        for r in cr["redundant"]:
            mid = r.get("msg_id", "")
            if mid in responded_ids and mid not in redundant_ids:
                redundant_ids.add(mid)
                all_redundant.append(r)

    # ── Compute metrics ──────────────────────────────────────────────────
    valid_quality = [r for r in quality_results if any(r.get(k) is not None
                     for k in ("correctness", "helpfulness", "specificity", "necessity"))]
    q_scores = [r["quality"] for r in valid_quality]
    quality = statistics.mean(q_scores) if q_scores else 0.0

    n_responded = len(responded)
    n_redundant = len(redundant_ids)
    n_useful = n_responded - n_redundant
    n_missed = len(all_missed)

    precision = n_useful / n_responded if n_responded > 0 else 1.0
    recall = n_useful / (n_useful + n_missed) if (n_useful + n_missed) > 0 else 1.0
    score = quality * recall  # headline metric: quality × recall

    for m in all_missed:
        print(f"    MISSED: {m['msg_id']} — {m.get('text', '')[:60]}")
    for r in all_redundant:
        print(f"    REDUNDANT: {r['msg_id']} — {r.get('reason', '')[:60]}")

    print(f"    Responded: {n_responded} | Redundant: {n_redundant} | Missed: {n_missed}")
    print(f"    Precision={precision:.1%} Recall={recall:.1%}")
    print(f"    Quality={quality:.2f} Score={score:.2f} ({judge_time:.1f}s)")

    total_time = ingest_time + process_time + judge_time

    return {
        "eval_version": "v2_chunked",
        "score": round(score, 2),
        "quality": round(quality, 2),
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "quality_breakdown": {
            k: round(statistics.mean([r[k] for r in valid_quality if r.get(k) is not None]), 2)
            for k in ("correctness", "helpfulness", "specificity", "necessity")
            if any(r.get(k) is not None for r in valid_quality)
        },
        "counts": {
            "history_msgs": len(history_msgs),
            "live_msgs": len(live_msgs),
            "cases_indexed": getattr(system, "num_cases", 0),
            "responded": n_responded,
            "escalated": len(escalated),
            "skipped": skipped_n,
            "redundant": n_redundant,
            "missed": n_missed,
        },
        "timing": {
            "ingest_s": round(ingest_time, 1),
            "process_s": round(process_time, 1),
            "judge_s": round(judge_time, 1),
            "total_s": round(total_time, 1),
        },
        "cost": system.cost.summary() if hasattr(system, "cost") else {},
        "responses": quality_results,
        "missed_questions": all_missed,
        "redundant_responses": all_redundant,
    }


def _generate_html_report(result: dict, history_msgs: list[dict], live_msgs: list[dict],
                           dataset_name: str, system_name: str, html_path: str) -> None:
    """Generate an HTML report with full 900/100 message history and bot responses."""
    import html as html_mod

    def esc(s):
        return html_mod.escape(str(s)) if s else ""

    responses_by_id = {}
    for r in result.get("responses", []):
        responses_by_id[r["msg_id"]] = r

    # Map live msg IDs to actions
    actions_by_id = {}
    for r in result.get("responses", []):
        actions_by_id[r["msg_id"]] = {"action": "respond", "text": r.get("bot_text", "")}
    for m in result.get("missed_questions", []):
        actions_by_id[m["msg_id"]] = {"action": "missed"}
    for r in result.get("redundant_responses", []):
        actions_by_id[r["msg_id"]] = {"action": "redundant", "reason": r.get("reason", "")}

    live_ids = {m["id"] for m in live_msgs}

    meta = result.get("meta", {})
    pretty = meta.get("pretty_name", dataset_name)
    qb = result.get("quality_breakdown", {})
    counts = result.get("counts", {})

    buf = []
    buf.append("""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SupportBench — {title}</title>
<style>
@import url('https://rsms.me/inter/inter.css');
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:'Inter',system-ui,sans-serif;background:#0d1117;padding:24px;color:#c9d1d9;font-size:13px;line-height:1.5}}
h1{{font-size:20px;margin-bottom:4px;color:#f0f6fc}}
h2{{font-size:16px;margin:24px 0 8px;color:#58a6ff}}
.meta{{color:#8b949e;font-size:12px;margin-bottom:16px}}
.summary{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:16px;margin-bottom:20px}}
.summary table{{border-collapse:collapse;width:100%}}
.summary th,.summary td{{padding:4px 10px;text-align:right;border-bottom:1px solid #21262d;font-size:12px}}
.summary th{{text-align:left;color:#8b949e;font-weight:600}}
.winner{{color:#3fb950;font-weight:700}}
.section-label{{background:#1f2937;color:#58a6ff;padding:6px 16px;font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.06em;border-bottom:1px solid #30363d}}
.msg{{padding:6px 16px;border-bottom:1px solid #1c2128;display:flex;gap:10px;font-size:12px}}
.msg:hover{{background:#1c2128}}
.msg-idx{{min-width:30px;color:#484f58;font-size:10px;text-align:right;padding-top:2px}}
.msg-sender{{min-width:80px;font-size:10px;color:#8b949e;font-weight:600;flex-shrink:0}}
.msg-body{{flex:1;word-break:break-word}}
.msg-ts{{font-size:9px;color:#484f58;margin-left:8px}}
.bot-row{{background:#0d2137;border-left:3px solid #58a6ff}}
.bot-label{{font-size:9px;font-weight:700;text-transform:uppercase;letter-spacing:.04em;padding:1px 5px;border-radius:3px;display:inline-block;margin-bottom:2px;background:#58a6ff;color:#0d1117}}
.score-badge{{font-size:10px;color:#8b949e;margin-top:2px}}
.score-badge b{{font-weight:700;color:#f0f6fc}}
.dims{{font-size:9px;color:#484f58}}
.missed-row{{background:#2d1b1b;border-left:3px solid #f85149}}
.missed-label{{font-size:9px;font-weight:700;padding:1px 5px;border-radius:3px;background:#f85149;color:#fff}}
.escalated-row{{background:#2d2400;border-left:3px solid #d29922}}
.escalated-label{{font-size:9px;font-weight:700;padding:1px 5px;border-radius:3px;background:#d29922;color:#0d1117}}
.redundant-row{{background:#1b2d2b;border-left:3px solid #3fb950}}
.filter-bar{{margin:12px 0;display:flex;gap:8px;flex-wrap:wrap}}
.filter-bar label{{font-size:11px;color:#8b949e;cursor:pointer}}
.filter-bar input{{margin-right:3px}}
.collapsed .msg-history{{display:none}}
#toggle-history{{cursor:pointer;background:#21262d;color:#58a6ff;border:1px solid #30363d;padding:4px 12px;border-radius:6px;font-size:11px;margin-bottom:12px}}
</style>
</head><body>
<h1>SupportBench — {title}</h1>
<p class="meta">{pretty} | {system} | {hist_size} history / {live_size} live | Score: {score}</p>
""".format(
        title=esc(f"{pretty} — {system_name}"),
        pretty=esc(pretty),
        system=esc(system_name),
        hist_size=counts.get("history_msgs", len(history_msgs)),
        live_size=counts.get("live_msgs", len(live_msgs)),
        score=f"{result.get('score', 0):.2f}",
    ))

    # Summary table
    buf.append('<div class="summary"><table>')
    buf.append('<tr><th>Metric</th><th>Value</th></tr>')
    buf.append(f'<tr><td style="text-align:left">Score (quality x recall)</td><td class="winner">{result.get("score", 0):.2f}</td></tr>')
    buf.append(f'<tr><td style="text-align:left">Quality</td><td>{result.get("quality", 0):.2f}</td></tr>')
    buf.append(f'<tr><td style="text-align:left">Precision</td><td>{result.get("precision", 0):.1%}</td></tr>')
    buf.append(f'<tr><td style="text-align:left">Recall</td><td>{result.get("recall", 0):.1%}</td></tr>')
    for k, v in qb.items():
        buf.append(f'<tr><td style="text-align:left">{esc(k)}</td><td>{v:.2f}</td></tr>')
    buf.append(f'<tr><td style="text-align:left">Responded / Escalated / Skipped</td><td>{counts.get("responded",0)} / {counts.get("escalated",0)} / {counts.get("skipped",0)}</td></tr>')
    buf.append(f'<tr><td style="text-align:left">Cases indexed</td><td>{counts.get("cases_indexed",0)}</td></tr>')
    if "missed" in counts:
        buf.append(f'<tr><td style="text-align:left">Missed / Redundant</td><td>{counts.get("missed",0)} / {counts.get("redundant",0)}</td></tr>')
    cost = result.get("cost", {})
    if cost:
        buf.append(f'<tr><td style="text-align:left">Cost</td><td>${cost.get("cost_usd", 0):.4f}</td></tr>')
    buf.append('</table></div>')

    # Toggle button for history
    buf.append('<button id="toggle-history" onclick="document.getElementById(\'messages\').classList.toggle(\'collapsed\')">Toggle History (900 msgs)</button>')

    buf.append('<div id="messages">')

    # History section
    buf.append('<div class="section-label">History (ingested for case extraction)</div>')
    buf.append('<div class="msg-history">')
    for idx, m in enumerate(history_msgs):
        sender = esc(m.get("sender", "?")[:10])
        body = esc(m.get("body", ""))
        ts = m.get("ts", "")
        buf.append(f'<div class="msg"><div class="msg-idx">{idx}</div><div class="msg-sender">{sender}</div><div class="msg-body">{body}</div><div class="msg-ts">{ts}</div></div>')
    buf.append('</div>')

    # Live section
    buf.append('<div class="section-label">Live (evaluated)</div>')
    for idx, m in enumerate(live_msgs):
        mid = m["id"]
        sender = esc(m.get("sender", "?")[:10])
        body = esc(m.get("body", ""))
        ts = m.get("ts", "")
        reply = f' reply_to={esc(m["reply_to_id"])}' if m.get("reply_to_id") else ""

        # Check if this message has a bot response
        resp = responses_by_id.get(mid)
        missed_q = any(mq["msg_id"] == mid for mq in result.get("missed_questions", []))

        extra_cls = ""
        if missed_q:
            extra_cls = " missed-row"

        buf.append(f'<div class="msg{extra_cls}"><div class="msg-idx">{len(history_msgs)+idx}</div><div class="msg-sender">{sender}{reply}</div><div class="msg-body">{body}</div><div class="msg-ts">{ts}</div></div>')

        if resp:
            bot_text = esc(resp.get("bot_text", ""))
            q = resp.get("quality", 0)
            corr = resp.get("correctness", "-")
            hlp = resp.get("helpfulness", "-")
            spec = resp.get("specificity", "-")
            nece = resp.get("necessity", "-")
            reasoning = esc(resp.get("reasoning", ""))
            buf.append(f'<div class="msg bot-row"><div class="msg-idx"></div><div class="msg-sender"><span class="bot-label">BOT</span></div><div class="msg-body">{bot_text}<div class="score-badge">Quality: <b>{q:.1f}</b></div><div class="dims">corr={corr} | help={hlp} | spec={spec} | nece={nece}</div><div class="dims">{reasoning}</div></div></div>')

        if missed_q:
            buf.append(f'<div class="msg missed-row"><div class="msg-idx"></div><div class="msg-sender"><span class="missed-label">MISSED</span></div><div class="msg-body">Bot did not respond to this question</div></div>')

    buf.append('</div>')  # messages

    buf.append('</body></html>')

    Path(html_path).parent.mkdir(parents=True, exist_ok=True)
    with open(html_path, "w") as f:
        f.write("\n".join(buf))
    print(f"  HTML report: {html_path}")


def main():
    parser = argparse.ArgumentParser(description="SupportBench — Score, Coverage, Cost")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", type=int, default=100, help="Live messages count")
    parser.add_argument("--history", type=int, default=None, help="History messages (default: 9x split)")
    parser.add_argument("--system", choices=["supportbot", "baseline", "baseline-conservative", "chunked-rag"], default="supportbot")
    parser.add_argument("--output", default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--cases-cache", default=None, help="Cache extracted cases to/from this JSON file")
    parser.add_argument("--mode", choices=["v1", "v2"], default="v2",
                        help="v1=legacy separate coverage judge, v2=unified chunked judge (default)")
    parser.add_argument("--html", default=None, help="Generate HTML comparison report at this path")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    for lib in ("httpx", "chromadb", "openai", "httpcore", "google"):
        logging.getLogger(lib).setLevel(logging.WARNING)

    if "GOOGLE_API_KEY" not in os.environ:
        print("ERROR: GOOGLE_API_KEY required", file=sys.stderr)
        sys.exit(1)

    history_size = args.history if args.history is not None else args.split * 9

    print(f"SupportBench | {args.dataset} | {args.system} | {history_size}/{args.split}")

    data = _load_dataset(args.dataset)
    meta = data.get("meta", {})
    group_id = meta.get("name", args.dataset)
    text_msgs = [m for m in data["messages"] if m.get("body")]

    total_needed = history_size + args.split
    if args.offset + total_needed > len(text_msgs):
        print(f"ERROR: need {total_needed} msgs, only {len(text_msgs)} available", file=sys.stderr)
        sys.exit(1)

    history_msgs = text_msgs[args.offset:args.offset + history_size]
    live_msgs = text_msgs[args.offset + history_size:args.offset + total_needed]

    settings = _make_settings()
    judge_llm = LLMClient(settings)
    group_description = meta.get("description", "")
    # Resolve dataset directory for media loading
    dataset_dir = DATASETS_DIR / args.dataset
    if not dataset_dir.exists():
        dataset_dir = DATASETS_DIR  # fallback: flat layout
    dataset_lang = meta.get("lang", "en")
    docs_urls = meta.get("docs_urls", [])
    if args.system == "supportbot":
        system = SupportBotSystem(settings, group_id, group_description=group_description,
                                  dataset_dir=dataset_dir, lang=dataset_lang,
                                  docs_urls=docs_urls)
    elif args.system == "baseline-conservative":
        system = BaselineSystem(settings, dataset_dir=dataset_dir, variant="conservative")
    elif args.system == "chunked-rag":
        system = ChunkedRAGSystem(settings, dataset_dir=dataset_dir)
    else:  # "baseline" = aggressive (default)
        system = BaselineSystem(settings, dataset_dir=dataset_dir, variant="aggressive")

    cases_cache = args.cases_cache
    if cases_cache is None and args.system == "supportbot":
        cases_cache = f"results/cases_{args.dataset}.json"

    if args.mode == "v2":
        print(f"  Mode: V2 (accumulated unified judge)")
        result = run_eval_v2(args.system, system, judge_llm, history_msgs, live_msgs,
                             cases_cache=cases_cache, dataset_dir=dataset_dir)
    else:
        coverage_cache = f"results/coverage_{args.dataset}.json"
        result = run_eval(args.system, system, judge_llm, history_msgs, live_msgs,
                          cases_cache=cases_cache, coverage_cache=coverage_cache,
                          dataset_dir=dataset_dir)

    output = {
        "benchmark": "SupportBench",
        "version": "2.0" if args.mode == "v2" else "1.0",
        "system": args.system, "dataset": args.dataset,
        "meta": {
            "pretty_name": meta.get("pretty_name", args.dataset),
            "lang": meta.get("lang", "?"), "domain": meta.get("domain", "?"),
            "history_size": history_size, "live_size": args.split, "offset": args.offset,
        },
        **result,
    }

    out_path = args.output or f"results/eval_{args.dataset}_{args.system}.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    c = result["counts"]
    qb = result.get("quality_breakdown", {})
    print(f"\n{'='*60}")
    print(f"  Score:     {result['score']:.1f} / 10  (quality × recall)")
    print(f"  Quality:   {result['quality']:.1f} / 10  (per-response avg)")
    print(f"  Precision: {result['precision']:.1%}")
    print(f"  Recall:    {result['recall']:.1%}")
    print(f"  Breakdown: " + " | ".join(f"{k}={v:.1f}" for k, v in qb.items()))
    if args.mode == "v2":
        print(f"  Redundant: {c['redundant']} | Missed: {c['missed']}")
    else:
        print(f"  Questions: {c['questions_found']} | TP={c['true_positives']} FP={c['false_positives']} Missed={c['missed']}")
    print(f"  Actions: {c['responded']}R {c['escalated']}E {c['skipped']}S | Cases: {c['cases_indexed']}")
    cost_info = result.get("cost", {})
    cost_usd = cost_info.get("cost_usd", 0)
    total_tok = cost_info.get("total_tokens", 0)
    print(f"  Cost:    ${cost_usd:.4f} ({total_tok:,} tokens)")
    print(f"  Time: {result['timing']['total_s']}s | Output: {out_path}")
    print(f"{'='*60}\n")

    # HTML report generation
    html_path = args.html
    if html_path is None:
        html_path = out_path.replace(".json", ".html")
    _generate_html_report(output, history_msgs, live_msgs, args.dataset, args.system, html_path)


if __name__ == "__main__":
    main()
