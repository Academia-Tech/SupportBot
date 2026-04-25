# 1-1 DM Links — Implementation Reference

**Status**: Production (Telegram, Discord, Signal, WhatsApp, Slack) · Last updated 2026-04-16

The 1-1 DM Links feature lets a support group publish per-platform deep links so end users can DM the bot directly — instead of writing in the group chat — and still get answers from that group's knowledge base. When the bot's confidence is low, the conversation escalates to a live admin who replies from their usual messenger; the admin's answer is relayed to the user, saved to a DM-only Q&A cache, and reused for future similar questions.

---

## 1. User flow

From the end user's side:

1. Admin generates a link for a platform on the dashboard (Share button → select platform).
2. User taps the link, which opens a DM thread with the bot on that platform (or shows a copy-paste token if deep-linking isn't possible on that messenger).
3. User asks a question. The bot answers from the group's KB with RAG + citations, identical quality to the group chat.
4. If the bot's confidence is low, it silently opens a ticket, DMs an admin for help, and tells the user it's "checking with the team."
5. Admin replies in their own DM thread. The bot relays the answer to the user and stores `(question, answer)` in a DM-only cache so the next user with the same question gets it instantly.

From the admin's side: toggle DMs on/off for any group via the Share button. Business plan only.

---

## 2. Enabling and gating

- **Share button** lives in `signal-web/pages/dashboard/index.tsx`. Opens a modal with per-platform link generation + copy / QR code.
- **Plan gate**: all link-generation and enable/disable endpoints require an active Business plan (`plan='unlimited' AND status IN ('active','trialing')`) — checked in `signal-bot/app/web_api.py:2993`. Free plans return 402.
- **Per-group kill switch**: `chat_groups.dm_enabled` (TINYINT, default 0). When off, incoming DMs for that group are ignored silently. Controlled by `GET/POST /groups/dm-enabled`.
- The Share button on a group card turns green (green border, white background) when `dm_enabled=1` so the state is visible without opening the modal.

---

## 3. Link generation

`POST /groups/dm-link` with `{group_id, platform}` → generates a 32-char hex token (uuid4), stores `(token, group_id, platform, created_by, created_at)` in `dm_links`, and returns the shareable URL plus a base64-encoded QR PNG.

Link creation is **idempotent per (group, platform)**: calling it twice returns the same token unless the group is explicitly reset.

URL formats per platform (built by `_build_dm_url` in `dm_responder.py`):

| Platform | URL form | Behaviour on click |
|---|---|---|
| Telegram | `https://t.me/<BotHandle>?start=<token>` | Opens DM with Telegram's native `/start` button pre-filled. |
| WhatsApp | `https://wa.me/<bot-number>?text=<token>` | Opens WhatsApp thread with the token prefilled as the first message. |
| Discord / Slack / Signal | `https://<domain>/join/<token>` | Next.js landing page: either 302s to a deep link where possible, or shows a copy-paste instruction with the token. |

Token extraction on the bot side uses a `_TOKEN_RE` regex (`dm_responder.py:25`) that matches the token anywhere in the user's first DM, so copy-paste works even on platforms without structured deep-linking.

---

## 4. DM response pipeline

Entry point: `handle_dm()` in `signal-bot/app/jobs/dm_responder.py:501`. Runs on a background daemon thread so the platform listener isn't blocked.

### User-to-group mapping

Three-priority lookup in `_find_groups_for_sender`:

1. **Explicit DM session** (`dm_sessions` row) — created on first link claim or auto-discovered when a known user DMs the bot. Highest precedence.
2. **Platform membership** — the sender's identity is found in `raw_messages` on that platform, so they're plausibly a member of groups the bot also sees. Those groups are ranked by embedding similarity of the question to group descriptions + top cases.
3. **Populated-groups fallback** — any group on the same platform with cases; then any populated group globally (cross-platform fallback).

Multiple candidate groups → `_pick_top_groups` selects top-K by embedding similarity and `answer_multigroup()` runs a multi-group synthesis.

### Answer generation

- Single group → `ultimate_agent.answer()` — same SCRAG + RCRAG + docs pipeline as the group chat bot, same citation format.
- Multi-group → `answer_multigroup()` with weighted top-K selection.
- **DM-only cache**: before invoking the synthesizer, `_lookup_dm_qa` queries `dm_qa` for previous admin-answered questions on this group with embedding similarity ≥ 0.78. Cache hits skip the synthesizer entirely. Language mismatch bypasses the cache so the answer is regenerated in the right language.
- **Group language override**: if `chat_groups.language` is set, the bot answers in that language regardless of the user's detected language (keeps brand tone consistent).
- Internal placeholders like `[[TAG_ADMIN]]` are stripped before sending.

### Low-confidence escalation

If every sub-agent returns empty, the bot:

1. Opens a `pending_tickets` row: `(user_id, platform, group_id, question, lang, status='open')`.
2. Picks the first admin in the group with an available DM channel, DMs them the question + user-mention + link to the conversation history.
3. Tells the user "checking with the team" in their language.
4. Admin replies in their DM:
   - Plain text → the reply is relayed to the user, cached in `dm_qa` keyed by embedding of the question, ticket marked `answered`.
   - `/skip` → ticket reassigned to the next admin in `tried_admins`; if none left, user is told "we'll get back to you."

### Cross-platform admin reply

If the admins are on a different platform than the user (user on Telegram, only admin on Signal), the relay is prefixed "From the team:" and the admin-mention falls back to a plain name (no structured mention).

---

## 5. Per-platform handlers

All platform adapters funnel into the same chain: `try_claim_link()` → `try_handle_admin_reply()` → `handle_dm()`. Platform-specific quirks:

| Platform | Entry | Token form | Mention format |
|---|---|---|---|
| Telegram | `_handle_telegram_direct_message` (`main.py:993`) | `/start <token>` parsed by `extract_start_token()` | `[Name](tg://user?id=<id>)` |
| Discord | `_handle_discord_direct_message` (`main.py:764`) | `/start TOKEN` or plain paste | `<@snowflake_id>` |
| Signal | Signal listener | Plain paste | No structured mention — falls back to name |
| WhatsApp | WhatsApp bridge handler (`main.py:397`) | `wa.me` prefill | `@<phone_digits>` |
| Slack | Slack Events API | Plain paste | `<@U_ID>` |

Telegram additionally supports admin language commands (`/en`, `/ua`) inside the DM thread.

**WhatsApp caveat**: WhatsApp's platform policy only lets a bot initiate a DM after the user has messaged the bot first. The `wa.me` link is the work-around — clicking it opens the thread with the token prefilled, which counts as user-initiated.

---

## 6. Data model

All tables defined in `signal-bot/app/db/schema_mysql.py:229-309`.

```sql
-- Shareable link per (group, platform)
CREATE TABLE dm_links (
  token      VARCHAR(64) PRIMARY KEY,
  group_id   VARCHAR(64) NOT NULL,
  platform   VARCHAR(32) NOT NULL,
  created_by VARCHAR(128),
  created_at DATETIME    NOT NULL DEFAULT CURRENT_TIMESTAMP,
  INDEX (group_id, platform)
);

-- User ↔ group binding; one row per (user, platform, group)
CREATE TABLE dm_sessions (
  user_id    VARCHAR(128),
  platform   VARCHAR(32),
  group_id   VARCHAR(64),
  created_at DATETIME    NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (user_id, platform, group_id)
);

-- Escalation tickets
CREATE TABLE pending_tickets (
  id                  BIGINT PRIMARY KEY AUTO_INCREMENT,
  user_id             VARCHAR(128) NOT NULL,
  platform            VARCHAR(32)  NOT NULL,
  group_id            VARCHAR(64)  NOT NULL,
  question            TEXT,
  lang                VARCHAR(8),
  status              ENUM('open','answered','skipped') NOT NULL,
  assignee_admin      VARCHAR(128),
  assignee_platform   VARCHAR(32),
  tried_admins        TEXT,                       -- CSV of admin_ids already attempted
  answer_text         LONGTEXT,
  history_token       VARCHAR(64),                -- link to conversation history page
  user_display_name   VARCHAR(128),
  user_handle         VARCHAR(128),
  created_at          DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- DM-only fast-path cache; separate from the main cases table
CREATE TABLE dm_qa (
  id             BIGINT PRIMARY KEY AUTO_INCREMENT,
  group_id       VARCHAR(64) NOT NULL,
  question       TEXT        NOT NULL,
  answer         LONGTEXT    NOT NULL,
  lang           VARCHAR(8),
  status         ENUM('solved','recommendation') NOT NULL,
  embedding_json LONGTEXT,                         -- JSON array of floats
  admin_id       VARCHAR(128),                     -- admin who answered
  ticket_id      BIGINT,                           -- originating ticket
  created_at     DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  INDEX (group_id, lang)
);

-- Append-only conversation log per user + platform
CREATE TABLE dm_messages (
  id                BIGINT PRIMARY KEY AUTO_INCREMENT,
  user_id           VARCHAR(128) NOT NULL,
  platform          VARCHAR(32)  NOT NULL,
  direction         ENUM('in','out') NOT NULL,
  text              LONGTEXT,
  image_paths_json  LONGTEXT,
  sender_name       VARCHAR(128),
  sender_handle     VARCHAR(128),
  created_at        DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  INDEX (user_id, platform, created_at)
);

-- Per-group DM kill switch (column on existing table)
ALTER TABLE chat_groups ADD COLUMN dm_enabled TINYINT(1) NOT NULL DEFAULT 0;
```

---

## 7. API endpoints

All under `/api/web/` and handled in `signal-bot/app/web_api.py:2979-3084`. Frontend wrappers in `signal-web/lib/api.ts`.

| Endpoint | Method | Purpose | Response |
|---|---|---|---|
| `/groups/dm-link` | POST | Generate (or fetch existing) link for `(group_id, platform)` | `{token, url, qr_png_base64}` |
| `/groups/dm-links` | GET | List all links for a group | `[{token, platform, created_by, created_at, url}]` |
| `/groups/dm-enabled` | GET | Fetch the per-group DM switch | `{group_id, enabled}` |
| `/groups/dm-enabled` | POST | Flip the DM switch | `{ok, group_id, enabled}` |

All four endpoints require Business plan. `GET /groups/dm-enabled` is the only one callable without Business (so the UI can show the current state to free users before upsell).

---

## 8. Known limitations

1. **No multi-turn admin threads per ticket** — each ticket accepts one admin answer. Re-opening a topic requires a new escalation cycle.
2. **`dm_qa` has no pruning** — answers accumulate indefinitely. Low risk today, but will want a background trim job once a group has thousands of cached Q&As.
3. **No per-group rate limit** on DMs — relies on upstream platform rate limits and the Business plan gate. A single group could in theory drive high token spend through DMs.
4. **WhatsApp 24-hour window** — WhatsApp's policy window means if a user hasn't messaged the bot in 24 h, admin replies may be delivered late or require a template message. Not currently handled in code.
5. **Cross-platform mentions degrade** — when the admin and user are on different platforms, the bot uses a plain "From the team:" prefix rather than a clickable mention.
6. **`/join/<token>` landing page** is a Next.js route that assumes the token is valid. Invalid/expired tokens currently show a generic copy-paste screen rather than a clear error.
7. **No link revocation UI** — link-level revocation happens only by deleting the row directly; the dashboard regenerates tokens via the same idempotent endpoint, so toggling `dm_enabled` is the user-facing kill switch.
