# Phone Bot — Implementation Reference

**Status**: Production (US + CA instant; other countries bundle-gated) · Last updated 2026-04-16

The Phone Bot lets a support group attach a real phone number. Customers call it, a voice AI answers from the group's knowledge base, and when the AI can't solve the issue it sequentially rings the group's admins and warm-transfers the caller.

Stack: **Telnyx** (PSTN + Call Control + Media Streams) · **Gemini 3.1 Flash Live** (voice AI, with 2.5 Flash Live and 2.0 Flash Live as runtime fallbacks) · signal-bot FastAPI service.

For the original motivation, pricing analysis, and phased rollout plan see [PHONE_BOT_PLAN.md](./PHONE_BOT_PLAN.md). This document covers what is actually implemented today.

---

## 1. End-to-end inbound call flow

```
Caller ──PSTN──► Telnyx ──webhook──► signal-bot /api/web/phone/webhook
                   │
                   └──Media Stream WS──► signal-bot /api/web/phone/media-stream/{token}
                                            │
                                            ▼
                                      Gemini 3.1 Flash Live
                                      (realtimeInput.audio ↔ audio)
                                            │ toolCall
                                            ▼
                                      answer_from_kb / escalate_to_human
```

Event-by-event (handlers in `signal-bot/app/phone/routes.py`):

| Telnyx event | Handler | What it does |
|---|---|---|
| `call.initiated` | `_handle_call_initiated` (routes.py:174) | Looks up group by dialed DID, enforces the 500 min/mo usage cap, answers the call, records row in `phone_calls`. |
| `call.answered` | `_handle_call_answered` (routes.py:242) | Starts call recording, begins preparing the Gemini session in parallel, then calls `telnyx.start_media_stream` to open the bidirectional WebSocket. |
| `streaming.started` | `_handle_streaming_started` (routes.py:365) | Creates the Gemini Live session, injects `group_id` into tool args, patches `answer_from_kb` with start/end chimes, attaches audio + interruption handlers. |
| `media` frames | Media-stream WebSocket (routes.py:587) | μ-law 8 kHz frames from the caller → Gemini. PCM16 24 kHz from Gemini → μ-law 8 kHz → caller. Barge-in fires a Telnyx `clear` event to flush buffered audio. |
| `call.hangup` | `_handle_call_hangup` (routes.py:341) | Closes the Gemini session, records elapsed seconds in `phone_usage`, finalizes `phone_calls` row. |
| `call.recording.saved` | `_handle_recording_saved` (routes.py:559) | Stores recording URL and spawns a background transcription + case-extraction job (`recording.py`). |
| `call.machine.detection.ended` | escalation AMD flow (see §2) | Drives the admin dial loop. |

### Audio pipeline

All conversions happen in `gemini_live.py`:

- **Caller → Gemini**: μ-law 8 kHz → PCM16 → resample 8 k → 16 k (`audioop.ratecv`).
- **Gemini → Caller**: PCM16 24 kHz → resample 24 k → 8 k → μ-law.
- **Bidirectional stream**: `stream_bidirectional_mode=rtp`, `stream_bidirectional_codec=PCMU` (`telnyx_client.py:198`). Required — without it, anything we write to the WS is dropped and the caller hears silence.
- **VAD** (`gemini_live.py:282`, tuned for phones 2026-04-16):
  - `startOfSpeechSensitivity: HIGH` — still start listening fast.
  - `endOfSpeechSensitivity: LOW` — don't close the turn on a mid-sentence pause, the caller may be thinking.
  - `prefixPaddingMs: 250` — keep the first quarter-second of audio (was 50 ms, which chopped opening consonants on narrowband μ-law).
  - `silenceDurationMs: 800` — wait nearly a second of silence before declaring end-of-turn, so unclear speech with hesitations is heard as one utterance instead of fragments.
- **ASR language pin** (`gemini_live.py`, `speechConfig.languageCode`): set to the group's configured language via `_lang_to_bcp47(config.language)`. Without this, Gemini Live auto-detects per turn and frequently mis-labels unclear / accented speech — especially on narrowband audio. The rare multilingual caller is a worthwhile trade for dramatically better recognition on unclear in-language speech.
- **Barge-in**: Gemini emits an `interrupted` event when the caller speaks over the bot; routes.py forwards a `{"event":"clear"}` message to Telnyx so buffered TTS is discarded immediately.
- **Tool chimes**: `_TOOL_START_ULAW` (rising 784 Hz → 1175 Hz "bip") plays before a KB lookup; `_TOOL_END_ULAW` (falling 1175 Hz → 784 Hz "bop") plays when the result returns (`routes.py:53-79`).
- **Non-blocking tool dispatch**: tool calls are fired with `asyncio.create_task` so the recv loop keeps pumping audio frames and `interrupted` events while RAG runs (`gemini_live.py:475`). Earlier versions awaited the tool inline and went silent for the full 2 s RAG budget.

### Model cascade

`gemini_live.py:24` tries in order: `gemini-3.1-flash-live-preview` → `gemini-2.5-flash-live` → `gemini-2.0-flash-live`. Each model swaps in without code changes.

---

## 2. Escalation flow

Triggered when Gemini calls the `escalate_to_human(reason)` tool. FSM lives in `signal-bot/app/phone/escalation.py`.

1. Tool handler returns `{"action": "hangup-and-escalate"}` (`tools.py:223`).
2. Routes.py waits 3 s for the bot to finish its handoff line, then closes the Gemini session while keeping the caller leg alive (`routes.py:481-488`).
3. `escalation.start_escalation()` loads admin contacts from `phone_escalations` (ordered by `position`) and dials the first one via Telnyx Call Control with `answering_machine_detection=premium` and a 25 s ring timeout (`escalation.py:24`, `telnyx_client.py:150`).
4. On each outbound leg:
   - Premium AMD detects `human` → `telnyx.bridge(caller_leg, admin_leg)`; `phone_calls.bridged_admin_idx` is set; the bot detaches and watches only for hangup.
   - AMD detects `machine`/`silence`, OR the leg hangs up with `no_answer`/`busy`/`rejected` → hang up the admin leg, advance `admin_idx`, dial the next admin.
5. If every admin is exhausted, `_handle_escalation_exhausted` (`routes.py:516`) plays a "couldn't reach anyone, we'll call you back" TTS line, hangs up the caller, and posts a missed-call card to the admin group chat via `writeback.post_missed_call()`.

> **Outbound dial prerequisite**: the Telnyx account needs an **Outbound Voice Profile** attached to the Call Control Application, with the destination country whitelisted. Without it, outbound dials return 403 and escalation silently fails. See §8 for the attached profile ID.

---

## 3. Number provisioning (assign / delete / pool)

Code: `signal-bot/app/phone/routes.py:935-1514`. Two counters govern behaviour:

| Counter | What it counts | Used for |
|---|---|---|
| `_count_phone_bots()` | Rows with `status='active'` only | UI "N of 5 used" counter |
| `_count_all_rentals()` | Rows with `status IN ('active','pooled')`, excluding `pending-%` stubs | Hard cap enforcement (max 5 per account) |

### Activate flow (`POST /phone/assign`)

1. If the group already has an active number → return it (idempotent).
2. Same-group pool hit → reactivate it.
3. Account-wide pool hit → re-home the oldest pooled number to this group (`acquired_at` is preserved so the 27-day release window continues).
4. Otherwise enforce `_count_all_rentals() < 5`, search Telnyx for an available DID in the chosen country, order it, persist to `phone_bots`.

### Language preservation

`phone_bots.language` is written with the precedence chain **request → prior row → group-detected → 'en'** (`routes.py:_save_phone_bot`). Setting a language before a number exists writes a stub row `telnyx_number='pending-<group_id>'`; on the next order that row is updated in place. Reactivating a pooled number from another group also carries the new group's language through, so switching UI language never resets the flag to British English.

### Soft-delete and pool sweeper

- `POST /phone/delete` sets `status='pooled'` instead of releasing the number (keeps it reusable cross-group).
- `pending-%` language stubs are deleted outright — they never held a real DID.
- `_pool_sweeper_loop()` runs hourly (`routes.py:139`) and calls `telnyx.release_number()` on any pooled row older than 27 days. That's inside the first monthly billing cycle, so no pooled number ever rolls into a second month of rental fees.
- Reactivation preserves `acquired_at`, so a number reassigned 20 days after purchase still releases at the 27-day mark of the *original* purchase.

### Hard cap behavior

The 5-rental cap counts both active and pooled. If a user hits it with pooled slots available, they must delete a pooled number first (or wait for the sweeper). The active counter shown in the UI only includes `status='active'`.

---

## 4. Tool calls exposed to Gemini

Declared in `signal-bot/app/phone/tools.py`.

### `answer_from_kb(question: str, language?: "en"|"uk")`

Searches the group's knowledge base. Uses the same dual-RAG stack as the chat bot:

- Runs case search (SCRAG at threshold 0.65, relaxing to 0.85 if empty) in parallel with docs search.
- Timeout budget: 2 s for cases, 0.6 s grace for docs if cases hit, 2.5 s total if cases miss.
- Strips URLs and `[[…]]` placeholders before returning; caps at 2000 chars.
- Returns `"I don't have that information."` if both time out.

Wrapped by routes.py so a bip sound plays before the call and a bop sound plays when the result returns — the caller hears "let me check that for you *bip*…*bop* yes, the answer is…" instead of silent dead air.

### `escalate_to_human(reason: str)`

Sync handler returns `{"action": "hangup-and-escalate"}`. Routes.py detects this action and kicks off the escalation FSM described in §2.

Tool responses are delivered via Gemini's `toolResponse.functionResponses` (`gemini_live.py:369`).

### System prompt

Built per-call by `_build_system_prompt` (`routes.py:1633`). Includes group name, group description, escalation contacts (by name — phone numbers never leak into the prompt), the language hint, a privacy clause (never read URLs / tokens / phone numbers aloud), a "never speak tool/capability names out loud" rule, and an **"UNCLEAR SPEECH"** rule added 2026-04-16 that instructs the model to politely ask the caller to repeat — once, in one short sentence — rather than guess or go silent. In tandem with the VAD and language-pin changes this fixed most "bot didn't hear me" complaints.

---

## 5. Dashboard API (`/api/web/phone/...`)

All endpoints require `group_id` query param unless noted. Endpoint handlers in `routes.py`.

| Endpoint | Method | Purpose |
|---|---|---|
| `/search` | POST | Preview available DIDs for a country |
| `/activate` | POST | Order a previously previewed number |
| `/assign` | POST | One-step pool-check → search → order (used by the dashboard) |
| `/usage` | GET | `{month, seconds, limit_seconds}` — limit is 30 000 s (500 min) |
| `/countries` | GET | Country list with status (`instant` / `bundle` / `disabled`) |
| `/status` | GET | Full group phone config: number, country, language, usage, escalations |
| `/escalations` | GET | Ordered admin list |
| `/escalations/add` | POST | Append a `{name, phone}` |
| `/escalations/remove` | POST | Remove by `position` |
| `/limits` | GET | `{used, max: 5}` — active numbers only |
| `/active-groups` | GET | `{group_ids: [...]}` for sidebar phone icons |
| `/language` | POST | Set bot language (`en` / `uk` / …) |
| `/delete` | POST | Soft-delete (move to pool) |
| `/debug-rag` | GET | Diagnostic — shows the cases `answer_from_kb` would retrieve |

Frontend wrappers in `signal-web/lib/api.ts` under the `phone*` methods.

---

## 6. Database schema

All tables defined in `signal-bot/app/db/schema_mysql.py`.

```sql
-- One row per group with a phone number (active, pooled, or pending-stub)
CREATE TABLE phone_bots (
  group_id      VARCHAR(64) PRIMARY KEY,
  telnyx_number VARCHAR(32) NOT NULL UNIQUE,   -- 'pending-<group_id>' for language stubs
  country_code  CHAR(2)     NOT NULL,
  voice_id      VARCHAR(64) NOT NULL DEFAULT 'Aoede',
  language      VARCHAR(8)  NOT NULL DEFAULT 'en',
  greeting      TEXT,
  record_calls  TINYINT(1)  NOT NULL DEFAULT 1,
  status        ENUM('active','pooled','pending') NOT NULL DEFAULT 'active',
  acquired_at   DATETIME,
  created_at    DATETIME    NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE phone_escalations (
  id          BIGINT PRIMARY KEY AUTO_INCREMENT,
  group_id    VARCHAR(64)  NOT NULL,
  position    INT          NOT NULL,             -- 0 = first to ring
  name        VARCHAR(128) NOT NULL,
  phone_e164  VARCHAR(32)  NOT NULL,
  UNIQUE KEY (group_id, position),
  FOREIGN KEY (group_id) REFERENCES phone_bots(group_id) ON DELETE CASCADE
);

CREATE TABLE phone_calls (
  id                BIGINT PRIMARY KEY AUTO_INCREMENT,
  telnyx_call_id    VARCHAR(128) UNIQUE,
  group_id          VARCHAR(64)  NOT NULL,
  caller_e164       VARCHAR(32)  NOT NULL,
  started_at        DATETIME     NOT NULL,
  ended_at          DATETIME,
  ai_seconds        INT          DEFAULT 0,
  bridged_admin_idx INT,                          -- null if never bridged
  ticket_id         VARCHAR(64),                  -- set if a missed-call ticket was opened
  recording_url     VARCHAR(512),
  transcript        LONGTEXT,
  status            ENUM('in_progress','resolved_by_ai','bridged','missed','failed') NOT NULL
);

CREATE TABLE phone_usage (
  group_id VARCHAR(64),
  month    CHAR(7),                               -- YYYY-MM
  seconds  INT NOT NULL DEFAULT 0,
  PRIMARY KEY (group_id, month)
);
```

---

## 7. Country configuration

Source: `signal-bot/app/phone/country_config.py`.

| Tier | Countries | Note |
|---|---|---|
| Instant | US, CA | Orderable immediately. |
| Bundle-gated (disabled in code today) | UA, GB, DE, FR, ES, IT, NL, PL, AU, BR | Needs a Telnyx Regulatory Bundle (one-time, free, 24-72 h review). Once the bundle is approved, flip status `'disabled'` → `'instant'` in `country_config.py` and restart the container — no other code change. |

The dashboard country picker reads `/phone/countries` on mount, so a config flip propagates without a frontend redeploy.

---

## 8. Environment variables

| Var | Required | Purpose |
|---|---|---|
| `TELNYX_API_KEY` | yes | Telnyx API bearer token (`telnyx_client.py:16`). |
| `TELNYX_APP_ID` | no | Pre-created Call Control Application ID. If unset, the service auto-discovers or creates one on startup keyed by webhook URL (`telnyx_client.py:45`). |
| `PHONE_WEBHOOK_BASE` | no | Override the webhook base URL. Default: `api.<apex-domain-of-public_url>` (`routes.py:111`). |
| `GEMINI_KEY` *or* `GOOGLE_API_KEY` | yes | Google Gemini API key used by the Live API (`gemini_live.py:148`). |

**Out-of-band Telnyx setup** (one-time, via Telnyx dashboard or API, not in code):

- An **Outbound Voice Profile** with the destination countries whitelisted (currently US, UA, GB, CA, AU, DE, FR, PL) and a daily spend cap, attached to the Call Control Application. Without this, outbound escalation dials return 403 even though inbound works.
- Call Control Application webhook URL = `https://<PHONE_WEBHOOK_BASE>/api/web/phone/webhook`.

---

## 9. Usage, limits, and billing

- **Free tier**: 500 minutes per group per month (`routes.py:10`, `usage.py:10`). Seconds are accumulated in `phone_usage` on each `call.hangup`.
- **Enforcement**: if a group exceeds its monthly cap, `call.initiated` refuses the call with a TTS message instead of queueing.
- **Rental cap**: 5 numbers per account (active + pooled combined), `MAX_NUMBERS_PER_ACCOUNT = 5` (`routes.py:935`).
- **Concurrency**: no explicit per-group concurrency limit today. Telnyx account-level concurrency applies.
- **Recording → case generation**: `record_calls=1` by default. After hangup, recordings are transcribed in the background and run through the existing `cases_from_transcript` pipeline so the next caller with the same question gets it from RAG.

---

## 10. Known limitations and sharp edges

1. **Initial greeting gap (~2-3 s)** — Gemini session prep now runs in parallel with media-stream setup, down from the 5 s silence seen earlier. Further reduction would require warming Gemini sessions before `call.answered`.
2. **ASR language drift** — Google's ASR occasionally mis-labels the caller's language on the first turn. Mitigated as of 2026-04-16 by pinning `speechConfig.languageCode` to the group's language (see §1 *ASR language pin*). Routes.py still prefers tool-call question text over ASR transcript bubbles when persisting turns (`routes.py:435`) as a belt-and-suspenders fallback. Downside: a caller speaking a different language than the group's configured one now gets transcribed via the pinned language model, which degrades recognition for that caller. Acceptable trade — the rare multilingual caller vs the common in-language caller with a poor line.
3. **Tool latency budget is tight** — 2 s for case search, 2.5 s total if cases miss. If both time out, the bot says "I don't have that information" rather than stalling.
4. **Pool sweeper is hourly** — a number deleted at 26 d 23 h won't be released for up to an hour. Benign, but shows up as a brief window of "pooled but unusable by other tenants."
5. **Escalation has no retry** — if every admin misses, the missed-call card is posted and no further automated attempts are made. Admins reply in-chat to trigger a callback.
6. **AMD false positives** — Premium AMD is ~98% accurate, not perfect. A single `machine` classification skips that admin. No second-chance retry today.
7. **Pending stubs** — `telnyx_number='pending-<group_id>'` rows store language before a number is ordered. They're cleaned up on reactivation but can linger if a user sets language and never clicks Activate.
8. **Hard cap counts pooled** — if a user has 5 pooled numbers and 0 active, they still can't order a new one. They must delete a pooled number outright first (no UI for this yet).

---

## 11. Speech recognition quality — where we are and the ceiling

Gemini 3.1 Flash Live is an **end-to-end audio model**: audio in → audio out, with ASR, reasoning, and TTS fused inside one model. This gives us the things a pipelined stack can't match cheaply:

- Sub-350 ms round-trip latency (pipelined STT→LLM→TTS is typically 700-900 ms).
- Native barge-in with `interrupted` events when the caller talks over the bot.
- Prosody and emotion preserved — the model hears *how* you said it, not just *what*.
- One provider, one key, one bill.

The flip side is that the embedded ASR is measurably worse than a dedicated speech model (Deepgram Nova-3, Whisper v3) on accented, mumbled, or heavily noise-contaminated speech. When the model can't hear clearly, we have no ASR confidence score to branch on — we can only prompt the model to ask for a repeat.

### What we've tuned for max recognition quality within Gemini Live

- `languageCode` pinned to the group language so the model isn't auto-detecting per turn (§1).
- VAD loosened so the turn isn't closed mid-sentence (§1).
- System prompt explicitly instructs the model to ask for a repeat rather than guess or stall (§4).
- Caller-side transcript bubble is synthesised from the Gemini *tool-call* `question` argument (more accurate than the raw ASR transcript), not from `inputAudioTranscription` (`routes.py:435`).
- Bidirectional RTP stream so the model's audio actually reaches the caller without a second hop.

After these changes, "bot didn't understand me" misses dropped from frequent to rare on real calls. **We are at or near the ceiling of what Gemini Live can do today on narrowband μ-law audio.**

### When to move off Gemini Live

If post-deploy monitoring shows unclear-speech misses still happening often enough to matter, the next step is **a pipelined fallback provider behind the same `VoiceProvider` interface** (`provider.py`). Concretely: Telnyx Media Streams → Deepgram Nova-3 streaming STT → Gemini 2.5 Flash (text) → Chirp 3 TTS (or self-hosted Kokoro-82M on the Oracle GPU). ~1 engineering day. We lose ~400 ms of latency and native barge-in, but gain:

- ASR accuracy that doesn't fold on mumbled speech.
- Confidence scores per utterance — code can explicitly re-ask on low confidence instead of relying on the model to notice.
- Full observability of each layer (transcripts, confidence, LLM input/output, TTS audio) for debugging.
- Unit cost drops slightly (~$0.014/min vs ~$0.017/min).

The abstraction is in place already — `VoiceProvider` in `provider.py`. Adding a `PipelinedProvider` alongside `GeminiLiveProvider` is a drop-in, selectable per-group via a config flag.

### Dev experience comparison

| Dimension | Gemini 3.1 Flash Live (today) | Pipelined STT + text LLM + TTS |
|---|---|---|
| Latency | ~320 ms | ~700-900 ms |
| Barge-in | Native | Stitch yourself |
| Prosody | Preserved | Flattened |
| ASR on hard audio | Mediocre | Best-in-class |
| Confidence-gated re-ask | Prompt-only | Explicit per-turn |
| Observability when it fails | Black box | Transcript at every layer |
| Providers to manage | 1 | 3 |
| $/min | ~$0.017 | ~$0.014 |

**Recommendation today**: stay on Gemini Live. Observe for a week. If "didn't hear me" complaints persist → ship the pipelined fallback. If not, today's config is the practical ceiling and we move on to other work.
