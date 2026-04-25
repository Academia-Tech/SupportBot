# Phone Bot — Architecture & Stack Plan

Status: **draft / for review** · Date: 2026-04-12 · Owner: Pavel

## TL;DR

- **Telephony / PSTN / call control**: **Telnyx** — dial-to (sequential fallback), native AMD, bridge, Media Streams WebSocket, recording to storage. Pay-as-you-go per minute, $1+/mo per DID.
- **Realtime voice AI**: **Gemini 3.1 Flash Live** (`gemini-3.1-flash-live-preview`, launched 2026-03-26) — ~**$0.017/min all-in** (including Telnyx), which is ~**$9 per 500 minutes per business**. Hits the "≤$10–20 per group" budget.
  - Same token pricing as 2.5 Flash Live ($3/1M audio in, $12/1M audio out) but **better voice quality**, **320ms median latency**, **90.8% on ComplexFuncBench Audio**, and **15-min max audio session** (up from 10).
  - **One gotcha**: 3.1's tool calling is **synchronous only** — the conversation pauses while our `answer_from_kb` runs. Our RAG is typically 200–500ms so this is usually imperceptible; if it's not, we have the agent emit a "let me check that…" filler before the tool call. And we auto-fallback to 2.5 Flash Live (async tools) if 3.1 errors since it's still preview.
- **Skipping ElevenLabs for v1**: $99/mo plan fee + $0.08–0.10/min is ~5× over budget for the 500-minute target.
- **Fallback path** if Gemini Live gets flaky or expensive under load: DIY stack of Telnyx Media Streams + Deepgram Nova-3 streaming STT + Gemini 2.5 Flash (text) + self-hosted TTS on the existing Oracle GPU (Kokoro-82M / XTTS-v2) — ~$0.014/min, one extra engineering day.
- **Recommendation**: Ship v1 on **Telnyx + Gemini 3.1 Flash Live** with **2.5 Flash Live as automatic runtime fallback**, both behind a `VoiceProvider` interface so we can swap providers per group without touching the escalation FSM.

## Cost target — 500 minutes / business / month ≤ $10–20

| Provider | AI cost | + Telnyx ($0.0055/min) | **Total for 500 min** | Effective $/min |
|---|---|---|---|---|
| ElevenLabs Pro plan | $40 + **$99/mo plan fee** | $2.75 | **~$142** | $0.28 |
| ElevenLabs Business | ~$40 + **$330/mo plan fee** | $2.75 | **~$373** | $0.75 |
| OpenAI Realtime mini | ~$15 (with 2.5× context accumulation) | $2.75 | **~$17.75** | $0.036 |
| **Gemini 2.5 Flash Live** ⭐ | ~$6 (with 2× context accumulation) | $2.75 | **~$8.75** | **$0.017** |
| DIY: Deepgram + Gemini text + self-host TTS | ~$4 | $2.75 | **~$6.75** | $0.014 |

Gemini Live math, conservatively: at 25 tokens/sec of audio ([Live API spec](https://cloud.google.com/vertex-ai/generative-ai/pricing)), 500 min × 60 sec × 25 tok/sec = 750k audio tokens per direction. Assume 30% user talk, 30% bot talk, 40% silence → ~225k each. With 2× multiplier for session-context-window accumulation (Live API charges past-turn tokens on every new turn), that's 450k audio-in + 450k audio-out:

- Audio input: 0.45M × $3.00/1M = **$1.35**
- Audio output: 0.45M × $12.00/1M = **$5.40**
- Total AI: **$6.75 per 500 min** + Telnyx $2.75 = **$9.50 per 500 min per business**.

That gives us a comfortable margin: even if a noisy call blows the audio token count 3× over the estimate, we're still at ~$18/500 min.

## Stack comparison

### Gemini 3.1 Flash Live vs Gemini 2.5 Flash Live vs ElevenLabs Agents vs OpenAI Realtime mini (as of 2026-04)

| | **Gemini 3.1 Flash Live** ⭐ | Gemini 2.5 Flash Live | ElevenLabs Agents | OpenAI Realtime mini |
|---|---|---|---|---|
| Model ID | `gemini-3.1-flash-live-preview` | `gemini-2.5-flash-live` | eleven_conversational_v2 | `gpt-4o-mini-realtime-preview` |
| Status | **Preview** (launched 2026-03-26) | GA | GA | GA |
| Raw price | Audio in $3/1M tok, audio out $12/1M tok | Same | $0.10/min (Pro), $0.08 Business + $99–330/mo plan fee | Audio in $10/1M tok, audio out $20/1M tok |
| LLM included | ✓ one bill | ✓ | ✗ LLM adds 10–30% | ✓ |
| Effective $/min all-in | **~$0.017** | **~$0.017** | ~$0.11 + plan fee | ~$0.036 |
| 500 min/month cost | **~$9** | **~$9** | ~$142 (Pro) | ~$18 |
| Tool use mid-call | ⚠ **Synchronous only** — conversation pauses during tool run | ✓ Async | ✓ Server + client tools + webhooks | ✓ |
| ComplexFuncBench Audio | **90.8%** | baseline | — | — |
| Barge-in / interruptions | ✓ native | ✓ "Interrupt naturally even in noisy environments" | ✓ | ✓ |
| Output token limit | **65k** | 8k | n/a | n/a |
| Latency (median) | **320ms** | sub-300ms | extra hop through ElevenLabs US | sub-500ms |
| Max session | **15 min** audio-only | 10 min | No advertised cap | long |
| Voice naturalness | **"Google's highest-quality audio model"** | Good | Best in class | Good |
| Ukrainian voice | Expected ✓ (successor to multilingual 2.5 — verify day 1) | ✓ | ✓ Excellent | ✓ |
| Telnyx integration | DIY: Media Streams (μ-law 8kHz) ↔ Gemini WS (PCM16 16kHz). 200–400 lines of glue. | Same | First-party SIP trunking UI, partner integration | DIY |
| Call recording | We get raw audio from Telnyx → free | Same | Post-call webhook | Same |
| Dev UX | 6/10 — raw WebSocket, we build the glue | 6/10 | 9/10 — "connect Telnyx" button | 6/10 |

**Why Gemini 3.1 Flash Live wins for v1**: Single provider, one API key, Ukrainian works, tool calling works, and the cost lands **5–15× under budget**. 3.1 gives us better voice naturalness and much higher output token limits at no extra cost. The only downsides: (1) we build the Telnyx↔Gemini media bridge ourselves — about 2–3 days of WebSocket plumbing and μ-law/PCM16 resampling, nothing novel; (2) 3.1 is preview status and has synchronous-only tool calls. We mitigate #2 with two guardrails: fast RAG (already sub-500ms) and a pre-tool filler line ("Let me check that for you…") so the caller hears something while the KB lookup runs. And the `VoiceProvider` abstraction auto-falls-back to 2.5 Flash Live if the 3.1 session fails to open.

### Even cheaper fallback: DIY stack

If Gemini Live has stability issues (it's relatively new, and "charged per turn for all tokens in the Session Context Window" can cause surprise bills on long noisy calls), the fallback is assembling the pipeline ourselves:

| Component | Provider | Cost | Notes |
|---|---|---|---|
| Telephony | Telnyx (inbound + media stream) | $0.0055/min = $2.75/500 min | same as above |
| STT | [Deepgram Nova-3 streaming](https://deepgram.com/pricing) | $0.0077/min = **$3.85/500 min** | Best-in-class accuracy, streaming |
| LLM | **Gemini 2.5 Flash (text)** via our existing signal-bot RAG | ~$0 (pennies) | We already have this working |
| TTS | **Self-hosted [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) or XTTS-v2 on the Oracle VM** | $0 marginal | Kokoro is 82M params, runs on CPU at realtime. Or pay Google Cloud TTS Neural2 ~$1.20/500 min |
| **Total** | | **~$6.75/500 min** ($0.014/min) | Full control, free call recordings, no vendor lock |

This is the v2 escape valve. I'd build v1 on Gemini Live (simpler, faster to ship) and only drop to DIY if we hit actual pain.

**Verdict**: **Telnyx + Gemini 3.1 Flash Live** for v1, **with Gemini 2.5 Flash Live as the automatic runtime fallback**. We already have a working RAG in `signal-bot`; we expose it as a single HTTP function the Gemini agent calls mid-call. 2–3 days of engineering to bridge Telnyx Media Streams to Gemini's WebSocket (μ-law 8kHz ↔ PCM16 16kHz resampling in Python is ~50 lines). Then the rest of the work (escalation FSM, writeback, callback) is provider-independent.

### Telnyx — PSTN/number layer

| Item | Price | Source |
|---|---|---|
| Local DID rental | From **$1/mo** (volume discounts) | [numbers](https://telnyx.com/pricing/numbers) |
| Inbound voice | **$0.002/min** + SIP termination by country | [call-control](https://telnyx.com/pricing/call-control) |
| Outbound voice | **$0.002/min** + per-country termination (US ~$0.002, Ukraine/Germany higher, see rate card) | ditto |
| Media Streams (WebSocket audio) | **$0.0035/min** | ditto |
| Call Recording | **$0.002/min**, storage free | ditto |
| Premium AMD | Extra per detection, exact price on rate card | [AMD release](https://telnyx.com/release-notes/premium-answering-machine-detection) |

Capabilities we need, all confirmed:

- **Sequential "Find-Me/Follow-Me" dialing** — documented demo at [github.com/team-telnyx/demo-findme-ivr](https://github.com/team-telnyx/demo-findme-ivr). Pattern: on `call.initiated` you `dial` admin1; on `call.hangup` with a no-answer cause you issue a new `dial` to admin2; on answer you `bridge` to the original caller leg.
- **Warm transfer / bridge** — `POST /calls/:id/actions/bridge` with `call_control_id` of the answered admin leg and the original caller leg. ([bridge](https://developers.telnyx.com/api/call-control/bridge-call))
- **AMD** — Premium AMD returns `human`, `machine`, `silence`, and granular sub-types via webhook. We use it to skip voicemail and advance to the next admin.
- **Voice AI Assistant AMD-on-transfer** — native behavior to stop a transfer if it hits voicemail. ([release notes](https://telnyx.com/release-notes/voice-ai-assistants-amd-on-transfer))
- **Recording** — per-call or always-on, stored on Telnyx or forwarded to our S3. We pipe it to the existing signal-bot worker that already generates cases from chat logs.

**Regulated countries — Telnyx Regulatory Bundles**: Ukraine and some EU countries require a one-time regulatory bundle. You submit business docs **once per country** via `POST /v2/regulatory_requirements` ($0 fee, 24-72h review), and from then on `POST /v2/number_orders` returns instant numbers in that country **forever** — same API call, same latency as an unregulated order. Bundles are reusable across unlimited orders.

**Country tiers**:

| Tier | Countries | What's needed | Ships when |
|---|---|---|---|
| **Instant (v1)** | 🇺🇸 US, 🇨🇦 CA, 🇬🇧 UK, 🇦🇺 AU, 🇳🇱 NL, 🇵🇱 PL, 🇧🇷 BR, 🇸🇬 SG, 🇮🇱 IL | Nothing — `POST /v2/number_orders` works immediately | Day 1 |
| **Bundle-gated (post-launch)** | 🇺🇦 UA, 🇩🇪 DE, 🇫🇷 FR, 🇮🇹 IT, 🇪🇸 ES, 🇨🇭 CH, 🇦🇹 AT | One bundle per country, ~30 min Pavel's time + 24-72h upstream review per country | Whenever, in any order, independently |

**Zero idle cost**: bundles are free to hold, numbers only cost when actually ordered, and they're only ordered at the moment a real customer clicks Generate. No pool, no pre-provisioning, no capex.

**Post-launch country unlock flow**: submit a bundle → approval webhook → flip a config flag → "Coming soon" label disappears from that country's row in the dashboard picker. **No code deploy required.** Same `telnyx.create_number_order()` call the instant countries already use. Unlock order is up to Pavel; recommended priority is UA → DE → FR/IT/ES (batch) → CH/AT (tail).

### Alternatives we don't need right now

- **Twilio + ConversationRelay** — more mature docs and huge ecosystem, but Twilio's margins on voice + media streams are ~30% higher than Telnyx for the same features, and their AI relay still needs an external LLM provider. Not worth the switching cost.
- **Vapi.ai / Retell AI** — opinionated, good dev UX, but they are BYOK stacks that bill a platform fee ($0.05–0.13/min) on top of the LLM/STT/TTS you provide. We'd end up paying more than going direct to ElevenLabs.
- **LiveKit Voice Agents** — open-source, self-hostable, cheapest at scale (just your infra + model inference cost). Strong candidate for v2 if we outgrow ElevenLabs pricing. Not for v1.
- **Telnyx Voice AI Assistant** (their own managed agent) — $0.08/min for their LLM+TTS+STT orchestration. Interesting because it's one provider for everything, and they already integrate ElevenLabs voices. **This is a reasonable second option for v1** and I'd test it in parallel if we're worried about vendor lock.

## How it works — call flow

Matching the user-described flow exactly:

```
┌────────────┐  PSTN  ┌────────────┐  Media-Stream WS  ┌────────────────┐
│  Caller    │───────▶│   Telnyx   │──────────────────▶│ Gemini 3.1     │
│            │◀───────│  (trunk +  │◀──────────────────│ Flash Live     │
└────────────┘  PSTN  │  Call Ctrl)│                   │ (LLM+TTS+STT)  │
                      └──────┬─────┘                   └────────┬───────┘
                             │                                   │ function
                             │ webhooks                          │ call
                             ▼                                   ▼
                      ┌──────────────┐               ┌──────────────────┐
                      │  signal-bot  │◀──────────────│  answer_from_kb  │
                      │  phone-svc   │   HTTP        │  (our webhook)   │
                      │  (FastAPI)   │               └──────────────────┘
                      └──────┬───────┘
                             │
                 ┌───────────┼──────────────┬──────────────┐
                 ▼           ▼              ▼              ▼
           escalation   case writer   chat writeback  recording
              FSM                     (existing bot)      store
```

### Step-by-step

**1. User calls the bot number.**
- Telnyx hits our `/phone/webhook` FastAPI endpoint with `call.initiated`.
- `phone-svc` looks up the dialed DID → finds the group it's assigned to → pulls group KB config → issues `answer` + starts a **Gemini 3.1 Flash Live** WebSocket session with 2.5 Live as auto-fallback. Session is configured with:
  - Voice + language (from group locale, default English)
  - System prompt that includes group description and the "emit a filler line like 'Let me check that for you…' before calling tools" instruction
  - One tool: `answer_from_kb(question, lang)` → POSTs to signal-bot's existing `/ask` endpoint (dual keyword+RAG search, already implemented)
  - A `escalate_to_human(reason)` tool the agent can call to trigger handoff

**2. Bot immediately talks and figures out the problem.**
- Agent greets in group's language, asks what they need.
- Any non-trivial user utterance → agent emits filler → calls `answer_from_kb`. The tool hits signal-bot, which runs:
  - cached doc search + keyword_agent + ultimate_agent (existing code)
  - returns a short answer + a confidence score + top-3 citation case IDs
- Agent reads the answer. Repeat until either resolved or user explicitly asks for a human, or confidence < threshold twice in a row.

**3. If needed, bot escalates.**
- Agent calls `escalate_to_human("reason")`. Tool returns `{action: "hangup-and-escalate"}`.
- phone-svc has the agent say a handoff line, then closes the Gemini Live WebSocket — **but keeps the caller leg alive** on Telnyx (separate call_control_id).

**4. Escalation = direct bridge, no AI.**
- phone-svc creates an **outbound leg** to escalation_admin[0]'s number with `answering_machine_detection = premium` and `client_state = {stage: "escalation", caller_leg, admin_idx: 0, ticket_id}`.

**5. Ring admins in a loop until all tried.**
- `call.answered` **with AMD = human** + admin_idx == k → `bridge` caller_leg ↔ admin_leg. Mark ticket `connected`.
- `call.answered` with AMD = machine/silence, OR `call.hangup` with cause `no_answer`/`busy`/`rejected` → `hangup` the admin leg, advance `admin_idx`, dial the next admin.
- Timeout per dial: 25 seconds (configurable).
- If `admin_idx > len(escalations)` we've exhausted the list.

**6. All tried, none picked up.**
- phone-svc tells Telnyx to play a short TTS clip on the caller leg ("We couldn't reach a team member, they'll call you back — I'm sending them the details now") then `hangup` the caller.
- Creates a `phone_ticket` row (status `queued`) with: caller number, group, transcript so far, timestamp, ticket_id.
- **Writeback** — reuses the existing chat bridge. phone-svc calls signal-bot's existing group-message API to post a structured message to the admins' group chat:
  ```
  📞 Missed call from +1 555 123 4567
  They asked: "<summary>"
  Reply to this thread to call them back through the bot.
  Ticket: #T-1234
  ```
  Listens on the admin's reply via the same chat adapter. The admin just replies in Telegram/Signal/etc. — no new UI to learn.

**7. Admin replies → bot calls user back.**
- New webhook from signal-bot chat adapter to phone-svc: `ticket_reply(ticket_id, text)`.
- phone-svc starts an **outbound Gemini Live session** with:
  - System prompt: "You're calling <caller> back on behalf of <group>. An admin just answered the question. Read this answer naturally in <lang>, make sure they understood, and answer any follow-ups using the KB only."
  - The admin's reply pre-loaded as context
- Dials caller → AMD-gated (if voicemail, leave a TTS message instead) → plays the answer.

**8. Admin picked up during ring = bot is out.**
- Once `bridge` succeeds, phone-svc detaches. Only job: watch for `call.hangup` on either leg → close the ticket as `resolved_by_admin`.

**9. Optional recording → case generation.**
- Telnyx `record-start` on both legs when the call begins, or gate behind a per-group flag.
- On `call.recording.saved` webhook, phone-svc stores the URL + transcript, queues a background job that runs the existing `cases_from_transcript` pipeline to emit a new KB case. No human intervention needed — the next caller with the same question gets it from RAG for free.

## Data model — new tables

Minimal additions to MySQL (via `signal-bot/app/db/schema_mysql.py`):

```sql
-- One row per group that has phone enabled
CREATE TABLE phone_bots (
  group_id        VARCHAR(64) PRIMARY KEY,
  telnyx_number   VARCHAR(32) NOT NULL UNIQUE,
  country_code    CHAR(2)      NOT NULL,
  voice_id        VARCHAR(64)  NOT NULL,          -- Gemini Live voice id
  language        VARCHAR(8)   NOT NULL DEFAULT 'en',
  greeting        TEXT,                            -- optional override
  record_calls    TINYINT(1)   NOT NULL DEFAULT 0,
  created_at      DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Ordered escalation contacts
CREATE TABLE phone_escalations (
  id              BIGINT PRIMARY KEY AUTO_INCREMENT,
  group_id        VARCHAR(64) NOT NULL,
  position        INT         NOT NULL,            -- 0 = first to ring
  name            VARCHAR(128) NOT NULL,
  phone_e164      VARCHAR(32)  NOT NULL,
  UNIQUE KEY (group_id, position),
  FOREIGN KEY (group_id) REFERENCES phone_bots(group_id) ON DELETE CASCADE
);

-- Per-call ledger (one row per inbound call)
CREATE TABLE phone_calls (
  id                 BIGINT PRIMARY KEY AUTO_INCREMENT,
  telnyx_call_id     VARCHAR(128) UNIQUE,
  group_id           VARCHAR(64) NOT NULL,
  caller_e164        VARCHAR(32) NOT NULL,
  started_at         DATETIME    NOT NULL,
  ended_at           DATETIME,
  ai_seconds         INT         DEFAULT 0,
  bridged_admin_idx  INT,                           -- null if no bridge
  ticket_id          VARCHAR(64),                   -- set if escalation was written back
  recording_url      VARCHAR(512),
  transcript         LONGTEXT,
  status             ENUM('in_progress','resolved_by_ai','bridged','missed','failed') NOT NULL
);

-- Monthly usage per group, for the dashboard bar
CREATE TABLE phone_usage (
  group_id   VARCHAR(64),
  month      CHAR(7),             -- YYYY-MM
  seconds    INT NOT NULL DEFAULT 0,
  PRIMARY KEY (group_id, month)
);
```

## Backend components to build

All in `signal-bot/app/phone/`:

```
phone/
├── __init__.py
├── routes.py             # FastAPI endpoints: /phone/webhook, /phone/ws, /phone/ticket-reply, /phone/assign
├── telnyx_client.py      # thin wrapper: search_numbers, create_order, answer, dial, bridge, hangup, start_recording
├── provider.py           # abstract VoiceProvider base class
├── gemini_live.py        # Gemini 3.1 Flash Live WebSocket glue: μ-law 8k ↔ PCM16 16k resample, event loop, 2.5 fallback
├── tools.py              # answer_from_kb, escalate_to_human, schedule_callback — function-call handlers
├── escalation.py         # state machine: dial loop, AMD handling, timeout
├── writeback.py          # "missed call" chat message + reply listener → callback trigger
├── country_config.py     # enabled countries map, flipped by config file / env flag — no deploy needed to enable a country
└── usage.py              # seconds accounting → phone_usage table, used by dashboard
```

New web API endpoints for the dashboard:
- `POST /api/web/phone/assign` — live call to Telnyx: search available numbers in the requested country, order one under the group's ownership (using the matching regulatory bundle if the country is bundle-gated), persist to `phone_bots`, return the assigned number. Zero-reservation JIT flow — 2-5 seconds end-to-end.
- `GET /api/web/phone/usage?group_id=...` — returns `{month, seconds, limit_seconds}` so the dashboard bar shows real data instead of localStorage.
- `GET /api/web/phone/countries` — returns the current `{instant, pending, disabled}` per-country enablement map so the dashboard picker can show "Coming soon" labels without a code deploy.

Existing dashboard localStorage logic becomes the fallback when a group has no phone_bot row yet.

## Cost model — one example (Gemini Live + Telnyx)

A 5-minute inbound call on a US number where the AI resolves it (no escalation), no recording:

- Telnyx inbound: 5 × $0.002 = **$0.010**
- Telnyx media stream: 5 × $0.0035 = **$0.0175**
- Gemini 2.5 Flash Live audio (estimated): ~**$0.060** (5 min × conservative $0.012/min)
- **Total: ~$0.088 per call** — ~6× cheaper than the ElevenLabs path

A 10-minute escalated call (5 min AI + 2× 20s rings + 4 min admin bridge + 30s handoff):

- AI portion: 5 min × $0.012 = $0.060
- Telnyx bridged (2 outbound legs × 4 min): $0.016
- Failed rings: negligible
- Total: ~**$0.10**

**500 free minutes/month at $0.017/min = ~$8.50 / group / month in raw cost.** Plus ~$1-2/mo for the DID itself. **Total worst case per active group ≈ $10-11/mo.** Comfortable enough to keep 500 min free, with a simple per-group concurrency cap (3 simultaneous) so nobody can blow up the bill. If a group actually approaches 500 min/mo, that's real usage — fine to paywall at that point.

**Zero idle cost**: we don't pre-buy any numbers. A DID starts billing **only** when a real customer clicks Generate and we run `POST /v2/number_orders`. Before that first order, our Telnyx bill is $0. Regulatory bundles are also free to hold on file — no subscription or recurring charge for having them approved. Growth scales costs linearly and only with active usage.

## Phases to ship

| Phase | What | Time | Ship criterion |
|---|---|---|---|
| **P0 — done** | Dashboard modal: number gen (mock), escalations, usage bar, test-call UI | — | ✅ In prod behind Business gate |
| **P1** | Telnyx account wiring: `/phone/webhook` endpoint that just logs events, `country_config.py` with 9 instant countries enabled, test DID in US | 1 day | We can place a test US number, call it, see the webhook arrive |
| **P2** | Gemini 3.1 Flash Live WebSocket bridge (μ-law↔PCM16) + `answer_from_kb` function tool + pre-tool filler line + happy-path inbound call. 2.5 Flash Live auto-fallback. | 3 days | You can call the DID, the bot answers in the group's language, queries the KB, gives a real answer, hangs up. Round-trip latency ≤ 1.5s. English + EN smoke test passes. |
| **P3** | Escalation FSM + Premium AMD + `bridge` warm transfer | 2 days | Bot hangs up, sequentially rings admin list, bridges on first human answer; voicemail is correctly skipped. |
| **P4** | Missed-call writeback into existing chat adapters + `ticket_reply` → outbound callback | 2 days | Missed call shows up in Telegram; admin reply triggers a callback to the user with the answer. |
| **P5** | MySQL schema + phone_bots/escalations/calls/usage tables + dashboard wired to real backend (remove localStorage) + `/phone/assign` JIT number ordering + picker shows "Coming soon" labels for non-instant countries | 2 days | Dashboard shows live monthly usage. User clicks Generate → real Telnyx US/UK/CA number in their group in ≤5s. Regulated countries are visibly labeled. |
| **P6** | Call recording + case auto-generation job | 1 day | Every answered call produces a reusable KB case. |
| **Post-launch (async)** | Submit Telnyx Regulatory Bundles for UA → DE → FR/IT/ES → CH/AT. Each is ~30 min of form-filling + 24-72h upstream review. On approval webhook, flip country flag in `country_config.py` → no redeploy, the "Coming soon" label flips to instant. | ~0 eng-days per country, serialized by Pavel at his pace | Each unlocked country appears in the picker and returns a real DID in ≤5s |

Total to v1: **~11 engineering-days**. We can ship P1–P3 to a single internal group first and iterate. Regulated countries unlock independently post-launch without any code work beyond a config flip per country.

## Open questions to resolve before building

1. **Voicemail on the *caller's* side** — if we try to call them back (step 7) and it goes to voicemail, do we (a) leave a TTS message with the answer, (b) skip and text via SMS, or (c) keep retrying? Default (a) with a max of 1 attempt.
2. **Billing gate** — 500 free min/mo is our pricing promise. Hard cap or soft? If hard, we need a grace-then-pause state. If soft, when does overage get billed and at what rate? Need a decision before P5.
3. **AMD false positives** — Premium AMD ≠ perfect. Should the "voicemail detected → skip" rule require two consecutive voicemail classifications, or one? Start with one, monitor, tighten later.
4. **Concurrency** — set per-group concurrency limit to bound cost. Default **3 simultaneous calls per group**.
5. **Data residency** — Telnyx media nodes in EU + Gemini Live in Google EU regions give us sub-300ms round-trip for EU callers. Fine for launch.

---

**Decision needed from you before I start P1:**

- **Telnyx account** — do you already have one, or should I sign us up? Only company info for the parent SupportBot LLC account — no per-country KYC needed for the 9 instant countries.
- **Google Cloud / Vertex AI** — confirm we can use the existing `signal-bot` GCP project for Gemini Live, or spin up a new one. Need the Live API enabled on the project.
- **Green-light on ~$0.017/min unit economics** + ~$1-2/mo DID rental. Total ≈ $10-11/mo per active business at full 500 min usage. Zero idle cost. Per-group concurrency cap 3.
- **v1 country list confirmed**: 🇺🇸 US, 🇨🇦 CA, 🇬🇧 UK, 🇦🇺 AU, 🇳🇱 NL, 🇵🇱 PL, 🇧🇷 BR, 🇸🇬 SG, 🇮🇱 IL. Regulated countries (UA/DE/FR/IT/ES/CH/AT) show "Coming soon" in picker; unlocked post-launch one bundle at a time via `country_config.py` flag flip.
- **Skip ElevenLabs entirely** unless/until users complain about Gemini 3.1 voice quality. Revisit then.

Sources consulted (non-exhaustive):
- [ElevenLabs pricing page](https://elevenlabs.io/pricing)
- [ElevenLabs blog: we cut Conversational AI pricing](https://elevenlabs.io/blog/we-cut-our-pricing-for-conversational-ai)
- [ElevenLabs Agents tool docs](https://elevenlabs.io/docs/agents-platform/customization/tools)
- [Gemini Vertex AI pricing page](https://cloud.google.com/vertex-ai/generative-ai/pricing)
- [Gemini 2.5 Flash Live API docs](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/models/gemini/2-5-flash-live-api)
- [Telnyx call-control pricing](https://telnyx.com/pricing/call-control)
- [Telnyx number pricing](https://telnyx.com/pricing/numbers)
- [Telnyx AI agent comparison article](https://telnyx.com/resources/ai-agent-comparison)
- [Telnyx AMD documentation](https://developers.telnyx.com/docs/voice/programmable-voice/answering-machine-detection)
- [Telnyx Find-Me/Follow-Me demo](https://github.com/team-telnyx/demo-findme-ivr)
- [Telnyx Bridge API](https://developers.telnyx.com/api/call-control/bridge-call)
- [Telnyx Ukraine DID requirements](https://support.telnyx.com/en/articles/3739745-ukraine-did-requirements)
