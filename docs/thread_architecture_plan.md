# Thread-Based Gate/Synthesizer Architecture

## Problem

Gate and synthesizer see a flat stream of ~40 interleaved messages. Even with msg_ids and reply arrows, the LLM must mentally reconstruct threads to decide who's OP vs helper, which thread a message belongs to. This causes redundant responses (bot answering helper chatter) and thread confusion.

## Core Idea

**Cases ARE threads.** The case extractor already groups related messages into cases with evidence_ids. Use this as the thread structure — feed gate/synthesizer a clean per-thread view with OP/HELPER labels instead of flat context. No new data structures.

## Pipeline

```
msg arrives
    ↓
Phase 1 (ALWAYS completes, persisted):
    case extractor runs
    → assigns msg to existing case(s) or creates new case
    → saves msg→case mapping to buffer (durable checkpoint)
    ↓
Phase 2 (interruptible, retryable):
    collect updated cases (deduped) + their new messages
    → gate extracts questions per case (sees clean thread with OP/HELPER labels)
    → synthesizer answers (same clean thread context)
    ↓
    if NEW msg arrives during Phase 2:
        → drop Phase 2
        → go back to Phase 1 with new msg (instant — just one new msg→case entry)
        → Phase 2 restarts with ALL accumulated unprocessed messages

Bot response → added to buffer as [BOT] message
    → next user message triggers pipeline, case extractor sees bot answer in thread
```

### Key Properties

- **Phase 1 is the checkpoint.** msg→case buffer is append-only, always persisted. Safe to interrupt Phase 2 anytime.
- **Phase 2 is stateless.** Pure function of the msg→case buffer. Safe to discard and re-run.
- **`has_newer_respond_job()` still works.** If newer message exists, drop current Phase 2 — the newer job picks up everything from the persisted buffer.
- **Same number of LLM calls.** Case extractor already runs. Gate/synth just get better input.

## What Gate Sees Now vs Proposed

### Now (flat, interleaved)

```
[Useruser_e msg_id=7391]: Телеметрія з ESC?
[Useruser_3 msg_id=7392 → Useruser_e msg_id=7391]: BATT_MONITOR,9
[Useruser_d msg_id=7395]: Як підключити гімбал?          ← different thread
[Useruser_e msg_id=7396 → Useruser_3 msg_id=7392]: Не працює
[Useruser_3 msg_id=7397 → Useruser_e msg_id=7396]: перевірте серіал  ← EVALUATE
```

### Proposed (single thread, labeled)

```
THREAD: "Телеметрія з Hobbywing ESC" (OP: Useruser_e)

[OP]     Useruser_e: Телеметрія з ESC?
[HELPER] Useruser_3 → OP: BATT_MONITOR,9
[OP]     Useruser_e → HELPER: Не працює
[HELPER] Useruser_3 → OP: перевірте серіал  ← NEW MESSAGE

Should the bot respond?
```

Classification becomes trivial: HELPER giving instructions to OP → don't respond.

## Design Details

### Case Extractor Modifications

- Relax non-overlapping constraint — threads interleave by design
- Output `evidence_ids` (msg_id list) as primary thread membership, not contiguous spans
- When new messages arrive, extractor assigns them to cases naturally (it already sees `msg_id` + `reply_to` in buffer)

### Role Labeling (code, no LLM)

- First sender in case evidence = **OP**
- Everyone else who replies = **HELPER**
- Bot = **BOT**

### Simplified Gate Prompt (~10 lines vs current 80)

```
THREAD from a technical support chat with labeled roles.
OP = original poster. HELPER = person helping.

consider=true ONLY if:
- OP describes a problem or asks for help
- OP provides diagnostic info needing interpretation
- A genuinely new question from anyone

consider=false if:
- HELPER giving advice/instructions to OP
- HELPER asking OP a diagnostic question
- Greetings, thanks, confirmations
- OP confirming solution worked
```

### Messages Not in Any Case

New standalone questions → single-message ephemeral thread, gate evaluates normally.
Pure noise (greetings, off-topic) → no case assignment, no gate call, auto-skip.

### Database / Storage

- `case_evidence` table already links cases to messages — no schema change
- Add msg→case buffer (could be a column on raw_messages, or a separate lightweight table)
- Cases with overlapping message membership are fine
- UI: thread messages shown prominently, non-thread messages transparent

## Implementation Phases

### Phase 1: Eval Prototype (validate idea)

All changes in `scripts/eval_supportbench.py`:

1. After ingestion, build `msg_id → case_indices` reverse index from evidence_ids
2. For each live message, assign to case(s) via reply chain / case extractor
3. Build thread context with OP/HELPER labels
4. Simplified gate prompt per thread
5. Compare precision/recall vs flat-context approach

### Phase 2: Production

1. Modify `_handle_buffer_update()` — case extractor persists msg→case assignments
2. Modify `_handle_maybe_respond()` — look up case membership, build thread context
3. Relax case extractor prompt constraints (overlapping spans)
4. Add role labels to context formatting
5. Update gate prompt (simplified thread version)

## Expected Impact

- Precision: 89% → 95%+ (eliminates helper-to-OP misclassification)
- Recall: stays at ~100% (OP questions clearly labeled)
- Quality: may improve (focused context, no thread confusion)

## When to Implement

After validating prompt-only fixes (msg_id + SKIP rules). If v69 achieves >93% precision, threads deferred to v2. If not, threads are next step.
