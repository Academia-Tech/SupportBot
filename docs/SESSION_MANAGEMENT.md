# Admin Session Management

**Last Updated**: 2026-04-07

## Key Concept: Sessions Are Independent of Signal Chat History

The bot stores admin sessions in **MySQL** (`admin_sessions` table).
This is **completely independent** from Signal message history on the user's phone.

**Clearing Signal chat history does NOT reset the bot session.**
If an admin deletes their conversation with the bot in the Signal app, the bot
still has their session in the database -- the bot gets no notification about
client-side history deletion.

## When Sessions Are Created / Cleared

| Event | Effect |
|---|---|
| Admin sends first message (no session exists) | Session created; welcome message + language detection sent |
| Admin re-opens chat after clearing history | Session still exists -> no re-welcome, continue from last state |
| Admin removes bot from contacts / blocks bot | Session fully deleted (`_handle_contact_removed`) |
| Admin sends `/wipe` | All data including session deleted |
| Bot removed from a group | Group data deleted, admin session preserved |

## Session State Machine

```
[None / no session]
        |
        | first message received
        v
awaiting_group_name <-------------------------------------+
        |                                                  |
        | admin types a group name                        |
        v                                                  |
awaiting_qr_scan                                          |
        |                                                  |
        | QR scanned & history processed (success or fail)|
        v                                                  |
idle ---+--------------------------------------------------+
        |
        | admin sends a new group name -> back to awaiting_qr_scan
```

### State descriptions

- **`None` (no row)** -- User has never messaged the bot, or was fully wiped.
  Next message triggers welcome + language detection.

- **`awaiting_group_name`** -- Bot sent the onboarding prompt; waiting for the
  admin to type a group name.

- **`awaiting_qr_scan`** -- A HISTORY_LINK job is running; waiting for the admin
  to scan the QR code in Signal Desktop.
  If the admin sends a new group name now, the pending job is cancelled and the
  new group search starts immediately.

- **`idle`** -- Setup complete, admin can send new group names to link additional groups.

## Language Detection

Language is detected **once**, on the very first message, from the message text:

- Ukrainian-specific characters (i, yi, ye, g') -> `uk`
- Any other Cyrillic characters -> `uk`
- Latin-only text -> `en`
- Default -> `uk`

Language can be overridden at any time with `/ua` or `/en`.

**Note**: The Ukrainian language command is `/ua`, NOT `/uk`.

## Admin Commands

| Command | Description |
|---------|-------------|
| `/en` | Switch UI language to English |
| `/ua` | Switch UI language to Ukrainian |
| `/wipe` | Erase all groups, cases, and sessions for this admin |
| `/union <group>` | Join two groups into a union (shared RAG and docs) |
| `/split` | Remove the current group from its union |
| `/tag <phone1>,<phone2>` | Set per-group mention targets for escalation (comma-separated phone numbers) |

### /union and /split

Groups in a union share their RAG collections and documentation URLs. When
CaseSearchAgent searches, it queries all groups in the union together.

- `/union <group-name>` -- join the current group with another group
- `/split` -- remove the current group from its union

Union membership is tracked via `chat_groups.union_id`.

### /tag

By default, `[[TAG_ADMIN]]` in bot responses is replaced with @mentions of all
linked admins for the group. The `/tag` command overrides this to mention specific
phone numbers instead.

- `/tag +380501234567,+380509876543` -- set custom mention targets
- Stored in `chat_groups.tag_targets_json`

## Session Fields (MySQL `admin_sessions`)

| Column | Description |
|---|---|
| `admin_id` | Signal phone number / UUID of the admin |
| `state` | Current state: `awaiting_group_name`, `awaiting_qr_scan`, `idle` |
| `pending_group_id` | Signal group ID being linked |
| `pending_group_name` | Human-readable group name |
| `pending_token` | One-time token for the active HISTORY_LINK job |
| `lang` | `uk` or `en` |
| `updated_at` | Timestamp of last update |

## Admin Whitelist

The `ADMIN_WHITELIST` environment variable restricts which phone numbers can DM
the bot. If set (comma-separated), only listed numbers can initiate admin sessions.

The `SUPERADMIN_LIST` allows certain users to re-ingest any group even if they
are not the linked admin.
