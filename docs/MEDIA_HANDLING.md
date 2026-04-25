# Media / Attachment Handling

**Last Updated**: 2026-04-07

## Live Group Messages (signal-bot)

When a message with an image arrives in a monitored group, the live ingestion
pipeline (`signal-bot/app/ingestion.py`) does the following:

1. Receives `image_paths` -- absolute paths to attachment files on disk
   (downloaded by signal-desktop into the local filesystem)
2. For each image, calls `llm.image_to_text_json()` which sends the image to
   Gemini (`model_img`, default: gemini-3.1-pro-preview) and returns
   `observations` + `extracted_text`
3. Appends the JSON result to `content_text` as `[image]\n{...}`
4. Uploads the image to **Cloudflare R2** (S3-compatible blob storage) via
   `signal-bot/app/r2.py`. Falls back to local disk if R2 is not configured.
   R2 uploads use infinite retry with exponential backoff.
5. Stores the image path reference in `raw_messages.image_paths_json`
6. Images are served via the `/r2/` proxy endpoint (or `/static/` for local files)

Non-image attachments (PDFs, files) are not currently OCR'd -- only images are
processed.

## Image Storage

### Cloudflare R2 (primary)

Configured via environment variables:
- `CLOUDFLARE_ACCOUNT_ID`
- `CLOUDFLARE_ACCESS_KEY_ID`
- `CLOUDFLARE_SECRET_ACCESS`
- `CLOUDFLARE_BUCKET`

R2 provides S3-compatible API. The `r2.py` module handles uploads with infinite
retry and exponential backoff to ensure uploads never silently fail.

### Local Fallback

If R2 is not configured, images are stored locally at `signal_bot_storage`
(default: `/var/lib/signal/bot/`) and served via the `/static/` mount.

## History Ingestion (signal-ingest -> signal-desktop)

Historical messages are read from Signal Desktop's SQLCipher database.

**`signal-desktop/app/db_reader.py`**

- Reads the `json` column from Signal Desktop's `messages` table
- Parses `attachments[]` array: extracts `path`, `fileName`, `contentType`
- Includes messages that have attachments even with no text body
- Attachment paths are relative to `<data_dir>/attachments.noindex/`

**`signal-desktop/app/main.py`**

- `/group/messages` response includes `attachments` per message
- `GET /attachment?path=<rel_path>` serves raw attachment bytes
  - Only serves paths inside `<data_dir>/attachments.noindex/` (path traversal blocked)
  - Returns correct `Content-Type` based on file extension

**`signal-ingest/ingest/main.py`**

After fetching messages, before chunking:

1. Identifies messages that have attachments (`m.get("attachments")`)
2. For each attachment:
   - **Image** (`image/*` content type): fetches bytes via `GET /attachment`,
     sends to Gemini for OCR, appends result as `[image: filename]\n<description>`
   - **Other file**: appends `[attachment: filename.ext]` to message body
3. This enriched body is then included in LLM chunks for case extraction

### Configuration

`MODEL_IMG` env var controls the Gemini model used for image OCR
(default: `gemini-3.1-pro-preview`). Set it in `.env` / docker-compose.

## Multimodal Limits

Configurable via environment (see `signal-bot/app/config.py`):

| Setting | Default | Description |
|---------|---------|-------------|
| `MAX_IMAGES_PER_GATE` | 3 | Max images sent to gate LLM call |
| `MAX_IMAGES_PER_RESPOND` | 5 | Max images sent to response LLM call |
| `MAX_KB_IMAGES_PER_CASE` | 2 | Max images stored per case |
| `MAX_IMAGE_SIZE_BYTES` | 5,000,000 | Skip images larger than 5MB |
| `MAX_TOTAL_IMAGE_BYTES` | 20,000,000 | Total image byte budget per call |

## Video Processing

When a video attachment arrives in a monitored group (`signal-bot/app/ingestion.py`):

1. **Thumbnail extraction**: OpenCV extracts a single frame at ~1 second for storage
   - Thumbnail uploaded to R2 as `{stem}_thumb.jpg`
2. **Full video description** via Gemini 2.5 Flash:
   - Videos longer than 120 seconds are trimmed to the first 2 minutes using
     `ffmpeg` stream copy (no re-encoding, near-instant)
   - Full video bytes sent to Gemini with a technical support-oriented prompt
   - Returns detailed description: visible screens, OCR text, sequence of events,
     anomalies
   - Falls back to thumbnail OCR if Gemini video description fails
3. **Audio transcription**: `ffmpeg` extracts the audio track (stream copy to m4a,
   fallback to mp3 re-encode), then Gemini 2.5 Flash transcribes speech verbatim
4. Results appended to `content_text`:
   - `[Відео: filename.mp4 — <description>]`
   - `[Транскрипт відео: <transcript>]`

### Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| `MAX_VIDEO_DURATION_SEC` | 120 | Trim videos longer than this before sending to Gemini |

### Dependencies

- **OpenCV** (`cv2`): thumbnail frame extraction
- **ffmpeg**: video trimming and audio extraction (must be in PATH)
- **Gemini 2.5 Flash**: video description and audio transcription

---

## What Is Not Handled

- **PDF attachments** in history: file name is noted but content is not extracted
- **URLs in message text**: not fetched or expanded -- the LLM only sees the raw URL string
