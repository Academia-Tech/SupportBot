# SupportBot Deployment Guide -- Oracle Cloud

**Last Updated**: 2026-03-10

---

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Architecture Overview](#architecture-overview)
3. [Docker Compose Services](#docker-compose-services)
4. [Deployment Steps](#deployment-steps)
5. [Signal Desktop Linking](#signal-desktop-linking)
6. [Monitoring & Logs](#monitoring--logs)
7. [Troubleshooting](#troubleshooting)
8. [Commands Reference](#commands-reference)

---

## Prerequisites

### Required
- **Oracle Cloud VM** at `161.33.64.115` (Always Free A1.Flex ARM)
- **SSH key**: `~/.ssh/supportbot_ed25519`
- **Google AI API Key** (Gemini models + text-embedding-004)
- **Signal account** (phone number for the bot)
- **Docker + Docker Compose** on the VM

### Optional
- **Cloudflare R2** credentials (for image storage; falls back to local disk)
- **Vercel** account (for supportbot.info landing page)

### Required Credentials (.env)

```bash
# Google AI (Gemini API via OpenAI-compatible endpoint)
GOOGLE_API_KEY=AIzaSyB...

# Signal
SIGNAL_BOT_E164=+380730017651

# MySQL (defaults work with docker-compose)
MYSQL_HOST=db
MYSQL_PORT=3306
MYSQL_USER=supportbot
MYSQL_PASSWORD=supportbot
MYSQL_DATABASE=supportbot

# ChromaDB
CHROMA_URL=http://rag:8000
CHROMA_COLLECTION=cases

# Optional: Cloudflare R2 image storage
CLOUDFLARE_ACCOUNT_ID=...
CLOUDFLARE_ACCESS_KEY_ID=...
CLOUDFLARE_SECRET_ACCESS=...
CLOUDFLARE_BUCKET=...
```

---

## Architecture Overview

```
+------------------+     +-------------------+     +------------------+
|  Signal Users    | --> |   OCI VM          | --> |  Google AI API   |
|  (Groups/DMs)    |     |   161.33.64.115   |     |  (Gemini)        |
+------------------+     +-------------------+     +------------------+
                               |
                    +----------+----------+----------+
                    |          |          |          |
              +-----v----+ +---v---+ +----v----+ +---v---------+
              | signal-  | | MySQL | | Chroma  | | signal-     |
              | bot:8000 | |  :3306| | :8002   | | desktop:8001|
              +----------+ +-------+ +---------+ +-------------+
                    |
              +-----v------+     +----------+
              | signal-    |     | signal-  |
              | ingest     |     | web      |
              +------------+     +----------+
```

### Services

| Service | Port | Description |
|---------|------|-------------|
| `signal-bot` | 8000 | Main FastAPI backend: ingest, worker, LLM, RAG |
| `signal-desktop` | 8001 | Headless Signal Desktop, SQLCipher DB, HTTP API |
| `signal-ingest` | - | History import via QR linking |
| `signal-web` | - | Next.js case viewer (via Caddy reverse proxy) |
| `db` | 3306 | MySQL 8 |
| `rag` | 8002 | ChromaDB (dual-RAG: cases_scrag + cases_rcrag) |

**No Redis.** **No signal-cli-rest-api.** Signal communication uses signal-desktop (headless) with its SQLCipher database and HTTP API.

---

## Docker Compose Services

The deployment uses `docker-compose.yml` at the project root. Key volumes:

```yaml
volumes:
  - /var/lib/signal/bot:/var/lib/signal/bot         # Bot Signal state
  - /var/lib/signal/desktop:/home/signal/.config/Signal  # Desktop state
  - /var/lib/signal/ingest:/var/lib/signal/ingest   # Ingest state
  - /var/lib/history:/var/lib/history               # QR images + artifacts
  - /var/lib/chroma:/chroma/chroma                  # ChromaDB data
  - mysql_data:/var/lib/mysql                       # MySQL data
```

**Do not delete these directories or run `docker compose down -v`.**

---

## Deployment Steps

### Quick Deploy (Existing VM)

```bash
# Push code + restart services
./scripts/deploy-oci.sh push

# Full redeploy (push + build + start)
./scripts/deploy-oci.sh full
```

### First-Time Setup

```bash
# 1. Generate SSH key (if not exists)
ssh-keygen -t ed25519 -f ~/.ssh/supportbot_ed25519 -N ""

# 2. Configure .env with your values
cp env.example .env
# Edit .env

# 3. Initialize VM
./scripts/deploy-oci.sh init

# 4. Full deploy
./scripts/deploy-oci.sh full

# 5. Link Signal account
./scripts/deploy-oci.sh link-signal

# 6. Set bot avatar
./scripts/deploy-oci.sh set-avatar
```

### Environment Variables

Key settings (see `signal-bot/app/config.py` for full list):

```bash
# Models (defaults shown)
MODEL_IMG=gemini-3.1-pro-preview
MODEL_DECISION=gemini-2.5-flash
MODEL_EXTRACT=gemini-3.1-pro-preview
MODEL_CASE=gemini-3.1-pro-preview
MODEL_RESPOND=gemini-3.1-pro-preview
MODEL_BLOCKS=gemini-3.1-pro-preview
EMBEDDING_MODEL=text-embedding-004

# Buffer limits
BUFFER_MAX_AGE_HOURS=168       # 7 days
BUFFER_MAX_MESSAGES=300

# Signal Desktop
USE_SIGNAL_DESKTOP=true
SIGNAL_DESKTOP_URL=http://signal-desktop-arm64:8001

# Public URL for case links
PUBLIC_URL=https://supportbot.info
```

---

## Signal Desktop Linking

The bot uses Signal Desktop (headless) for message send/receive. Linking is done via QR code.

### Initial Link

1. SSH into the VM:
   ```bash
   ./scripts/deploy-oci.sh ssh
   ```

2. Check signal-desktop is running:
   ```bash
   cd supportbot
   docker compose logs -f signal-desktop
   ```

3. Get QR code screenshot:
   ```bash
   curl http://localhost:8001/screenshot > qr.png
   ```
   Or open `http://<server>:8001/screenshot` in a browser.

4. Scan from your phone:
   - Signal -> Settings -> Linked Devices -> Link New Device
   - Choose "Transfer message history" when prompted

5. Verify link:
   ```bash
   curl http://localhost:8001/status
   ```

### History Bootstrap (Admin DM Flow)

After linking, DM the bot to import group history:

1. Send any message -> bot detects language, sends welcome
2. Send group name -> bot searches for group, generates QR
3. Scan QR -> bot imports 45-day history, extracts cases
4. Import complete notification sent

---

## Monitoring & Logs

### View Logs

```bash
# All services
./scripts/deploy-oci.sh logs

# Specific service
./scripts/deploy-oci.sh logs signal-bot

# SSH + manual
ssh -i ~/.ssh/supportbot_ed25519 ubuntu@161.33.64.115
cd supportbot
docker compose logs -f signal-bot
```

### Health Check

```bash
curl http://161.33.64.115:8000/healthz
# Expected: {"ok":true, ...}
```

### Check Status

```bash
./scripts/deploy-oci.sh status
```

---

## Troubleshooting

### Bot Not Responding

1. Check logs: `./scripts/deploy-oci.sh logs signal-bot`
2. Check health: `curl http://161.33.64.115:8000/healthz`
3. Verify Signal is linked: `curl http://161.33.64.115:8001/status`

### Signal Desktop Issues

```bash
# Restart signal-desktop
./scripts/deploy-oci.sh ssh
cd supportbot
docker compose restart signal-desktop
```

### ChromaDB Issues

```bash
docker compose restart rag
```

### Database Issues

```bash
docker compose exec db mysql -u supportbot -psupportbot supportbot
```

### Worker Stalled

The `/healthz` endpoint reports `worker_heartbeat_age_s`. If this exceeds 300 seconds, the worker is stalled. Restart:

```bash
docker compose restart signal-bot
```

---

## Commands Reference

| Command | Description |
|---------|-------------|
| `./scripts/deploy-oci.sh init` | First-time VM setup |
| `./scripts/deploy-oci.sh push` | Push code + restart |
| `./scripts/deploy-oci.sh full` | Push + build + start |
| `./scripts/deploy-oci.sh ssh` | SSH into VM |
| `./scripts/deploy-oci.sh logs [service]` | View logs |
| `./scripts/deploy-oci.sh status` | Service status |
| `./scripts/deploy-oci.sh stop` | Stop services |
| `./scripts/deploy-oci.sh restart` | Restart services |
| `./scripts/deploy-oci.sh link-signal` | Link Signal account |
| `./scripts/deploy-oci.sh set-avatar` | Set bot avatar |

---

## Costs

| Resource | Cost |
|----------|------|
| OCI VM (Always Free A1.Flex) | Free |
| OCI Block Storage (50GB free) | Free |
| Google AI (Gemini API) | ~$0.01-0.05/1K messages |
| Cloudflare R2 | Free tier covers typical usage |
| Signal | Free |
| Vercel (supportbot.info) | Free |

---

## Security

- SSH key authentication only (no password auth)
- Signal Desktop bound to internal Docker network
- MySQL bound to internal Docker network
- API keys stored in `.env` (not committed to git)
- Admin whitelist for DM access (`ADMIN_WHITELIST`)
- Per-group tag targets for escalation (`/tag` command)

---

**Document Version**: 2.0
**Last Updated**: 2026-03-10
