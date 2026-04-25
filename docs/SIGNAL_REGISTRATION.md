# Signal Registration & Persistence (prevent re-linking)

**Last Updated**: 2026-04-07

This project uses **Signal Desktop (headless)** inside Docker as the primary Signal adapter. The Signal registration/linking state is stored on disk under bind-mounted directories.

If those files are lost/emptied, Signal will show the device as unregistered and you'll have to link again.

---

## How to know the bot is fully ready

The system is "ready" when all of these are true:

- Docker services are up:

```bash
docker compose ps
```

- API health is OK:

```bash
curl http://localhost:8000/healthz
```

- Signal Desktop is linked and running:

```bash
curl http://localhost:8001/status
```

- History ingestion worker is running:

```bash
docker compose logs --tail=50 signal-ingest
```

To confirm **chat responding**, send a test message in a connected group that mentions the bot (see `BOT_MENTION_STRINGS`, e.g. `@supportbot`) and verify a response arrives.

---

## Prevent re-registration / re-linking

### 1) Persist Signal storage on the host (bind mounts)

Use **bind mounts** for Signal state so it can't disappear due to Docker volume renames.

The `docker-compose.yml` mounts:

- `/var/lib/signal/bot` -- main bot account state
- `/var/lib/signal/desktop` -- Signal Desktop data directory
- `/var/lib/signal/ingest` -- ingestion/link-device state used for history bootstrap
- `/var/lib/history` -- QR images + history artifacts

**Do not delete these directories.**

### 2) Never run "delete volumes" commands

Avoid commands that wipe persistent storage:

- `docker compose down -v`
- `docker system prune --volumes`

### 3) Don't unlink the device from your phone

If you remove the linked device in Signal (**Settings -> Linked devices**), you will have to link again. That's expected Signal behavior.

### 4) Back up the Signal state

Recommended backup (run on the host):

```bash
sudo mkdir -p /var/backups/supportbot
sudo tar -C /var/lib -czf /var/backups/supportbot/signal_$(date +%Y%m%d_%H%M%S).tgz signal
sudo tar -C /var/lib -czf /var/backups/supportbot/history_$(date +%Y%m%d_%H%M%S).tgz history
```

Restore example:

```bash
sudo tar -C /var/lib -xzf /var/backups/supportbot/signal_YYYYMMDD_HHMMSS.tgz
sudo tar -C /var/lib -xzf /var/backups/supportbot/history_YYYYMMDD_HHMMSS.tgz
docker compose up -d
```

---

## How to link the bot -- Signal Desktop (recommended)

Signal Desktop is the primary adapter. It runs headlessly with Xvfb (virtual display) and receives the full 45-day history transfer when you link.

### Initial linking

1. Make sure `signal-desktop` container is running:
   ```bash
   docker compose logs -f signal-desktop
   ```

2. Get a screenshot of the QR code (during first-time setup):
   ```bash
   curl http://localhost:8001/screenshot > qr.png
   ```
   Or open `http://<server>:8001/screenshot` in a browser.

3. Scan the QR code from your phone:
   - Signal -> Settings -> Linked devices -> Link new device
   - Choose "Transfer message history" when prompted

4. Wait for sync to complete (check status):
   ```bash
   curl http://localhost:8001/status
   ```

5. Trigger history ingestion by DM'ing the bot with the group name.

### Resource usage

Signal Desktop requires more resources than signal-cli:
- ~500MB RAM for Signal Desktop process
- ~2GB shared memory (set via `shm_size: 2gb` in docker-compose)

---

## Fallback: signal-cli mode

If you can't run Signal Desktop (e.g., limited resources), you can use the legacy signal-cli mode:

```bash
USE_SIGNAL_DESKTOP=false
```

**Limitation**: signal-cli does NOT receive the 45-day history transfer. It only captures new messages that arrive while `signal-cli receive` is running.

With signal-cli, you can link via the debug QR endpoint:

1. Set `HTTP_DEBUG_ENDPOINTS_ENABLED=true` in `.env`
2. Restart: `docker compose up -d --build signal-bot`
3. Open `http://<server>:8000/signal/link-device/qr` in a browser
4. Scan with Signal -> Settings -> Linked devices -> Link new device
5. Set `HTTP_DEBUG_ENDPOINTS_ENABLED=false` and restart

---

## Offline history extraction (manual)

For one-time imports, you can extract history from Signal Desktop on your local machine:

1. **Extract the key** (on Windows with Signal Desktop):
   ```bash
   python tests/decrypt_key_win.py
   ```
   This saves `signal_key.txt` to your Desktop.

2. **Copy the key and extract the backup**:
   ```bash
   # Copy signal_key.txt to tests/secrets/signal_key.txt
   # Extract your Signal Desktop backup to tests/data/extracted/Signal1/
   ```

3. **Export messages**:
   ```bash
   python tests/read_signal_db.py
   ```

4. **Mine cases**:
   ```bash
   python tests/mine_real_cases.py
   ```

This produces structured cases that can be imported into the bot's knowledge base.
