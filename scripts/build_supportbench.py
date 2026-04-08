#!/usr/bin/env python3
"""
Convert Telegram exports into unified SupportBench format.

Input:  local_data/telegram_exports/{name}.json  (Telegram export format)
Output: datasets/{name}.json                      (SupportBench unified format)

Unified format matches production RawMessage schema:
{
  "meta": { ... dataset metadata ... },
  "messages": [
    {
      "id": str,              # unique message ID (tg_{group}_{msg_id})
      "group_id": str,        # dataset/group name
      "ts": int,              # Unix timestamp in milliseconds
      "sender": str,          # anonymized sender hash
      "sender_name": null,    # not available from TG exports
      "body": str,            # message text content
      "media": str|null,      # media type: "photo", "video", "document", etc.
      "reply_to_id": str|null,# ID of message being replied to
      "reactions": 0          # not available from TG exports
    }
  ]
}
"""
import json
import hashlib
import sys
from datetime import datetime
from pathlib import Path

EXPORTS_DIR = Path(__file__).parent.parent / "local_data" / "telegram_exports"
OUTPUT_DIR = Path(__file__).parent.parent / "datasets"

# Final SupportBench datasets
DATASETS = {
    # ua_ardupilot: already in datasets/ with rich format (reactions, media_path, etc.)
    # — kept as-is, not rebuilt from raw export.
    "mikrotik_ua": {
        "pretty_name": "MikroTik-UA",
        "source": "t.me/mtikua",
        "lang": "uk",
        "domain": "networking_infrastructure",
        "description": "Ukrainian MikroTik networking equipment troubleshooting",
    },
    "domotica_es": {
        "pretty_name": "Domotica-ES",
        "source": "t.me/GizChinaHomeAssistant",
        "lang": "es",
        "domain": "smarthome_automation",
        "description": "Spanish Home Assistant / smart home automation support",
    },
    "naseros": {
        "pretty_name": "NASeros-ES",
        "source": "t.me/NASeros",
        "lang": "es",
        "domain": "nas_networking",
        "description": "Spanish NAS, networking, storage, and infrastructure support",
    },
    "tasmota": {
        "pretty_name": "Tasmota-EN",
        "source": "t.me/tasmota",
        "lang": "en",
        "domain": "iot_firmware",
        "description": "English Tasmota IoT device firmware flashing and configuration",
    },
    "adguard_en": {
        "pretty_name": "AdGuard-EN",
        "source": "t.me/adguard_en",
        "lang": "en",
        "domain": "adblocking_vpn_dns",
        "description": "English AdGuard ad-blocking, VPN, and DNS filtering support",
    },
}


def make_msg_id(group: str, tg_id: int) -> str:
    """Create a unique message ID from group name and Telegram message ID."""
    return f"tg_{group}_{tg_id}"


def parse_ts(date_str: str) -> int:
    """Convert ISO date string to Unix timestamp in milliseconds."""
    if not date_str:
        return 0
    # Handle various ISO formats from Telethon
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S.%f%z",
                "%Y-%m-%d %H:%M:%S%z", "%Y-%m-%d %H:%M:%S.%f%z"):
        try:
            dt = datetime.strptime(date_str, fmt)
            return int(dt.timestamp() * 1000)
        except ValueError:
            continue
    # Fallback: try without timezone
    try:
        dt = datetime.fromisoformat(date_str.replace("+00:00", "").replace("Z", ""))
        return int(dt.timestamp() * 1000)
    except Exception:
        return 0


def convert_dataset(name: str, meta: dict) -> dict | None:
    """Convert a single Telegram export to SupportBench format."""
    input_file = EXPORTS_DIR / f"{name}.json"
    if not input_file.exists():
        print(f"  SKIP: {input_file} not found")
        return None

    with open(input_file, "r", encoding="utf-8") as f:
        raw_msgs = json.load(f)

    # Build reply_to ID mapping (tg msg_id -> our ID)
    tg_id_map = {m["id"]: make_msg_id(name, m["id"]) for m in raw_msgs}

    messages = []
    for m in raw_msgs:
        msg_id = make_msg_id(name, m["id"])
        reply_to = None
        if m.get("reply_to"):
            reply_to = tg_id_map.get(m["reply_to"])

        messages.append({
            "id": msg_id,
            "group_id": name,
            "ts": parse_ts(m.get("date", "")),
            "sender": m.get("sender", "unknown"),
            "body": m.get("text", ""),
            "reply_to_id": reply_to,
            "grouped_id": None,
            "media_type": m.get("media"),
            "media_path": None,
            "webpage_url": None,
            "reactions": None,
            "views": m.get("views"),
            "forwards": m.get("forwards"),
        })

    # Compute stats
    total = len(messages)
    with_text = sum(1 for m in messages if m["body"].strip())
    with_reply = sum(1 for m in messages if m["reply_to_id"])
    with_media = sum(1 for m in messages if m["media_type"])
    unique_senders = len(set(m["sender"] for m in messages))

    first_ts = messages[0]["ts"] if messages else 0
    last_ts = messages[-1]["ts"] if messages else 0

    dataset = {
        "meta": {
            "name": name,
            "version": "1.0",
            "benchmark": "SupportBench",
            **meta,
            "stats": {
                "total_messages": total,
                "with_text": with_text,
                "with_replies": with_reply,
                "reply_rate": round(with_reply / max(total, 1), 3),
                "with_media": with_media,
                "unique_senders": unique_senders,
                "first_ts": first_ts,
                "last_ts": last_ts,
            },
        },
        "messages": messages,
    }

    return dataset


def print_quality_report(dataset: dict):
    """Print a quality assessment for a dataset."""
    meta = dataset["meta"]
    stats = meta["stats"]
    msgs = dataset["messages"]

    print(f"\n{'='*60}")
    print(f"  {meta['name']} ({meta['lang'].upper()}) — {meta['domain']}")
    print(f"  {meta['description']}")
    print(f"{'='*60}")
    print(f"  Messages:     {stats['total_messages']:,}")
    print(f"  With text:    {stats['with_text']:,}")
    print(f"  Reply rate:   {stats['reply_rate']*100:.1f}%")
    print(f"  With media:   {stats['with_media']:,}")
    print(f"  Unique users: {stats['unique_senders']:,}")

    # Sample messages with replies (shows support pattern)
    reply_chains = []
    msg_by_id = {m["id"]: m for m in msgs}
    for m in msgs:
        if m["reply_to_id"] and m["body"].strip() and m["reply_to_id"] in msg_by_id:
            parent = msg_by_id[m["reply_to_id"]]
            if parent["body"].strip():
                reply_chains.append((parent, m))

    print(f"  Reply chains: {len(reply_chains):,}")

    # Show 5 sample Q&A pairs
    print(f"\n  Sample troubleshooting exchanges:")
    shown = 0
    for parent, reply in reply_chains:
        if len(parent["body"]) > 30 and len(reply["body"]) > 20 and shown < 5:
            p_text = parent["body"][:120].replace("\n", " ↵ ")
            r_text = reply["body"][:120].replace("\n", " ↵ ")
            mt_p = parent.get("media_type") or parent.get("media")
            mt_r = reply.get("media_type") or reply.get("media")
            media_p = f" [{mt_p}]" if mt_p else ""
            media_r = f" [{mt_r}]" if mt_r else ""
            print(f"    Q{media_p}: {p_text}")
            print(f"    A{media_r}: {r_text}")
            print()
            shown += 1


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("SupportBench Builder")
    print(f"Input:  {EXPORTS_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Datasets: {list(DATASETS.keys())}")

    all_stats = []

    # Include pre-built ua_ardupilot dataset
    ardupilot_file = OUTPUT_DIR / "ua_ardupilot.json"
    if ardupilot_file.exists():
        with open(ardupilot_file, "r", encoding="utf-8") as f:
            ard = json.load(f)
        all_stats.append(ard["meta"])
        print(f"\nKept existing: ua_ardupilot ({ardupilot_file.stat().st_size / (1024*1024):.1f} MB)")

    for name, meta in DATASETS.items():
        print(f"\nConverting {name}...")
        dataset = convert_dataset(name, meta)
        if dataset is None:
            continue

        # Save
        out_file = OUTPUT_DIR / f"{name}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)

        size_mb = out_file.stat().st_size / (1024 * 1024)
        print(f"  Saved: {out_file} ({size_mb:.1f} MB)")

        # Quality report
        print_quality_report(dataset)
        all_stats.append(dataset["meta"])

    # Summary
    print(f"\n{'='*60}")
    print(f"  SUPPORTBENCH SUMMARY")
    print(f"{'='*60}")
    total_msgs = sum(s["stats"]["total_messages"] for s in all_stats)
    total_media = sum(s["stats"]["with_media"] for s in all_stats)
    langs = sorted(set(s["lang"] for s in all_stats))
    domains = sorted(set(s["domain"] for s in all_stats))
    print(f"  Datasets:       {len(all_stats)}")
    print(f"  Total messages: {total_msgs:,}")
    print(f"  Total media:    {total_media:,}")
    print(f"  Languages:      {', '.join(langs)}")
    print(f"  Domains:        {len(domains)}")
    for d in domains:
        print(f"    - {d}")

    # Save manifest
    manifest = {
        "benchmark": "SupportBench",
        "version": "1.0",
        "total_messages": total_msgs,
        "total_media": total_media,
        "languages": langs,
        "domains": domains,
        "datasets": all_stats,
    }
    manifest_file = OUTPUT_DIR / "manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"\n  Manifest: {manifest_file}")


if __name__ == "__main__":
    main()
