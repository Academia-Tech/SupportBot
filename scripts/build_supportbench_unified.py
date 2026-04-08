#!/usr/bin/env python3
"""
Build final unified SupportBench format from media exports.

Reads:  datasets/{name}/{name}.json  (raw Telegram export with media paths)
Writes: datasets/{name}.json          (unified SupportBench format with meta)

Unified format combines production-style IDs/timestamps with media paths and emoji reactions.
"""
import json
import statistics
from collections import Counter
from datetime import datetime
from pathlib import Path

DATASETS_DIR = Path(__file__).parent.parent / "datasets"

DATASETS = {
    "ua_ardupilot": {
        "source": "t.me/ardupilot_ua",
        "lang": "uk",
        "domain": "uav_drone_systems",
        "description": "Ukrainian Ardupilot UAV / drone flight controller support",
    },
    "ua_selfhosted": {
        "source": "t.me/selfhostedua",
        "lang": "uk",
        "domain": "selfhosting_infrastructure",
        "description": "Ukrainian self-hosting, Docker, server infrastructure support",
    },
    "domotica_es": {
        "source": "t.me/GizChinaHomeAssistant",
        "lang": "es",
        "domain": "smarthome_automation",
        "description": "Spanish Home Assistant / smart home automation support",
    },
    "naseros": {
        "source": "t.me/NASeros",
        "lang": "es",
        "domain": "nas_networking",
        "description": "Spanish NAS, networking, storage, and infrastructure support",
    },
    "lineageos": {
        "source": "t.me/Lineageos_group",
        "lang": "en",
        "domain": "mobile_os_customrom",
        "description": "English LineageOS custom ROM installation and troubleshooting",
    },
    "tasmota": {
        "source": "t.me/tasmota",
        "lang": "en",
        "domain": "iot_firmware",
        "description": "English Tasmota IoT device firmware flashing and configuration",
    },
}

DATASET_ORDER = [
    "ua_ardupilot", "ua_selfhosted",
    "domotica_es", "naseros",
    "lineageos", "tasmota",
]

PRETTY_NAMES = {
    "ua_ardupilot": "Ardupilot-UA",
    "ua_selfhosted": "SelfHost-UA",
    "domotica_es": "Domotica-ES",
    "naseros": "NASeros-ES",
    "lineageos": "LineageOS-EN",
    "tasmota": "Tasmota-EN",
}


def make_id(group: str, tg_id: int) -> str:
    return f"tg_{group}_{tg_id}"


def parse_ts(date_str: str) -> int:
    """Convert ISO date string to Unix timestamp in milliseconds."""
    if not date_str:
        return 0
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S.%f%z",
                "%Y-%m-%d %H:%M:%S%z", "%Y-%m-%d %H:%M:%S.%f%z"):
        try:
            dt = datetime.strptime(date_str, fmt)
            return int(dt.timestamp() * 1000)
        except ValueError:
            continue
    try:
        dt = datetime.fromisoformat(date_str.replace("+00:00", "").replace("Z", ""))
        return int(dt.timestamp() * 1000)
    except Exception:
        return 0


def convert_dataset(name: str) -> dict | None:
    """Convert raw export to unified format."""
    raw_file = DATASETS_DIR / name / f"{name}.json"
    if not raw_file.exists():
        print(f"  SKIP: {raw_file} not found")
        return None

    meta_info = DATASETS[name]

    with open(raw_file, "r", encoding="utf-8") as f:
        raw_msgs = json.load(f)

    # Build ID mapping for reply resolution
    tg_id_map = {m["id"]: make_id(name, m["id"]) for m in raw_msgs}

    messages = []
    for m in raw_msgs:
        msg_id = make_id(name, m["id"])
        # Always preserve reply_to with tg_ prefix, even if target is outside window
        raw_reply = m.get("reply_to")
        reply_to = make_id(name, raw_reply) if raw_reply else None

        messages.append({
            "id": msg_id,
            "group_id": name,
            "ts": parse_ts(m.get("date", "")),
            "sender": m.get("sender", "unknown"),
            "body": m.get("text", ""),
            "reply_to_id": reply_to,
            "grouped_id": m.get("grouped_id"),
            "media_type": m.get("media_type"),
            "media_path": m.get("media_path"),
            "webpage_url": m.get("webpage_url"),
            "reactions": m.get("reactions", {}),
            "views": m.get("views"),
            "forwards": m.get("forwards"),
        })

    # Compute stats
    total = len(messages)
    with_text = sum(1 for m in messages if m["body"].strip())
    with_reply = sum(1 for m in messages if m["reply_to_id"])
    with_media = sum(1 for m in messages if m["media_type"])
    with_media_file = sum(1 for m in messages if m["media_path"])
    unique_senders = len(set(m["sender"] for m in messages))
    total_reactions = sum(
        sum(m["reactions"].values()) if isinstance(m["reactions"], dict) else 0
        for m in messages
    )

    # Reaction emoji breakdown
    emoji_counts = Counter()
    for m in messages:
        if isinstance(m["reactions"], dict):
            for emoji, count in m["reactions"].items():
                emoji_counts[emoji] += count

    # Media type breakdown
    media_types = Counter(m["media_type"] for m in messages if m["media_type"])

    # Q&A exchanges
    msg_by_id = {m["id"]: m for m in messages}
    qa_exchanges = 0
    for m in messages:
        if m["reply_to_id"] and m["reply_to_id"] in msg_by_id:
            parent = msg_by_id[m["reply_to_id"]]
            if "?" in parent["body"] and len(parent["body"]) > 20 and len(m["body"]) > 20:
                qa_exchanges += 1

    # Time span
    timestamps = [m["ts"] for m in messages if m["ts"] > 0]
    first_ts = min(timestamps) if timestamps else 0
    last_ts = max(timestamps) if timestamps else 0

    dataset = {
        "meta": {
            "name": name,
            "pretty_name": PRETTY_NAMES.get(name, name),
            "version": "1.0",
            "benchmark": "SupportBench",
            **meta_info,
            "stats": {
                "total_messages": total,
                "with_text": with_text,
                "with_replies": with_reply,
                "reply_rate": round(with_reply / max(total, 1), 3),
                "with_media": with_media,
                "with_media_files": with_media_file,
                "unique_senders": unique_senders,
                "qa_exchanges": qa_exchanges,
                "total_reactions": total_reactions,
                "top_reactions": dict(emoji_counts.most_common(10)),
                "media_types": dict(media_types),
                "first_ts": first_ts,
                "last_ts": last_ts,
            },
        },
        "messages": messages,
    }

    return dataset


def main():
    print("Building unified SupportBench format...\n")

    all_stats = []
    for name in DATASET_ORDER:
        print(f"  Processing {name}...")
        dataset = convert_dataset(name)
        if dataset is None:
            continue

        # Write unified file (top-level)
        out_file = DATASETS_DIR / f"{name}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)

        s = dataset["meta"]["stats"]
        size_mb = out_file.stat().st_size / (1024 * 1024)
        print(f"    → {out_file.name}: {s['total_messages']:,} msgs, "
              f"{s['with_media_files']} media files, "
              f"{s['total_reactions']} reactions, "
              f"{size_mb:.1f} MB")
        if s["top_reactions"]:
            top = " ".join(f"{e}×{c}" for e, c in list(s["top_reactions"].items())[:5])
            print(f"    → Top reactions: {top}")

        all_stats.append(dataset["meta"])

    # Save manifest
    total_msgs = sum(s["stats"]["total_messages"] for s in all_stats)
    total_media = sum(s["stats"]["with_media_files"] for s in all_stats)
    total_reactions = sum(s["stats"]["total_reactions"] for s in all_stats)
    langs = sorted(set(s["lang"] for s in all_stats))
    domains = sorted(set(s["domain"] for s in all_stats))

    manifest = {
        "benchmark": "SupportBench",
        "version": "1.0",
        "total_messages": total_msgs,
        "total_media_files": total_media,
        "total_reactions": total_reactions,
        "languages": langs,
        "domains": domains,
        "datasets": all_stats,
    }
    manifest_file = DATASETS_DIR / "manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # Save stats
    stats_file = DATASETS_DIR / "stats.json"
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"  SUPPORTBENCH UNIFIED BUILD COMPLETE")
    print(f"{'='*60}")
    print(f"  Datasets:        {len(all_stats)}")
    print(f"  Total messages:  {total_msgs:,}")
    print(f"  Media files:     {total_media:,}")
    print(f"  Total reactions: {total_reactions:,}")
    print(f"  Languages:       {', '.join(langs)}")
    print(f"  Domains:         {len(domains)}")
    print(f"  Manifest:        {manifest_file}")
    print(f"  Stats:           {stats_file}")


if __name__ == "__main__":
    main()
