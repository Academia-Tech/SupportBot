#!/usr/bin/env python3
"""
Compute comprehensive SupportBench statistics for the paper.
Outputs: stats summary, LaTeX tables, and detailed analysis.
"""
import json
import statistics
from collections import Counter
from pathlib import Path
from datetime import datetime

DATASETS_DIR = Path(__file__).parent.parent / "datasets"

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

DOMAIN_NAMES = {
    "uav_drone_systems": "UAV/Drones",
    "selfhosting_infrastructure": "Self-hosting",
    "smarthome_automation": "Smart Home",
    "nas_networking": "NAS/Network",
    "mobile_os_customrom": "Mobile OS",
    "iot_firmware": "IoT Firmware",
}


def load_dataset(name: str) -> dict:
    with open(DATASETS_DIR / f"{name}.json", "r", encoding="utf-8") as f:
        return json.load(f)


def compute_stats(name: str, data: dict) -> dict:
    msgs = data["messages"]
    meta = data["meta"]
    msg_by_id = {m["id"]: m for m in msgs}

    # Basic counts
    total = len(msgs)
    with_text = sum(1 for m in msgs if m["body"].strip())
    with_reply = sum(1 for m in msgs if m["reply_to_id"])
    with_media = sum(1 for m in msgs if m.get("media_type"))
    unique_senders = len(set(m["sender"] for m in msgs))
    questions = sum(1 for m in msgs if "?" in m["body"])

    # Resolved reply chains (reply_to within our window)
    resolved_replies = sum(
        1 for m in msgs
        if m["reply_to_id"] and m["reply_to_id"] in msg_by_id
    )

    # Media breakdown
    media_types = Counter(m["media_type"] for m in msgs if m.get("media_type"))

    # Message length stats
    lengths = [len(m["body"]) for m in msgs if m["body"].strip()]
    avg_len = statistics.mean(lengths) if lengths else 0
    median_len = statistics.median(lengths) if lengths else 0

    # Short messages (< 10 chars)
    short = sum(1 for m in msgs if 0 < len(m["body"].strip()) < 10)
    empty = sum(1 for m in msgs if not m["body"].strip())

    # Code/technical content
    tech_markers = ["```", "http://", "https://", "192.168", "sudo ",
                    "docker ", "apt ", "pip ", "npm ", "git ",
                    "zigbee", "mqtt", "wifi", "gpio", "uart"]
    tech_msgs = sum(
        1 for m in msgs
        if any(kw in m["body"].lower() for kw in tech_markers)
    )

    # Q&A exchanges (question msg with resolved reply)
    qa_exchanges = 0
    for m in msgs:
        if m["reply_to_id"] and m["reply_to_id"] in msg_by_id:
            parent = msg_by_id[m["reply_to_id"]]
            if "?" in parent["body"] and len(parent["body"]) > 20 and len(m["body"]) > 20:
                qa_exchanges += 1

    # Conversation threads (connected components via reply chains)
    thread_sizes = []
    visited = set()
    for m in msgs:
        if m["id"] in visited:
            continue
        # Find thread root
        chain = [m["id"]]
        visited.add(m["id"])
        # Find all replies to this message
        for other in msgs:
            if other["reply_to_id"] in chain and other["id"] not in visited:
                chain.append(other["id"])
                visited.add(other["id"])
        if len(chain) > 1:
            thread_sizes.append(len(chain))

    avg_thread = statistics.mean(thread_sizes) if thread_sizes else 0
    max_thread = max(thread_sizes) if thread_sizes else 0

    # Time span
    timestamps = [m["ts"] for m in msgs if m["ts"] > 0]
    if timestamps:
        first_dt = datetime.fromtimestamp(min(timestamps) / 1000)
        last_dt = datetime.fromtimestamp(max(timestamps) / 1000)
        span_days = (last_dt - first_dt).days
    else:
        first_dt = last_dt = None
        span_days = 0

    # Messages per day
    msgs_per_day = total / max(span_days, 1)

    return {
        "name": name,
        "pretty_name": PRETTY_NAMES.get(name, name),
        "lang": meta["lang"],
        "domain": DOMAIN_NAMES.get(meta.get("domain", ""), meta.get("domain", "")),
        "total": total,
        "with_text": with_text,
        "with_reply": with_reply,
        "resolved_replies": resolved_replies,
        "reply_rate": resolved_replies / max(total, 1),
        "with_media": with_media,
        "media_types": dict(media_types),
        "unique_senders": unique_senders,
        "questions": questions,
        "qa_exchanges": qa_exchanges,
        "avg_msg_len": avg_len,
        "median_msg_len": median_len,
        "short_msgs": short,
        "empty_msgs": empty,
        "tech_msgs": tech_msgs,
        "num_threads": len(thread_sizes),
        "avg_thread_size": avg_thread,
        "max_thread_size": max_thread,
        "span_days": span_days,
        "msgs_per_day": msgs_per_day,
        "first_date": first_dt.strftime("%Y-%m-%d") if first_dt else "N/A",
        "last_date": last_dt.strftime("%Y-%m-%d") if last_dt else "N/A",
    }


def print_latex_table(all_stats: list[dict]):
    """Generate LaTeX table for the paper."""
    print("\n% ── LaTeX Table: SupportBench Dataset Statistics ──")
    print(r"\begin{table*}[t]")
    print(r"\centering")
    print(r"\small")
    print(r"\begin{tabular}{llrrrrrrr}")
    print(r"\toprule")
    print(r"\textbf{Dataset} & \textbf{Lang} & \textbf{Msgs} & \textbf{Users} & \textbf{Reply\%} & \textbf{Q\&A} & \textbf{Media} & \textbf{Tech\%} & \textbf{Days} \\")
    print(r"\midrule")

    total_msgs = 0
    total_users = 0
    total_media = 0
    total_qa = 0

    for s in all_stats:
        tech_pct = s["tech_msgs"] / max(s["total"], 1) * 100
        reply_pct = s["reply_rate"] * 100
        print(
            f"  {s['pretty_name']:<15} & {s['lang'].upper():<2} "
            f"& {s['total']:,} & {s['unique_senders']:,} "
            f"& {reply_pct:.1f} & {s['qa_exchanges']:,} "
            f"& {s['with_media']:,} & {tech_pct:.1f} "
            f"& {s['span_days']} \\\\"
        )
        total_msgs += s["total"]
        total_users += s["unique_senders"]
        total_media += s["with_media"]
        total_qa += s["qa_exchanges"]

    print(r"\midrule")
    print(
        f"  {'\\textbf{Total}':<15} & 3  "
        f"& {total_msgs:,} & {total_users:,} "
        f"& -- & {total_qa:,} "
        f"& {total_media:,} & -- "
        f"& -- \\\\"
    )
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\caption{SupportBench dataset statistics. Q\&A = question--answer exchanges with resolved reply chains. Tech\% = messages containing technical markers (URLs, code, protocols). Reply\% = fraction of messages with resolved in-window reply references.}")
    print(r"\label{tab:supportbench}")
    print(r"\end{table*}")


def print_summary(all_stats: list[dict]):
    """Print human-readable summary."""
    print(f"\n{'='*70}")
    print(f"  SUPPORTBENCH v1.0 — Comprehensive Statistics")
    print(f"{'='*70}")

    total_msgs = sum(s["total"] for s in all_stats)
    total_media = sum(s["with_media"] for s in all_stats)
    total_users = sum(s["unique_senders"] for s in all_stats)
    total_qa = sum(s["qa_exchanges"] for s in all_stats)
    total_questions = sum(s["questions"] for s in all_stats)
    langs = sorted(set(s["lang"] for s in all_stats))
    domains = sorted(set(s["domain"] for s in all_stats))

    print(f"\n  Overview:")
    print(f"    Datasets:         {len(all_stats)}")
    print(f"    Total messages:   {total_msgs:,}")
    print(f"    Total media:      {total_media:,}")
    print(f"    Unique senders:   {total_users:,}")
    print(f"    Questions (?):    {total_questions:,}")
    print(f"    Q&A exchanges:    {total_qa:,}")
    print(f"    Languages:        {', '.join(l.upper() for l in langs)} ({len(langs)})")
    print(f"    Domains:          {len(domains)}")

    print(f"\n  Per-dataset breakdown:")
    print(f"  {'Name':<18} {'Lang':<4} {'Msgs':>6} {'Users':>6} {'Reply%':>7} {'Q&A':>5} {'Media':>6} {'Avg len':>8} {'Threads':>8} {'Days':>5}")
    print(f"  {'-'*90}")
    for s in all_stats:
        print(
            f"  {s['pretty_name']:<18} {s['lang'].upper():<4} "
            f"{s['total']:>6,} {s['unique_senders']:>6,} "
            f"{s['reply_rate']*100:>6.1f}% {s['qa_exchanges']:>5,} "
            f"{s['with_media']:>6,} {s['avg_msg_len']:>7.0f}ch "
            f"{s['num_threads']:>7,} {s['span_days']:>5}"
        )

    # Media breakdown
    print(f"\n  Media types across all datasets:")
    all_media = Counter()
    for s in all_stats:
        all_media.update(s["media_types"])
    for mtype, count in all_media.most_common():
        print(f"    {mtype:<12} {count:>5,}")

    print(f"\n  Message length distribution:")
    print(f"    {'Dataset':<18} {'Avg':>6} {'Median':>7} {'Empty':>6} {'Short':>6} {'Tech':>6}")
    print(f"    {'-'*55}")
    for s in all_stats:
        print(
            f"    {s['pretty_name']:<18} {s['avg_msg_len']:>5.0f}ch "
            f"{s['median_msg_len']:>6.0f}ch "
            f"{s['empty_msgs']:>5,} {s['short_msgs']:>5,} {s['tech_msgs']:>5,}"
        )

    print(f"\n  Thread structure:")
    print(f"    {'Dataset':<18} {'Threads':>8} {'Avg size':>9} {'Max size':>9}")
    print(f"    {'-'*47}")
    for s in all_stats:
        print(
            f"    {s['pretty_name']:<18} {s['num_threads']:>7,} "
            f"{s['avg_thread_size']:>8.1f} {s['max_thread_size']:>8,}"
        )


def main():
    print("Computing SupportBench statistics...\n")

    all_stats = []
    for name in DATASET_ORDER:
        filepath = DATASETS_DIR / f"{name}.json"
        if not filepath.exists():
            print(f"  SKIP: {name} — file not found")
            continue
        print(f"  Processing {name}...")
        data = load_dataset(name)
        stats = compute_stats(name, data)
        all_stats.append(stats)

    print_summary(all_stats)
    print_latex_table(all_stats)

    # Save stats to JSON
    stats_file = DATASETS_DIR / "stats.json"
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=2, ensure_ascii=False)
    print(f"\n  Stats saved to: {stats_file}")


if __name__ == "__main__":
    main()
