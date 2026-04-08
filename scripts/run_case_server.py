#!/usr/bin/env python3
"""Serve SupportBench cases locally for evaluation.

Run this before eval_supportbench.py. Cases are loaded from the cases cache
file and served at http://localhost:8099/case/eval_N with media from the
dataset directory.

Usage:
    python3 scripts/run_case_server.py --dataset ua_ardupilot
    # Then in another terminal:
    python3 scripts/eval_supportbench.py --dataset ua_ardupilot ...
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_case_server import start_case_server

DATASETS_DIR = Path(__file__).resolve().parent.parent / "datasets"


def main():
    parser = argparse.ArgumentParser(description="Serve SupportBench cases locally")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--cases-cache", default=None, help="Cases JSON (default: results/cases_<dataset>.json)")
    parser.add_argument("--port", type=int, default=8099)
    args = parser.parse_args()

    cases_path = args.cases_cache or f"results/cases_{args.dataset}.json"
    if not Path(cases_path).exists():
        print(f"ERROR: cases file not found: {cases_path}", file=sys.stderr)
        print("Run eval_supportbench.py first to generate cases, or specify --cases-cache", file=sys.stderr)
        sys.exit(1)

    dataset_json = DATASETS_DIR / f"{args.dataset}.json"
    if not dataset_json.exists():
        print(f"ERROR: dataset not found: {dataset_json}", file=sys.stderr)
        sys.exit(1)

    dataset_dir = DATASETS_DIR / args.dataset
    if not dataset_dir.exists():
        dataset_dir = DATASETS_DIR

    with open(cases_path) as f:
        cases = json.load(f)
    with open(dataset_json) as f:
        data = json.load(f)
    msgs = data["messages"] if isinstance(data, dict) else data

    server = start_case_server(cases, msgs, dataset_dir=dataset_dir, port=args.port)
    print(f"Case server running at http://localhost:{args.port}")
    print(f"  Cases: {len(cases)}")
    print(f"  Index: http://localhost:{args.port}/")
    print(f"  Example: http://localhost:{args.port}/case/eval_0")
    print(f"  Media: {dataset_dir}/media/")
    print(f"\nPress Ctrl+C to stop.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping...")
        server.shutdown()


if __name__ == "__main__":
    main()
