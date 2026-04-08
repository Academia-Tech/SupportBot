#!/usr/bin/env python3
"""
Upload SupportBench datasets to HuggingFace.

Reads HF_TOKEN and HF_USERNAME from .env, creates the dataset repo,
and uploads all dataset folders (JSON + media).

Usage:
    python scripts/upload_supportbench_hf.py
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo

# Load .env
load_dotenv(Path(__file__).parent.parent / ".env")

HF_TOKEN = os.environ["HF_TOKEN"]
HF_USERNAME = os.environ["HF_USERNAME"]
REPO_NAME = "SupportBench"
REPO_ID = f"{HF_USERNAME}/{REPO_NAME}"

DATASETS_DIR = Path(__file__).parent.parent / "datasets"

# Datasets to upload (subdirectories with media/)
DATASET_NAMES = [
    "ua_ardupilot", "ua_selfhosted",
    "domotica_es", "naseros",
    "lineageos", "tasmota",
]


def main():
    api = HfApi(token=HF_TOKEN)

    # Create repo (dataset type)
    print(f"Creating repo: {REPO_ID}")
    try:
        create_repo(
            repo_id=REPO_ID,
            repo_type="dataset",
            private=False,
            token=HF_TOKEN,
            exist_ok=True,
        )
        print(f"  Repo ready: https://huggingface.co/datasets/{REPO_ID}")
    except Exception as e:
        print(f"  Repo creation: {e}")

    # Upload README
    readme = DATASETS_DIR / "README.md"
    if not readme.exists():
        write_readme(readme)
    print("Uploading README.md...")
    api.upload_file(
        path_or_fileobj=str(readme),
        path_in_repo="README.md",
        repo_id=REPO_ID,
        repo_type="dataset",
    )

    # Upload each dataset folder
    for name in DATASET_NAMES:
        dataset_dir = DATASETS_DIR / name
        if not dataset_dir.exists():
            print(f"  SKIP: {name} — directory not found")
            continue

        json_file = dataset_dir / f"{name}.json"
        media_dir = dataset_dir / "media"

        if not json_file.exists():
            print(f"  SKIP: {name} — no JSON file")
            continue

        # Count files
        media_count = len(list(media_dir.iterdir())) if media_dir.exists() else 0
        media_size = sum(f.stat().st_size for f in media_dir.iterdir() if f.is_file()) / (1024*1024) if media_dir.exists() else 0

        print(f"\nUploading {name}: {json_file.name} + {media_count} media files ({media_size:.0f} MB)...")

        # Upload the whole folder
        api.upload_folder(
            folder_path=str(dataset_dir),
            path_in_repo=name,
            repo_id=REPO_ID,
            repo_type="dataset",
            commit_message=f"Add {name} dataset ({media_count} media files)",
        )
        print(f"  Done: {name}")

    # Upload unified top-level JSONs
    for name in DATASET_NAMES:
        fpath = DATASETS_DIR / f"{name}.json"
        if fpath.exists():
            print(f"Uploading unified {name}.json...")
            api.upload_file(
                path_or_fileobj=str(fpath),
                path_in_repo=f"{name}.json",
                repo_id=REPO_ID,
                repo_type="dataset",
            )

    # Upload top-level metadata files
    for f in ["manifest.json", "stats.json"]:
        fpath = DATASETS_DIR / f
        if fpath.exists():
            print(f"Uploading {f}...")
            api.upload_file(
                path_or_fileobj=str(fpath),
                path_in_repo=f,
                repo_id=REPO_ID,
                repo_type="dataset",
            )

    print(f"\n{'='*60}")
    print(f"  Upload complete!")
    print(f"  https://huggingface.co/datasets/{REPO_ID}")
    print(f"{'='*60}")


def write_readme(path: Path):
    """Generate a HuggingFace dataset card (fallback if datasets/README.md missing)."""
    # Copy from the canonical README
    canonical = DATASETS_DIR / "README.md"
    if canonical.exists():
        import shutil
        shutil.copy2(canonical, path)
        print(f"  Copied README from {canonical}")
        return
    print(f"  WARN: {canonical} not found, skipping README generation")


if __name__ == "__main__":
    main()
