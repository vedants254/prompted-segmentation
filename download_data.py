"""
Download both Roboflow datasets into data/ with correct formats.

Setup:
    1. Create a free account at https://roboflow.com
    2. Go to Settings -> API Key
    3. Create a .env file with:  ROBOFLOW_API_KEY=your_key_here
    4. Run:  python download_data.py
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from roboflow import Roboflow

load_dotenv()

API_KEY = os.getenv("ROBOFLOW_API_KEY")
if not API_KEY:
    raise RuntimeError(
        "ROBOFLOW_API_KEY not found in .env file.\n"
        "Create a .env file with:  ROBOFLOW_API_KEY=your_key_here\n"
        "Get your free key: https://app.roboflow.com/settings/api"
    )

DATA_ROOT = Path("data")

# ── Dataset specs ─────────────────────────────────────────────────────────────
DATASETS = {
    "taping": {
        "workspace": "vedants-workspace-z9n1a",
        "project":   "drywall-join-detect-6hifq",
        "version":   1,
        "format":    "yolov8",           # bbox labels: class cx cy w h
    },
    "cracks": {
        "workspace": "vedants-workspace-z9n1a",
        "project":   "cracks-3ii36-eio8q",
        "version":   1,
        "format":    "yolov8",           # seg labels: class x1 y1 x2 y2 ...
    },
}


def download():
    rf = Roboflow(api_key=API_KEY)

    for name, spec in DATASETS.items():
        dest = DATA_ROOT / name
        if dest.exists() and any(dest.iterdir()):
            print(f"[skip] {name}/ already exists - delete it to re-download")
            continue

        print(f"\n{'='*60}")
        print(f"Downloading: {name}  ({spec['workspace']}/{spec['project']})")
        print(f"{'='*60}")

        project = rf.workspace(spec["workspace"]).project(spec["project"])

        if "version" in spec:
            v = project.version(spec["version"])
        else:
            versions = project.versions()
            if not versions:
                print(f"  [error] No versions found for {name} - skipping")
                continue
            v = versions[0]

        print(f"  Using version: {v.version}")
        v.download(spec["format"], location=str(dest))
        print(f"  [done] saved to {dest}")

    # Summary
    print(f"\n{'='*60}")
    print("Dataset summary:")
    for split in ["train", "valid", "test"]:
        for ds_name in ["taping", "cracks"]:
            p = DATA_ROOT / ds_name / split / "images"
            n = len(list(p.glob("*"))) if p.exists() else 0
            print(f"  {ds_name}/{split}: {n} images")


if __name__ == "__main__":
    download()
