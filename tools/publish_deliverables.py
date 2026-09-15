"""Publish the latest deliverables to GitHub.

Copies the current video/geometry/reports into the git-tracked ``deliverables/``
directory, writes a manifest with timestamp + revision + key config, commits
and pushes the current branch. Run after every accepted modification:

    python -m tools.publish_deliverables            # commit + push
    python -m tools.publish_deliverables --no-push  # commit only
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/001_sam31_slam"
DEST = ROOT / "deliverables"

FILES = [
    "background_video.mp4",
    "background_scene.glb",
    "background_mesh.ply",
    "mask_overlay.mp4",
    "mask_overlay_inpaint.mp4",
]
MASK_DIRS = ["masks", "masks_inpaint"]
REPORTS = [
    "inpainting_evaluation.json",
    "video_evaluation.json",
    "miss_windows.json",
    "geometry_redepth_report.json",
    "pipeline_status.json",
]


def git(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"git {args[0]} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-push", action="store_true")
    args = parser.parse_args()

    DEST.mkdir(exist_ok=True)
    (DEST / "reports").mkdir(exist_ok=True)
    for name in FILES:
        source = OUT / name
        if source.exists():
            shutil.copy2(source, DEST / name)
        else:
            print(f"warn: skipping missing {name}")
    for name in REPORTS:
        source = OUT / name
        if source.exists():
            shutil.copy2(source, DEST / "reports" / name)
    masks_archive = DEST / "masks.tar.gz"
    with tarfile.open(masks_archive, "w:gz") as tar:
        for name in MASK_DIRS:
            source = OUT / name
            if source.exists():
                tar.add(source, arcname=name)

    manifest = {
        "published_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "revision": git("rev-parse", "--short", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "backend": "svor",
        "files": FILES + ["masks.tar.gz"],
        "mask_dirs": MASK_DIRS,
    }
    (DEST / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    git("add", "deliverables")
    staged = git("status", "--porcelain", "deliverables")
    if not staged:
        print("deliverables unchanged; nothing to commit")
        return
    git(
        "commit",
        "-m",
        f"deliverables: {manifest['published_at']} ({manifest['revision']}) svor 产出",
    )
    if not args.no_push:
        branch = manifest["branch"]
        git("push", "origin", branch)
        print(f"pushed deliverables to origin/{branch}")
    print("published:", ", ".join(FILES))


if __name__ == "__main__":
    main()
