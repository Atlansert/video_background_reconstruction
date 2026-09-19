"""Subtract SAM3.1-detected walls from the kitchen-section masks.

The kitchen sections (630-1030, 1440-1799) carry large wall over-masks that
date back to the original segmentation. With the missed objects (fridge /
wardrobe / glass panels) added on top, the fill reaches 0.6-0.7 of the frame
and SVOR's generation destabilizes: grey smears, black blobs, hallucinated
sofas/benches (measured on the 2026-09-19 re-runs). Walls are BACKGROUND in
the pure-background mode — they must come from real pixels via the source
composite, never from generation.

This tool detects walls with a low-threshold "wall" text prompt, dilates the
detection and subtracts it from the current masks inside the two kitchen
windows, then rebuilds masks_inpaint. masks/ outside the windows is untouched.

Usage (vbr-seg):
    python -m tools.subtract_walls            # detect + subtract + rebuild
    python -m tools.subtract_walls --dry-run  # coverage report only
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

from vbr.cli import PROJECT_ROOT
from vbr.config import load_config
from vbr.models.segmentation import SegmentationAdapter
from vbr.video import prepare_inpainting_masks

WINDOWS = [(630, 1030), (1440, 1799)]
WALL_PROMPTS = ["wall", "plain wall"]
THRESHOLDS = {"wall": 0.30, "plain wall": 0.35}
DILATE_PX = 5
INPAINT_PARAMS = dict(close_px=9, temporal_radius=2, dilate_px=1,
                      hull_min_area=0, hull_max_extra=0.35, overlap=0.12)


def in_windows(frame: int) -> bool:
    return any(a <= frame <= b for a, b in WINDOWS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/vggt_slam.yaml")
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = (PROJECT_ROOT / cfg["output_dir"]).resolve()
    all_frames = output_dir / "frames_all"
    masks = output_dir / "masks"
    logs_dir = output_dir / "logs"
    adapter = SegmentationAdapter(dict(cfg["segmentation"]), PROJECT_ROOT)

    frame_ids = [f for f in range(WINDOWS[0][0], WINDOWS[-1][1] + 1, args.stride)
                 if in_windows(f)]
    wall_dir = output_dir / "masks_wall_detection"
    already_detected = wall_dir.exists() and len(list(wall_dir.glob("*.png"))) > 100
    if not already_detected:
        shutil.rmtree(wall_dir, ignore_errors=True)
        env_name, env, checkpoint = adapter._environment()
        subset = wall_dir / "frames_subset"
        subset.mkdir(parents=True, exist_ok=True)
        for fid in frame_ids:
            source = all_frames / f"{fid:06d}.jpg"
            shutil.copy(source, subset / source.name)
        command = [
            "conda", "run", "--no-capture-output", "-n", env_name,
            "python", "-m", "vbr.sam31_keyframes",
            "--frames", str(subset.resolve()),
            "--output", str(wall_dir.resolve()),
            "--checkpoint", str(checkpoint),
            "--prompts-json", json.dumps(WALL_PROMPTS),
            "--preserve-prompts-json", "[]",
            "--prompt-thresholds-json", json.dumps(THRESHOLDS),
            "--threshold", "0.45",
            "--max-objects", "64",
            "--allow-empty-union",
        ]
        log_path = logs_dir / "sam31_wall_detection.log"
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(command, cwd=PROJECT_ROOT, env=env,
                                    stdout=log, stderr=subprocess.STDOUT, text=True)
        if result.returncode:
            tail = "\n".join(log_path.read_text(encoding="utf-8").splitlines()[-40:])
            raise RuntimeError(f"wall detection failed; see {log_path}\n{tail}")

    # Wall seeds exist only at strided frames; SAM2-propagate them to every
    # frame so odd frames subtract too (temporal union would otherwise pull
    # the wall back in through even neighbours).
    wall_prop = output_dir / "masks_wall_propagated"
    # sam2_propagate requires a non-empty seed at frame 0; the windows start
    # later, so a tiny placeholder is enough (frames outside the windows are
    # never subtracted, whatever the tracker does there).
    if not (wall_dir / "000000.png").exists():
        first = cv2.imread(str(sorted(wall_dir.glob("*.png"))[0]))
        stub = np.zeros(first.shape[:2], np.uint8)
        h, w = stub.shape
        stub[h // 2 - 150:h // 2 + 150, w // 2 - 150:w // 2 + 150] = 255
        cv2.imwrite(str(wall_dir / "000000.png"), stub)
    if not (wall_prop / "000000.png").exists():
        adapter.propagate(all_frames, wall_dir, wall_prop, logs_dir,
                          log_name="sam2_wall_propagate.log")

    kernel = np.ones((DILATE_PX, DILATE_PX), np.uint8)
    wall_source = output_dir / "masks_wall_propagated"
    if args.dry_run:
        stats = []
        for fid in frame_ids[::25]:
            mask = cv2.imread(str(masks / f"{fid:06d}.png"), 0)
            wall = cv2.imread(str(wall_source / f"{fid:06d}.png"), 0)
            if wall is None:
                continue
            wall_d = cv2.dilate((wall > 127).astype(np.uint8), kernel) * 255
            stats.append({"frame": fid,
                          "mask": round(float(np.count_nonzero(mask > 127)) / mask.size, 3),
                          "wall_in_mask": round(float(np.count_nonzero(
                              ((mask > 127) & (wall_d > 127)).sum()) / mask.size), 3)})
        print(json.dumps(stats, indent=1))
        return

    backup = output_dir / "snapshots" / "masks" / "masks_pre_wall_subtract"
    if not backup.exists():
        shutil.copytree(masks, backup)

    tmp = output_dir / "masks_wallsub_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    removed = 0
    for frame in range(1799):
        mask = cv2.imread(str(masks / f"{frame:06d}.png"), 0)
        if mask is None:
            raise RuntimeError(f"missing mask {frame}")
        if in_windows(frame):
            wall = cv2.imread(str(wall_source / f"{frame:06d}.png"), 0)
            if wall is None:
                raise RuntimeError(f"missing propagated wall mask {frame}")
            wall_d = cv2.dilate((wall > 127).astype(np.uint8), kernel) * 255
            removed += int(np.count_nonzero((mask > 127) & (wall_d > 127)))
            mask = np.where(wall_d > 127, 0, mask).astype(np.uint8)
        cv2.imwrite(str(tmp / f"{frame:06d}.png"), mask)

    shutil.rmtree(masks)
    tmp.rename(masks)
    prepare_inpainting_masks(masks, output_dir / "masks_inpaint", **INPAINT_PARAMS)
    print(json.dumps({"frames": 1799, "wall_pixels_removed": removed}))


if __name__ == "__main__":
    main()
