"""Temporal median smoothing applied only inside the inpaint mask.

Kills 1-frame flashes in the final background video by replacing each masked
pixel with the median of itself and its temporal neighbors. Runs after
ProPainter and is intended as a final lightweight pass.

Usage (vbr environment):
    python -m tools.post_temporal_smooth --output-dir outputs/001_sam31_slam \
        --kernel 3
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np


def smooth_video(
    input_video: Path,
    masks_dir: Path,
    output_video: Path,
    kernel: int = 3,
    coverage_floor: float = 0.02,
):
    capture = cv2.VideoCapture(str(input_video))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    radius = (int(kernel) - 1) // 2
    writer = cv2.VideoWriter(
        str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    buffer: list = []
    regions: list = []
    changed_pixels = 0
    smoothed_frames = 0
    start = 0
    frame_index = 0

    for t in range(total):
        target_end = min(total, t + radius + 1)
        while start + len(buffer) < target_end:
            ok, frame = capture.read()
            if not ok:
                target_end = start + len(buffer)
                break
            mask = cv2.imread(
                str(masks_dir / f"{frame_index:06d}.png"), cv2.IMREAD_GRAYSCALE
            )
            if mask is None:
                region = np.zeros((height, width), dtype=bool)
            else:
                eroded = cv2.erode((mask > 0).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
                region = (mask > 0) & eroded
            buffer.append(frame)
            regions.append(region)
            frame_index += 1
        center_index = min(t, radius)
        center_frame = buffer[center_index]
        region = regions[center_index]
        if len(buffer) >= 2 and region.mean() >= coverage_floor:
            median = np.median(
                np.stack(buffer, axis=0).astype(np.float32), axis=0
            ).astype(np.uint8)
            updated = center_frame.copy()
            updated[region] = median[region]
            changed_pixels += int(
                np.count_nonzero(updated[region] != center_frame[region])
            )
            smoothed_frames += 1
            center_frame = updated
        writer.write(center_frame)
        while start < t - radius + 1:
            buffer.pop(0)
            regions.pop(0)
            start += 1
    capture.release()
    writer.release()
    return {
        "kernel": int(kernel),
        "frames_written": total,
        "smoothed_frames": smoothed_frames,
        "changed_pixels": changed_pixels,
        "output": str(output_video.resolve()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--kernel", type=int, default=3)
    args = parser.parse_args()
    root = args.output_dir.resolve()
    background = root / "background_video.mp4"
    masks = root / "masks_inpaint"
    smoothed = root / "background_video_smoothed.mp4"
    report = smooth_video(background, masks, smoothed, kernel=args.kernel)
    (root / "post_temporal_smooth.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()