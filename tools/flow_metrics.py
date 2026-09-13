"""Flow-normalized temporal consistency metrics for background videos.

- warping_error: mean |I_t - Warp(I_{t-1}, F_t->t-1)| — the residual after
  removing camera/scene motion, i.e. the true flicker measure (cf. tOF).
  Reported overall, inside the (dilated) inpaint fills, and outside.
- mask_jitter: mean frame-to-frame XOR fraction of the inpaint masks — how
  much the generation region boundary moves per frame.

Usage (vbr environment):
    python -m tools.flow_metrics --video outputs/001_sam31_slam/background_video.mp4 \
        --masks outputs/001_sam31_slam/masks_inpaint [--stride 5] [--out report.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def read_frames(path: Path):
    capture = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    return frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--masks", type=Path, default=None)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    frames = read_frames(args.video)
    count = len(frames)
    masks = None
    if args.masks is not None:
        masks = [
            cv2.imread(str(args.masks / f"{i:06d}.png"), cv2.IMREAD_GRAYSCALE)
            for i in range(count)
        ]
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

    warp_all, warp_inside, warp_outside = [], [], []
    jitter = []
    for t in range(1, count, max(1, args.stride)):
        gray_prev = cv2.cvtColor(frames[t - 1], cv2.COLOR_BGR2GRAY)
        gray_cur = cv2.cvtColor(frames[t], cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            gray_cur, gray_prev, None, 0.5, 3, 21, 3, 5, 1.2, 0
        )
        height, width = gray_cur.shape
        grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
        warped = cv2.remap(
            frames[t - 1],
            (grid_x + flow[:, :, 0]).astype(np.float32),
            (grid_y + flow[:, :, 1]).astype(np.float32),
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        residual = cv2.absdiff(frames[t], warped).astype(np.float32).mean(axis=2)
        warp_all.append(float(residual.mean()))
        if masks is not None and masks[t] is not None:
            fill = cv2.dilate((masks[t] > 0).astype(np.uint8), kernel_dilate) > 0
            outside = ~fill
            if fill.sum() > 100:
                warp_inside.append(float(residual[fill].mean()))
            if outside.sum() > 100:
                warp_outside.append(float(residual[outside].mean()))
            if masks[t - 1] is not None:
                xor = (masks[t] > 0) ^ (masks[t - 1] > 0)
                jitter.append(float(xor.mean()))

    report = {
        "video": str(args.video),
        "frames": count,
        "sampled_pairs": len(warp_all),
        "warping_error_mean": float(np.mean(warp_all)),
        "warping_error_p95": float(np.percentile(warp_all, 95)),
        "warping_error_inside_mean": float(np.mean(warp_inside)) if warp_inside else None,
        "warping_error_outside_mean": float(np.mean(warp_outside)) if warp_outside else None,
        "mask_jitter_mean": float(np.mean(jitter)) if jitter else None,
    }
    print(json.dumps(report, indent=2))
    if args.out is not None:
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
