"""Confidence-weighted fusion of ProPainter (stable propagation) and SVOR
(generative fills), following the reference architecture:

    I_final = C * I_propainter + (1 - C) * I_svor

The confidence map is ProPainter's temporal self-consistency: pixels whose
flow-warped residual stays low across a temporal window are real propagated
background (trust ProPainter); pixels where it flaps are propagated/hallucinated
fills (prefer SVOR).

Usage (vbr environment):
    python -m tools.confidence_fusion --svor <svor.mp4> --propainter <pp.mp4> \
        --masks outputs/001_sam31_slam/masks_inpaint --out fused.mp4 \
        [--tau 10] [--smooth-frames 7]
"""
from __future__ import annotations

import argparse
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
    parser.add_argument("--svor", required=True, type=Path)
    parser.add_argument("--propainter", required=True, type=Path)
    parser.add_argument("--masks", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--tau", type=float, default=10.0,
                        help="residual scale: C = exp(-R_smooth / tau)")
    parser.add_argument("--smooth-frames", type=int, default=7,
                        help="temporal window for smoothing the residual map")
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    svor = read_frames(args.svor)
    propainter = read_frames(args.propainter)
    count = min(len(svor), len(propainter))
    assert count > 0, "empty inputs"
    height, width = svor[0].shape[:2]
    masks = [
        cv2.imread(str(args.masks / f"{i:06d}.png"), cv2.IMREAD_GRAYSCALE)
        for i in range(count)
    ]
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

    import subprocess
    process = subprocess.Popen(
        [
            "/usr/bin/ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}", "-r", "29.97002997002997",
            "-i", "-",
            "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", "-an",
            str(args.out),
        ],
        stdin=subprocess.PIPE,
    )

    residuals = [None] * count  # per-frame flow residual vs previous frame
    grids = {}
    for t in range(1, count):
        gray_prev = cv2.cvtColor(propainter[t - 1], cv2.COLOR_BGR2GRAY)
        gray_cur = cv2.cvtColor(propainter[t], cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            gray_cur, gray_prev, None, 0.5, 3, 21, 3, 5, 1.2, 0
        )
        grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
        warped = cv2.remap(
            propainter[t - 1],
            (grid_x + flow[:, :, 0]).astype(np.float32),
            (grid_y + flow[:, :, 1]).astype(np.float32),
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        residuals[t] = cv2.absdiff(propainter[t], warped).astype(np.float32).mean(axis=2)
        if t % 400 == 0:
            print(f"flow {t}/{count}", flush=True)

    # temporally smoothed residual -> confidence
    stacked = np.stack(
        [r if r is not None else np.zeros((height, width), np.float32) for r in residuals]
    )
    window = max(1, args.smooth_frames)
    smooth = np.zeros_like(stacked)
    for offset in range(-window // 2, window // 2 + 1):
        smooth += np.roll(stacked, offset, axis=0)
    smooth /= (window + 1)
    confidence = np.exp(-smooth / max(1e-6, args.tau))  # [T, H, W] in (0, 1]

    fused_stats = {"mean_conf_fill": [], "mean_conf_outside": []}
    for t in range(count):
        conf = cv2.GaussianBlur(confidence[t], (7, 7), 2.5)[:, :, None]
        mask = masks[t]
        if mask is None:
            mask = np.zeros((height, width), np.uint8)
        fill = cv2.dilate((mask > 0).astype(np.uint8), kernel_dilate) > 0
        fused_stats["mean_conf_fill"].append(float(conf[fill].mean()))
        fused_stats["mean_conf_outside"].append(float(conf[~fill].mean()))
        blended = (
            propainter[t].astype(np.float32) * conf
            + svor[t].astype(np.float32) * (1.0 - conf)
        )
        process.stdin.write(np.clip(blended, 0, 255).astype(np.uint8).tobytes())
        if t % 400 == 0:
            print(f"fuse {t}/{count}", flush=True)
    process.stdin.close()
    process.wait()
    if process.returncode:
        raise RuntimeError("fusion encode failed")

    report = {
        "frames": count,
        "tau": args.tau,
        "mean_conf_fill": float(np.mean(fused_stats["mean_conf_fill"])),
        "mean_conf_outside": float(np.mean(fused_stats["mean_conf_outside"])),
    }
    print(f"FUSION REPORT: {report}")
    if args.report is not None:
        args.report.write_text(__import__("json").dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
