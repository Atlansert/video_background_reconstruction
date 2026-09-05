"""Evaluate inpainted background videos for foreground residue and flicker.

Metrics (computed against the original frames and the foreground masks):
- residue: mean |background - original| inside the dilated mask. Inpainting
  should change masked pixels substantially; near-zero change means the
  foreground object was copied through untouched.
- residual_structure: std of (background - original) inside the mask. A
  blurred copy-through leaves low-gradient smears, so this complements the
  mean.
- flicker: mean |background(t+1) - background(t)| inside the mask, normalized
  by the same statistic outside the mask. A temporally stable inpainting
  tracks the ratio near 1; flickering fills push it well above 1.

Usage (vbr environment):
    python -m tools.evaluate_inpainting --output-dir outputs/001_sam31
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def read_frames(path: Path):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video {path}")
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        yield frame
    capture.release()


def evaluate(video_dir: Path, input_video: Path, stride: int, sample_frames: int, dilation_px: int):
    background_video = video_dir / "background_video.mp4"
    masks_dir = video_dir / "masks"

    background = []
    for index, frame in enumerate(read_frames(background_video)):
        if index % stride == 0:
            background.append(frame)
        if len(background) >= sample_frames:
            break

    original_iter = read_frames(input_video)
    original = []
    for index, frame in enumerate(original_iter):
        if index % stride == 0:
            original.append(frame)
        if len(original) >= len(background):
            break

    residues = []
    structures = []
    flicker_ratios = []
    for index, bg_frame in enumerate(background):
        frame_id = index * stride
        mask_path = masks_dir / f"{frame_id:06d}.png"
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None or index >= len(original):
            continue
        binary = (mask > 0).astype(np.uint8)
        inner = cv2.erode(binary, np.ones((3, 3), np.uint8)) > 0
        if inner.sum() < 50:
            continue
        diff = cv2.absdiff(bg_frame, original[index]).astype(np.float32)
        residues.append(float(diff[inner].mean()))
        structures.append(float(diff[inner].std()))

        if index + 1 < len(background):
            temporal = cv2.absdiff(bg_frame, background[index + 1]).astype(np.float32)
            dilated = cv2.dilate(binary, np.ones((dilation_px, dilation_px), np.uint8)) > 0
            outer = ~dilated
            inside = float(temporal[dilated].mean()) if dilated.any() else 0.0
            outside = float(temporal[outer].mean()) if outer.sum() > 100 else 1.0
            if outside > 1e-6:
                flicker_ratios.append(inside / outside)

    report = {
        "frames_sampled": len(residues),
        "stride": stride,
        "residue_mean": float(np.mean(residues)) if residues else None,
        "residue_p90": float(np.percentile(residues, 90)) if residues else None,
        "residual_structure_mean": float(np.mean(structures)) if structures else None,
        "flicker_ratio_mean": float(np.mean(flicker_ratios)) if flicker_ratios else None,
        "flicker_ratio_p90": float(np.percentile(flicker_ratios, 90))
        if flicker_ratios
        else None,
    }
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--input-video",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "video" / "001.mp4",
    )
    parser.add_argument("--stride", type=int, default=30)
    parser.add_argument("--sample-frames", type=int, default=60)
    parser.add_argument("--dilation-px", type=int, default=7)
    args = parser.parse_args()

    report = evaluate(
        args.output_dir, args.input_video, args.stride, args.sample_frames, args.dilation_px
    )
    destination = args.output_dir / "inpainting_evaluation.json"
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
