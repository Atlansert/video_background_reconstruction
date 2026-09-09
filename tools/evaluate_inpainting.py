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


def copy_fractions(
    original_video: Path,
    background_video: Path,
    masks_dir: Path,
    stride: int = 5,
    threshold: float = 11.0,
    max_frames: int = 0,
) -> dict:
    """Per-sampled-frame copy-through fraction inside the eroded mask.

    Copy-through means the inpainted output is still nearly identical to the
    original frame ({video_index: fraction}).
    """
    paths = sorted(masks_dir.glob("*.png"), key=lambda p: int(p.stem))
    sampled = [index for index in range(len(paths)) if index % stride == 0]
    if max_frames:
        sampled = sampled[:max_frames]
    sampled_ids = {index: int(paths[index].stem) for index in sampled}
    originals = {}
    for index, frame in enumerate(read_frames(original_video)):
        if index in sampled_ids:
            originals[index] = frame
        if len(originals) >= len(sampled_ids):
            break
    backgrounds = {}
    for index, frame in enumerate(read_frames(background_video)):
        if index in sampled_ids:
            backgrounds[index] = frame
        if len(backgrounds) >= len(originals):
            break
    fractions = {}
    for index, frame_id in sampled_ids.items():
        if index not in originals or index not in backgrounds:
            continue
        mask = cv2.imread(str(masks_dir / f"{frame_id:06d}.png"), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        inner = cv2.erode((mask > 0).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        if not inner.any():
            continue
        diff = cv2.absdiff(originals[index], backgrounds[index]).astype(np.float32).mean(axis=2)
        fractions[int(frame_id)] = float((diff[inner] < threshold).mean())
    return fractions


def windowed_copy_report(
    fractions: dict, windows: list, overall_frames: int | None = None
) -> dict:
    """Split the copy-through fractions into the persistent-miss windows.

    ``windows`` items are {start, end} frame-id ranges. Fractions outside any
    window are reported as "outside". The overall mean is included so the
    window numbers stay comparable to the headline copy_through_mean.
    """
    if not fractions:
        return {"frames_sampled": 0, "per_window": [], "in_window_mean": None, "outside_mean": None}
    per_window = []
    window_fractions = []
    for window in windows:
        values = [
            value
            for frame_id, value in sorted(fractions.items())
            if window["start"] <= frame_id <= window["end"]
        ]
        if values:
            per_window.append(
                {
                    "start": window["start"],
                    "end": window["end"],
                    "frames_sampled": len(values),
                    "mean": float(np.mean(values)),
                    "max": float(np.max(values)),
                }
            )
            window_fractions.extend(values)
    outside = [
        value
        for frame_id, value in sorted(fractions.items())
        if not any(window["start"] <= frame_id <= window["end"] for window in windows)
    ]
    return {
        "frames_sampled": len(fractions),
        "per_window": per_window,
        "in_window_mean": float(np.mean(window_fractions)) if window_fractions else None,
        "outside_mean": float(np.mean(outside)) if outside else None,
        "overall_mean": float(np.mean(list(fractions.values()))),
    }


def glitch_flags(
    background_video: Path,
    masks_dir: Path,
    stride: int = 30,
    max_frames: int = 60,
    dilation_px: int = 7,
    ratio_threshold: float = 1.8,
    abs_threshold: float = 8.0,
) -> dict:
    """Count frames whose masked-region temporal change dwarfs the unmasked one.

    A glitch frame is one where the mean temporal gradient inside the dilated
    mask exceeds ``ratio_threshold`` times the unmasked p95 gradient, with an
    absolute floor so quiet scenes do not flag noise.
    """
    backgrounds = []
    for index, frame in enumerate(read_frames(background_video)):
        if index % stride == 0:
            backgrounds.append((index, frame))
        if len(backgrounds) >= max_frames:
            break
    glitches = []
    kernel = np.ones((dilation_px, dilation_px), np.uint8)
    for position, (frame_id, frame) in enumerate(backgrounds[:-1]):
        next_frame = backgrounds[position + 1][1]
        mask = cv2.imread(str(masks_dir / f"{frame_id:06d}.png"), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        dilated = cv2.dilate((mask > 0).astype(np.uint8), kernel) > 0
        outside = ~dilated
        if not dilated.any() or outside.sum() < 100:
            continue
        temporal = cv2.absdiff(frame, next_frame).astype(np.float32).mean(axis=2)
        inside_mean = float(temporal[dilated].mean())
        outside_p95 = float(np.percentile(temporal[outside], 95))
        if inside_mean > ratio_threshold * outside_p95 and inside_mean - outside_p95 > abs_threshold:
            glitches.append(
                {
                    "frame_id": frame_id,
                    "inside_mean": round(inside_mean, 2),
                    "outside_p95": round(outside_p95, 2),
                }
            )
    return {
        "frames_checked": len(backgrounds) - 1,
        "glitch_frames": glitches,
        "glitch_count": len(glitches),
        "glitch_fraction": len(glitches) / max(1, len(backgrounds) - 1),
    }


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
    parser.add_argument(
        "--windows-json",
        type=Path,
        default=None,
        help="Path to a JSON list of {start, end} persistent-miss windows "
        "(e.g. miss_windows.json) for the per-window copy report",
    )
    args = parser.parse_args()

    report = evaluate(
        args.output_dir, args.input_video, args.stride, args.sample_frames, args.dilation_px
    )
    masks_dir = args.output_dir / "masks"
    inpaint_masks = args.output_dir / "masks_inpaint"
    copy_masks_dir = inpaint_masks if inpaint_masks.exists() else masks_dir
    windows = []
    if args.windows_json and args.windows_json.exists():
        loaded = json.loads(args.windows_json.read_text(encoding="utf-8"))
        if isinstance(loaded, dict) and "windows" in loaded:
            windows = loaded["windows"]
        elif isinstance(loaded, list):
            windows = loaded
    fractions = copy_fractions(
        args.input_video, args.output_dir / "background_video.mp4", copy_masks_dir
    )
    report["windowed_copy"] = windowed_copy_report(fractions, windows)
    report["glitch"] = glitch_flags(
        args.output_dir / "background_video.mp4",
        masks_dir,
        stride=args.stride,
        max_frames=args.sample_frames,
        dilation_px=args.dilation_px,
    )
    destination = args.output_dir / "inpainting_evaluation.json"
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
