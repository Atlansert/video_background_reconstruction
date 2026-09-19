"""Clip the repaired masks down to 'pre-repair baseline + missed objects only'.

The round-4/R5 repair (full prompt suite with preserve disabled) correctly
added the fridge / wardrobe / glass panels, but the same run also masked
walls, ceiling strips and doors inside the wide boxes. With those additions
the late-section fill grew to ~0.74 of the frame and SVOR's generation
collapsed into grey/black mush plus hallucinated furniture (measured on the
2026-09-19 re-run). The pre-repair masks produced a clean video.

This tool rebuilds the masks as:

    v3 = pre_repair ∪ (repaired ∩ object_boxes)

so only additions INSIDE per-segment object boxes survive. No inference is
re-run; the object coverage already propagated to every frame by the repair
is kept wherever it falls inside a box.

Usage (vbr-seg):
    python -m tools.clip_repair_masks            # writes masks/ + masks_inpaint/
    python -m tools.clip_repair_masks --dry-run  # coverage report only
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path("/data/lzx/video_background_reconstruction/outputs/001_sam31_slam")
PRE = ROOT / "snapshots" / "masks" / "masks_inpaint_pre_latefix"  # baseline source masks
# NOTE: the baseline above is the inpaint variant; the plain masks baseline is
# masks_pre_latefix. Both are restored below from their own snapshot dirs.

# Per-segment object boxes (normalized xyxy, camera-pan aware). Additions are
# kept only inside these regions.
SEGMENTS = [
    # --- mid section 630-1030 (first kitchen pass) ---
    (630, 730, (0.42, 0.00, 0.70, 0.50)),
    (730, 780, (0.48, 0.00, 0.76, 0.48)),
    (780, 830, (0.24, 0.00, 0.58, 0.95)),
    (830, 880, (0.10, 0.00, 0.46, 1.00)),
    (880, 1000, (0.00, 0.00, 0.20, 1.00)),
    (630, 730, (0.58, 0.00, 1.00, 0.16)),
    (730, 780, (0.56, 0.00, 1.00, 0.16)),
    (780, 830, (0.46, 0.00, 1.00, 0.28)),
    (830, 880, (0.32, 0.00, 1.00, 0.35)),
    (880, 980, (0.15, 0.00, 0.95, 0.45)),
    (980, 1030, (0.00, 0.00, 0.90, 0.50)),
    # --- late section 1440-1799 (second kitchen pass) ---
    (1440, 1515, (0.58, 0.00, 0.84, 0.46)),
    (1515, 1542, (0.50, 0.00, 0.74, 0.48)),
    (1542, 1572, (0.44, 0.00, 0.58, 0.50)),
    (1570, 1652, (0.05, 0.05, 0.44, 0.64)),
    (1652, 1799, (0.08, 0.02, 0.38, 0.64)),
    (1560, 1595, (0.58, 0.00, 0.80, 0.42)),
    (1595, 1652, (0.42, 0.00, 0.66, 0.48)),
    (1652, 1799, (0.44, 0.00, 0.62, 0.48)),
    (1440, 1570, (0.58, 0.00, 1.00, 0.22)),
    (1570, 1652, (0.15, 0.00, 0.68, 0.22)),
    (1652, 1799, (0.15, 0.00, 0.55, 0.25)),
]

INPAINT_PARAMS = dict(close_px=9, temporal_radius=2, dilate_px=1,
                      hull_min_area=0, hull_max_extra=0.35, overlap=0.12)


def box_mask(shape, box):
    h, w = shape
    m = np.zeros(shape, np.uint8)
    x1, y1, x2, y2 = box
    m[int(y1 * h):int(y2 * h), int(x1 * w):int(x2 * w)] = 255
    return m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    pre_dir = ROOT / "snapshots" / "masks" / "masks_pre_latefix"
    repaired_dir = ROOT / "snapshots" / "masks" / "masks_post_repair_full"
    current = ROOT / "masks"
    if not repaired_dir.exists():
        repaired_dir.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copytree(current, repaired_dir)  # keep the full-repair state

    out_dir = ROOT / "masks_clip_tmp"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = []
    for frame in range(1799):
        pre = cv2.imread(str(pre_dir / f"{frame:06d}.png"), 0)
        rep = cv2.imread(str(repaired_dir / f"{frame:06d}.png"), 0)
        if pre is None or rep is None:
            raise RuntimeError(f"missing mask frame {frame}")
        region = np.zeros_like(pre)
        for start, end, box in SEGMENTS:
            if start <= frame <= end:
                region |= box_mask(pre.shape, box)
        merged = np.where(region > 0, np.maximum(pre, rep), pre).astype(np.uint8)
        merged[merged > 127] = 255
        if not args.dry_run:
            cv2.imwrite(str(out_dir / f"{frame:06d}.png"), merged)
        if frame % 50 == 0:
            report.append({
                "frame": frame,
                "pre": round(float(np.count_nonzero(pre > 127)) / pre.size, 3),
                "full_repair": round(float(np.count_nonzero(rep > 127)) / rep.size, 3),
                "clipped": round(float(np.count_nonzero(merged > 127)) / merged.size, 3),
            })
    print(json.dumps(report, indent=1))

    if args.dry_run:
        import shutil
        shutil.rmtree(out_dir, ignore_errors=True)
        return

    # promote: current masks -> snapshots, clipped -> masks
    import shutil
    final_backup = ROOT / "snapshots" / "masks" / "masks_pre_clip_final"
    if not final_backup.exists():
        shutil.copytree(current, final_backup)
    shutil.rmtree(current)
    out_dir.rename(current)

    from vbr.video import prepare_inpainting_masks
    prepare_inpainting_masks(current, ROOT / "masks_inpaint", **INPAINT_PARAMS)
    print("masks and masks_inpaint rebuilt")


if __name__ == "__main__":
    main()
