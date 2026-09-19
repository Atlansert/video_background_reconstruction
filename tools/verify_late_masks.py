"""Verify the late-section mask repair: object-box coverage + early-section drift.

Usage (vbr-seg):
    python -m tools.verify_late_masks
"""
from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path

ROOT = Path("/data/lzx/video_background_reconstruction/outputs/001_sam31_slam")
NEW = ROOT / "masks_inpaint"
OLD = ROOT / "snapshots" / "masks" / "masks_inpaint_pre_latefix"

# (label, start, end, box xyxy normalized) — same segments as the repair driver
BOXES = [
    ("fridge   1440-1570", 1440, 1570, (0.46, 0.00, 0.86, 0.60)),
    ("fridge   1570-1652", 1570, 1652, (0.05, 0.05, 0.44, 0.64)),
    ("fridge   1652-1799", 1652, 1799, (0.08, 0.02, 0.38, 0.64)),
    ("wardrobe 1560-1799", 1560, 1799, (0.42, 0.00, 0.66, 0.48)),
    ("cabinets 1440-1799", 1440, 1799, (0.14, 0.00, 1.00, 0.45)),
]

def coverage(mask, box):
    h, w = mask.shape
    x1, y1, x2, y2 = int(box[0] * w), int(box[1] * h), int(box[2] * w), int(box[3] * h)
    region = mask[y1:y2, x1:x2] > 127
    return float(np.count_nonzero(region)) / region.size

print(f"{'segment':22s} {'before':>8s} {'after':>8s}")
for label, start, end, box in BOXES:
    vals_old, vals_new = [], []
    for fid in range(start, end + 1, 10):
        old = cv2.imread(str(OLD / f"{fid:06d}.png"), 0)
        new = cv2.imread(str(NEW / f"{fid:06d}.png"), 0)
        if old is not None:
            vals_old.append(coverage(old, box))
        if new is not None:
            vals_new.append(coverage(new, box))
    print(f"{label:22s} {np.mean(vals_old):8.3f} {np.mean(vals_new):8.3f}")

# early-section drift: XOR between pre-repair and repaired masks (0-1400)
xor_fracs = []
for fid in range(0, 1401, 25):
    old = cv2.imread(str(OLD / f"{fid:06d}.png"), 0)
    new = cv2.imread(str(ROOT / "masks_inpaint" / f"{fid:06d}.png"), 0)
    if old is None or new is None:
        continue
    xor_fracs.append(np.count_nonzero((old > 127) != (new > 127)) / old.size)
print(f"\nearly-section (0-1400) mean XOR vs pre-repair: {np.mean(xor_fracs):.4f}")
