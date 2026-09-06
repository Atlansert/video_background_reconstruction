"""Render a contact sheet (grid of sampled frames) from a video.

Usage (vbr environment):
    python -m tools.video_contact_sheet --video outputs/001_sam31_slam/background_video.mp4 \
        --output outputs/001_sam31_slam/background_video_contact.png --count 12
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--cell-width", type=int, default=480)
    parser.add_argument("--overlay", type=Path, default=None,
                        help="optional second video (e.g. mask overlay) shown under each frame")
    args = parser.parse_args()

    capture = cv2.VideoCapture(str(args.video))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.unique(np.linspace(0, max(total - 1, 0), args.count, dtype=int))
    wanted = set(int(i) for i in indices)

    frames = {}
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if index in wanted:
            frames[index] = frame
        index += 1
    capture.release()

    overlay_frames = {}
    if args.overlay and args.overlay.exists():
        capture = cv2.VideoCapture(str(args.overlay))
        index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if index in wanted:
                overlay_frames[index] = frame
            index += 1
        capture.release()

    scale = args.cell_width / frames[next(iter(frames))].shape[1]
    cells = []
    for i in sorted(frames):
        frame = frames[i]
        height = int(frame.shape[0] * scale)
        cell = cv2.resize(frame, (args.cell_width, height))
        if i in overlay_frames:
            below = cv2.resize(overlay_frames[i], (args.cell_width, height))
            cell = np.vstack([cell, below])
        cv2.putText(cell, str(i), (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 255, 0), 2)
        cells.append(cell)
    if not cells:
        raise SystemExit("No frames sampled")

    columns = 3
    rows = [np.hstack(cells[r:r + columns]) for r in range(0, len(cells), columns)]
    width = max(row.shape[1] for row in rows)
    grid = np.zeros((sum(r.shape[0] for r in rows), width, 3), dtype=np.uint8)
    y = 0
    for row in rows:
        grid[y:y + row.shape[0], : row.shape[1]] = row
        y += row.shape[0]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), grid)
    print(f"wrote {args.output} ({len(cells)} frames from {total})")


if __name__ == "__main__":
    main()
