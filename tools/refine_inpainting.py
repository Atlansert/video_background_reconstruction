"""Two-pass ProPainter: amplify copy-through regions back into the mask.

Pass 1 produces the standard inpainted video. Residual feedback then finds
connected regions inside the inpaint mask where the output is still nearly
identical to the original (object copied through), unions them a few frames
forward/backward, dilates slightly, and re-runs ProPainter so its flow module
cannot reuse the original object.

Usage (vbr environment):
    python -m tools.refine_inpainting --output-dir outputs/001_sam31_slam \
        --config configs/vggt_slam.yaml --max-rounds 2
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from vbr.config import load_config
from vbr.models.inpainting import ProPainterAdapter
from vbr.video import video_info


def read_frames(path: Path):
    capture = cv2.VideoCapture(str(path))
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        yield frame
    capture.release()


def find_copy_through(
    original_video: Path,
    background_video: Path,
    masks_dir: Path,
    stride: int = 5,
    threshold: float = 11.0,
    min_area: int = 1200,
    max_frames: int = 0,
):
    """Sample frames and return {video_index: residual_bool}.

    Copy-through is a connected component inside the masked region whose
    inpaint output is still nearly identical to the original frame.
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
    copies = {}
    for index, frame_id in sampled_ids.items():
        if index not in originals or index not in backgrounds:
            continue
        mask = cv2.imread(str(masks_dir / f"{frame_id:06d}.png"), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        inner = cv2.erode((mask > 0).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        diff = cv2.absdiff(originals[index], backgrounds[index]).astype(np.float32).mean(axis=2)
        residual = (diff < threshold) & inner
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            residual.astype(np.uint8), connectivity=8
        )
        keep = np.zeros_like(residual)
        for component in range(1, count):
            if int(stats[component, cv2.CC_STAT_AREA]) >= min_area:
                keep |= labels == component
        if keep.any():
            copies[index] = keep
    return copies


def expand_copy_through(copies, masks_dir, temporal_radius=1, dilate_px=7):
    """Merge copy-through regions into the inpaint masks for a second pass."""
    paths = sorted(masks_dir.glob("*.png"), key=lambda p: int(p.stem))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (max(1, int(dilate_px)), max(1, int(dilate_px)))
    )
    unified = {index: residual.copy() for index, residual in copies.items()}
    for index in list(copies):
        for offset in range(1, temporal_radius + 1):
            for neighbor in (index + offset, index - offset):
                other = unified.get(neighbor)
                if other is not None and other.shape == unified[index].shape:
                    unified[index] = unified[index] | other
    changed_pixels = 0
    for index, residual in unified.items():
        frame_id = int(paths[index].stem)
        mask = cv2.imread(str(masks_dir / f"{frame_id:06d}.png"), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        region = cv2.dilate(residual.astype(np.uint8), kernel) > 0
        combined = (mask > 0) | region
        changed_pixels += int(np.count_nonzero(combined != (mask > 0)))
        cv2.imwrite(str(masks_dir / f"{frame_id:06d}.png"), combined.astype(np.uint8) * 255)
    return changed_pixels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--config", default="configs/vggt_slam.yaml")
    parser.add_argument("--max-rounds", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=11.0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    root = args.output_dir.resolve()
    project_root = root.parents[1].resolve()
    video_path = project_root / cfg.get("input_video", "video/001.mp4")
    masks_inpaint = root / "masks_inpaint"
    background_video = root / "background_video.mp4"
    info = video_info(video_path)

    shutil.copytree(masks_inpaint, root / "snapshots" / "masks" / "masks_inpaint_pass1_backup", dirs_exist_ok=True)
    adapter = ProPainterAdapter(cfg["video_completion"], project_root)
    rounds_done = 0
    for round_index in range(1, args.max_rounds + 1):
        copies = find_copy_through(
            video_path, background_video, masks_inpaint, threshold=args.threshold
        )
        if not copies:
            print(f"round {round_index}: no copy-through found")
            break
        changed = expand_copy_through(copies, masks_inpaint)
        print(f"round {round_index}: {len(copies)} frames, {changed} pixels added")
        if changed == 0:
            break
        adapter.run(
            video_path,
            root / "frames_all",
            masks_inpaint,
            background_video,
            info["fps"],
        )
        rounds_done = round_index
        (root / "inpaint_refine_report.json").write_text(
            json.dumps(
                {
                    "rounds": rounds_done,
                    "sample_indices": sorted(copies),
                    "changed_pixels": changed,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    print(f"done: {background_video}")


if __name__ == "__main__":
    main()