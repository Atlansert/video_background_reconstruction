"""Grid-search ProPainter parameters on a short clip.

Extracts a contiguous frame range with its inpaint masks, re-indexes them,
runs ProPainter with variant settings, and scores each variant on
copy-through and in-mask temporal flicker against the original clip.

Usage (vbr environment):
    python -m tools.search_propainter_params --output-dir outputs/001_sam31_slam \
        --start 760 --end 990 --config configs/vggt_slam.yaml
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


def _write_clip(output_dir: Path, start: int, end: int):
    frames_dir = output_dir / "frames_all"
    masks_dir = output_dir / "masks_inpaint"
    if not frames_dir.exists() or not masks_dir.exists():
        raise FileNotFoundError("frames_all and masks_inpaint must exist first")
    clip = output_dir / "clip_search"
    mask_clip = output_dir / "masks_inpaint_clip"
    shutil.rmtree(clip, ignore_errors=True)
    shutil.rmtree(mask_clip, ignore_errors=True)
    clip.mkdir(parents=True, exist_ok=True)
    mask_clip.mkdir(parents=True, exist_ok=True)
    for local, frame_id in enumerate(range(start, end + 1)):
        source = frames_dir / f"{frame_id:06d}.jpg"
        if not source.exists():
            continue
        shutil.copy(source, clip / f"{local:06d}.jpg")
        mask = masks_dir / f"{frame_id:06d}.png"
        if mask.exists():
            shutil.copy(mask, mask_clip / f"{local:06d}.png")


def _read_range(path: Path, start: int, count: int):
    capture = cv2.VideoCapture(str(path))
    frames = []
    capture.set(cv2.CAP_PROP_POS_FRAMES, start)
    for _ in range(count):
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    return frames


def score_variant(
    clip_frames: Path,
    masks_dir: Path,
    project_root: Path,
    cfg: dict,
    variant: dict,
    start: int,
    count: int,
    original_video: Path,
):
    adapter = ProPainterAdapter({**cfg, **variant}, project_root)
    report = adapter.run(
        original_video,
        clip_frames,
        masks_dir,
        project_root / "outputs" / "_clip_search.mp4",
        fps=30,
    )
    (project_root / "outputs" / "_clip_search.mp4").unlink(missing_ok=True)
    (project_root / "outputs" / "_clip_search.json").unlink(missing_ok=True)
    raw = Path(report["raw_output"])
    rebuilt = _read_range(raw, 0, count)
    originals = _read_range(original_video, start, count)
    masks = []
    for local in range(count):
        mask = cv2.imread(str(masks_dir / f"{local:06d}.png"), cv2.IMREAD_GRAYSCALE)
        masks.append(mask if mask is not None else np.zeros_like(originals[0][:, :, 0]))
    scores = []
    for local in range(1, count - 1):
        mask = masks[local] if masks[local] is not None else None
        if mask is None or not np.any(mask):
            continue
        inner = cv2.erode((mask > 0).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        if not inner.any():
            continue
        diff = cv2.absdiff(originals[local], rebuilt[local]).astype(np.float32).mean(axis=2)
        copy_fraction = float((diff[inner] < 11).mean())
        temporal = cv2.absdiff(rebuilt[local - 1], rebuilt[local]).astype(np.float32).mean(axis=2)
        dilated = cv2.dilate(mask, np.ones((7, 7), np.uint8)) > 0
        inside = float(temporal[dilated].mean())
        outside = float(temporal[~dilated].mean()) if (~dilated).sum() > 100 else 1.0
        scores.append((copy_fraction, inside / max(outside, 1e-6)))
    if not scores:
        return {"variant": variant, "clip_frames": 0}
    copies = [item[0] for item in scores]
    ratios = [item[1] for item in scores]
    return {
        "variant": variant,
        "clip_frames": len(scores),
        "copy_mean": float(np.mean(copies)),
        "copy_over_0.2": int(np.count_nonzero(np.asarray(copies) > 0.2)),
        "flicker_ratio_mean": float(np.mean(ratios)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--config", default="configs/vggt_slam.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    root = args.output_dir.resolve()
    project_root = root.parents[1].resolve()
    video_path = project_root / cfg.get("input_video", "video/001.mp4")

    _write_clip(root, args.start, args.end)
    count = args.end - args.start + 1

    grid = [
        {"neighbor_length": 20, "ref_stride": 5, "subvideo_length": 80, "raft_iter": 20},
        {"neighbor_length": 40, "ref_stride": 10, "subvideo_length": 80, "raft_iter": 20},
        {"neighbor_length": 80, "ref_stride": 10, "subvideo_length": 80, "raft_iter": 20},
        {"neighbor_length": 20, "ref_stride": 5, "subvideo_length": 240, "raft_iter": 20},
        {"neighbor_length": 40, "ref_stride": 10, "subvideo_length": 240, "raft_iter": 20},
    ]

    results = []
    for variant in grid:
        result = score_variant(
            root / "clip_search",
            root / "masks_inpaint_clip",
            project_root,
            cfg["video_completion"],
            variant,
            args.start,
            count,
            video_path,
        )
        results.append(result)
        print(json.dumps(result))

    (root / "proppainter_param_search.json").write_text(
        json.dumps({"start": args.start, "end": args.end, "results": results}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()