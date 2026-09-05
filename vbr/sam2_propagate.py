"""Propagate SAM 3.1 masks between adjacent keyframes with SAM2."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np


def _numeric_paths(directory: Path, suffix: str) -> list[Path]:
    return sorted(directory.glob(f"*.{suffix}"), key=lambda path: int(path.stem))


def _clean_mask(mask: np.ndarray, min_area: int) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    clean = np.zeros_like(binary)
    for index in range(1, count):
        if stats[index, cv2.CC_STAT_AREA] >= min_area:
            clean[labels == index] = 255
    return clean


def _mask_from_logits(logits, height, width):
    values = logits.detach().cpu().numpy()
    while values.ndim > 3 and values.shape[1] == 1:
        values = values[:, 0]
    union = np.any(values > 0, axis=0).astype(np.uint8) * 255
    if union.shape != (height, width):
        union = cv2.resize(union, (width, height), interpolation=cv2.INTER_NEAREST)
    return union


def run(args: argparse.Namespace) -> None:
    import torch
    from sam2.build_sam import build_sam2_video_predictor

    frames_dir = Path(args.frames).resolve()
    key_masks_dir = Path(args.key_masks).resolve()
    out_dir = Path(args.output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    frame_paths = _numeric_paths(frames_dir, "jpg")
    if not frame_paths:
        raise RuntimeError(f"No numeric JPEG frames found in {frames_dir}")
    frame_index_by_id = {int(path.stem): index for index, path in enumerate(frame_paths)}
    key_masks = []
    for path in _numeric_paths(key_masks_dir, "png"):
        frame_id = int(path.stem)
        if frame_id not in frame_index_by_id:
            continue
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is not None and np.any(mask):
            key_masks.append((frame_index_by_id[frame_id], frame_id, mask))
    if not key_masks:
        raise RuntimeError("SAM2 propagation has no non-empty SAM 3.1 seed masks")
    if key_masks[0][0] != 0:
        raise RuntimeError("The first video frame must have a non-empty keyframe mask")

    started = time.time()
    predictor = build_sam2_video_predictor(
        args.model_config,
        str(Path(args.checkpoint).resolve()),
        device="cuda",
        apply_postprocessing=True,
    )
    first_image = cv2.imread(str(frame_paths[0]))
    height, width = first_image.shape[:2]
    coverage = np.zeros(len(frame_paths), dtype=float)

    if key_masks[-1][0] != len(frame_paths) - 1:
        key_masks.append((len(frame_paths) - 1, int(frame_paths[-1].stem), key_masks[-1][2]))

    for interval_index, (start, _, seed_mask) in enumerate(key_masks):
        if interval_index + 1 < len(key_masks):
            end = key_masks[interval_index + 1][0]
        else:
            end = start
        if end < start:
            continue
        with tempfile.TemporaryDirectory(prefix="vbr_sam2_chunk_") as temporary:
            chunk_dir = Path(temporary)
            for local_index, source in enumerate(frame_paths[start : end + 1]):
                os.symlink(source, chunk_dir / f"{local_index:06d}.jpg")
            state = predictor.init_state(
                video_path=str(chunk_dir),
                offload_video_to_cpu=True,
                offload_state_to_cpu=args.offload_state,
                async_loading_frames=True,
            )
            predictor.add_new_mask(
                state, frame_idx=0, obj_id=1, mask=seed_mask.astype(bool)
            )
            for local_index, _, logits in predictor.propagate_in_video(state):
                global_index = start + local_index
                if global_index == end and interval_index + 1 < len(key_masks) - 1:
                    continue
                mask = _clean_mask(
                    _mask_from_logits(logits, height, width), args.min_area
                )
                cv2.imwrite(
                    str(out_dir / f"{int(frame_paths[global_index].stem):06d}.png"),
                    mask,
                )
                coverage[global_index] = float(np.count_nonzero(mask)) / mask.size
            exact = _clean_mask(seed_mask, args.min_area)
            cv2.imwrite(
                str(out_dir / f"{int(frame_paths[start].stem):06d}.png"), exact
            )
            coverage[start] = float(np.count_nonzero(exact)) / exact.size
            del state
        if interval_index % 10 == 0:
            torch.cuda.empty_cache()
        print(
            f"SAM2 interval {interval_index + 1}/{len(key_masks)}: "
            f"frames {start}-{end}"
        )

    report = {
        "backend": "sam2.1_hiera_large",
        "mode": "independent_keyframe_intervals",
        "frames": len(frame_paths),
        "seed_frames": len(key_masks),
        "mean_coverage": float(np.mean(coverage)),
        "max_coverage": float(np.max(coverage)),
        "nonempty_frames": int(np.count_nonzero(coverage > 0)),
        "elapsed_seconds": time.time() - started,
    }
    (out_dir / "sam2_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if report["nonempty_frames"] < len(frame_paths) * 0.8:
        raise RuntimeError(
            f"SAM2 produced masks for only {report['nonempty_frames']}/{len(frame_paths)} frames"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", required=True)
    parser.add_argument("--key-masks", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--min-area", type=int, default=100)
    parser.add_argument("--offload-state", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
