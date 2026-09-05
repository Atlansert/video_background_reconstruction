"""Masked dense reconstruction with the VGGT model used by VGGT-SLAM."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


def _mask_in_model_space(mask_path: Path, coords: np.ndarray, size: int) -> np.ndarray:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    canvas = np.zeros((size, size), dtype=np.uint8)
    if mask is None:
        return canvas
    x1, y1, x2, y2 = np.rint(coords[:4]).astype(int)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(size, x2), min(size, y2)
    if x2 <= x1 or y2 <= y1:
        return canvas
    resized = cv2.resize(mask, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
    canvas[y1:y2, x1:x2] = resized
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", required=True)
    parser.add_argument("--masks", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--model-size", type=int, default=518)
    parser.add_argument("--confidence-percentile", type=float, default=35.0)
    parser.add_argument("--pixel-stride", type=int, default=2)
    parser.add_argument("--voxel-size", type=float, default=0.025)
    args = parser.parse_args()

    repo = Path("external/VGGT-SLAM").resolve()
    sys.path[:0] = [str(repo / "third_party" / "vggt"), str(repo)]
    import open3d as o3d
    import torch
    from vggt.models.vggt import VGGT
    from vggt.utils.geometry import depth_to_world_coords_points
    from vggt.utils.load_fn import load_and_preprocess_images_square
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    paths = sorted(Path(args.frames).glob("*.jpg"), key=lambda path: int(path.stem))
    if not paths:
        raise RuntimeError("No numeric JPEG frames found")
    selected = np.linspace(0, len(paths) - 1, min(args.max_frames, len(paths)), dtype=int)
    paths = [paths[index] for index in np.unique(selected)]

    device = "cuda"
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    model = VGGT()
    state = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    model.eval().to(dtype=dtype, device=device)

    images, original_coords = load_and_preprocess_images_square(
        [str(path) for path in paths], args.model_size
    )
    images = images.to(device)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=dtype):
        tokens, patch_start_idx, _, _ = model.aggregator(images[None])
        pose_encoding = model.camera_head(tokens)[-1]
        depth, confidence = model.depth_head(tokens, images[None], patch_start_idx)
        extrinsics, intrinsics = pose_encoding_to_extri_intri(
            pose_encoding, images.shape[-2:]
        )

    extrinsics, intrinsics, depth, confidence = [
        value.squeeze(0).float().cpu().numpy()
        for value in (extrinsics, intrinsics, depth, confidence)
    ]
    original_coords_np = original_coords.cpu().numpy()
    colors_model = (
        images.float().cpu().numpy().transpose(0, 2, 3, 1).clip(0, 1) * 255
    ).astype(np.uint8)

    confidence_cutoff = float(np.percentile(confidence, args.confidence_percentile))
    points_parts = []
    color_parts = []
    raw_background_points = 0
    rejected_foreground_points = 0
    mask_coverages = []
    model_masks = []
    for index, frame_path in enumerate(paths):
        world_xyz, _, valid_depth = depth_to_world_coords_points(
            depth[index, ..., 0], extrinsics[index], intrinsics[index]
        )
        mask = _mask_in_model_space(
            Path(args.masks) / f"{frame_path.stem}.png",
            original_coords_np[index],
            args.model_size,
        )
        model_masks.append(mask > 0)
        mask_coverages.append(float(np.count_nonzero(mask)) / mask.size)
        confident = confidence[index] >= confidence_cutoff
        valid = valid_depth & confident & np.isfinite(world_xyz).all(axis=-1)
        rejected_foreground_points += int(np.count_nonzero(valid & (mask > 0)))
        background = valid & (mask == 0)
        raw_background_points += int(np.count_nonzero(background))
        flat_indices = np.flatnonzero(background)[:: max(1, args.pixel_stride)]
        points_parts.append(world_xyz.reshape(-1, 3)[flat_indices])
        color_parts.append(colors_model[index].reshape(-1, 3)[flat_indices])

    points = np.concatenate(points_parts, axis=0)
    colors = np.concatenate(color_parts, axis=0)
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points.astype(np.float64)))
    cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)
    cloud = cloud.voxel_down_sample(args.voxel_size)
    if len(cloud.points) >= 1000:
        cloud, _ = cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.5)

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_point_cloud(str(output), cloud):
        raise RuntimeError(f"Failed to write point cloud {output}")
    np.savez_compressed(
        output.with_suffix(".npz"),
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        depth=depth.astype(np.float32),
        confidence=confidence.astype(np.float32),
        confidence_cutoff=np.float32(confidence_cutoff),
        frame_paths=np.asarray([str(path.resolve()) for path in paths]),
        frame_ids=np.asarray([int(path.stem) for path in paths], dtype=np.int32),
        original_coords=original_coords_np,
        foreground_masks=np.asarray(model_masks, dtype=bool),
    )
    report = {
        "backend": "vggt_direct_masked",
        "selected_frames": len(paths),
        "frame_ids": [int(path.stem) for path in paths],
        "confidence_percentile": args.confidence_percentile,
        "confidence_cutoff": confidence_cutoff,
        "mean_mask_coverage_model_space": float(np.mean(mask_coverages)),
        "foreground_points_rejected": rejected_foreground_points,
        "raw_background_points": raw_background_points,
        "output_points": len(cloud.points),
        "pointcloud": str(output),
        "reconstruction": str(output.with_suffix(".npz")),
    }
    output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if rejected_foreground_points == 0:
        raise RuntimeError("No foreground pixels reached VGGT point filtering")
    print(
        f"Masked VGGT reconstruction: rejected {rejected_foreground_points} foreground "
        f"samples, wrote {len(cloud.points)} points to {output}"
    )


if __name__ == "__main__":
    main()
