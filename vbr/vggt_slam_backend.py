"""Full VGGT-SLAM backend with mask-aware background filtering.

Runs inside the vbr-slam environment (like vbr.vggt_direct) and reuses the
upstream VGGT-SLAM Solver: optical-flow keyframing, per-submap VGGT
inference, SL(4) pose-graph optimization and image-retrieval loop closure.
Results are exported through the same NPZ/PLY interface as vbr.vggt_direct
so the TSDF and geometry stages need no per-backend changes.

Scale handling: every submap's VGGT prediction carries its own monocular
scale. The optimized SL(4) homographies encode the inter-submap scales;
those scales are folded into the depth maps (one factor per submap) so the
exported extrinsics stay rigid and Open3D TSDF can consume them directly.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np


class _Noop:
    """Attribute/call chain that absorbs every viewer interaction."""

    def __getattr__(self, name):
        return _Noop()

    def __call__(self, *args, **kwargs):
        return _Noop()


class _HeadlessViewer(_Noop):
    def __init__(self, *args, **kwargs):
        pass


def _configure_paths(repo: Path) -> None:
    sys.path[:0] = [
        str(repo / "third_party" / "vggt"),
        str(repo),
        str(repo / "third_party" / "dinov2"),
    ]


def _configure_offline_models(repo: Path, args) -> None:
    """Point SALAD/DINOv2 at local checkpoints so loop closure works offline."""
    salad = args.salad_checkpoint or "checkpoints/salad/dino_salad.ckpt"
    salad_path = Path(salad)
    if not salad_path.is_absolute():
        salad_path = Path.cwd() / salad_path
    if salad_path.exists():
        os.environ.setdefault("SALAD_CHECKPOINT", str(salad_path.resolve()))
    dinov2_repo = repo / "third_party" / "dinov2"
    if dinov2_repo.exists():
        os.environ.setdefault("DINOV2_REPO", str(dinov2_repo.resolve()))
    dinov2_ckpt = args.dinov2_checkpoint or "checkpoints/dinov2/dinov2_vitb14_pretrain.pth"
    dinov2_path = Path(dinov2_ckpt)
    if not dinov2_path.is_absolute():
        dinov2_path = Path.cwd() / dinov2_path
    if dinov2_path.exists():
        os.environ.setdefault("DINOV2_CHECKPOINT", str(dinov2_path.resolve()))


def homography_scale(homography: np.ndarray) -> float:
    """Scale factor of an optimized SL(4) cam-to-world homography.

    SL(4) values are projective matrices only defined up to a common factor,
    so they are normalized by the bottom-right entry first. The result is
    expected to be close to a similarity [s*R, t; 0 1] with R orthonormal,
    giving s = det(R-part)^(1/3).
    """
    homography = np.asarray(homography, dtype=np.float64)
    pivot = homography[-1, -1]
    if not np.isfinite(pivot) or abs(pivot) < 1e-12:
        return 1.0
    homography = homography / pivot
    determinant = float(np.linalg.det(homography[:3, :3]))
    if not np.isfinite(determinant) or abs(determinant) < 1e-12:
        return 1.0
    return abs(determinant) ** (1.0 / 3.0)


def rigid_from_similarity(homography: np.ndarray, scale: float) -> np.ndarray:
    """Rigid cam-to-world pose for depth scaled by `scale`.

    If X_world = s * R @ X_cam + t, a depth map multiplied by s unprojects
    through the rigid pose [R, t] into the same world frame.
    """
    homography = np.asarray(homography, dtype=np.float64)
    pivot = homography[-1, -1]
    if np.isfinite(pivot) and abs(pivot) > 1e-12:
        homography = homography / pivot
    rotation_matrix = homography[:3, :3] / max(scale, 1e-12)
    u, _, vt = np.linalg.svd(rotation_matrix)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u = u.copy()
        u[:, -1] *= -1.0
        rotation = u @ vt
    pose = np.eye(4)
    pose[:3, :3] = rotation
    pose[:3, 3] = homography[:3, 3]
    return pose


def mask_in_model_space(
    mask_path: Path, coords: np.ndarray, model_height: int, model_width: int
) -> np.ndarray:
    """Map a full-resolution foreground mask into VGGT model space.

    Supports both preprocessing modes via `coords` = [x1, y1, x2, y2, ...]:
    crop mode fills the whole canvas, square (letterbox) mode occupies the
    content band inside it.
    """
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    canvas = np.zeros((model_height, model_width), dtype=np.uint8)
    if mask is None:
        return canvas
    x1, y1, x2, y2 = np.rint(np.asarray(coords)[:4]).astype(int)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(model_width, x2), min(model_height, y2)
    if x2 <= x1 or y2 <= y1:
        return canvas
    resized = cv2.resize(mask, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
    canvas[y1:y2, x1:x2] = resized
    return canvas


def frame_rgb_in_model_space(
    frame_path: Path, coords: np.ndarray, model_height: int, model_width: int
) -> np.ndarray:
    """Color image for point cloud sampling, mapped like mask_in_model_space."""
    canvas = np.zeros((model_height, model_width, 3), dtype=np.uint8)
    image = cv2.imread(str(frame_path))
    if image is None:
        return canvas
    x1, y1, x2, y2 = np.rint(np.asarray(coords)[:4]).astype(int)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(model_width, x2), min(model_height, y2)
    if x2 <= x1 or y2 <= y1:
        return canvas
    resized = cv2.resize(image, (x2 - x1, y2 - y1), interpolation=cv2.INTER_AREA)
    canvas[y1:y2, x1:x2] = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return canvas


def _model_space_size(model_mode: str, width: int, height: int) -> tuple:
    """(height, width) of VGGT model-space images for the preprocessing mode.

    Mirrors the geometry of vggt.utils.load_fn: square mode letterboxes into
    518x518; crop mode resizes width to 518 and rounds the height to a
    multiple of 14.
    """
    if model_mode == "square":
        return 518, 518
    target = 518
    return round(height * (target / width) / 14) * 14, target


def _model_space_coords(
    model_mode: str, width: int, height: int, model_width: int, model_height: int
) -> list:
    """Content bounds [x1, y1, x2, y2] of the frame inside the model canvas."""
    if model_mode == "square":
        larger = max(width, height)
        scale = model_width / larger
        left = (larger - width) // 2
        top = (larger - height) // 2
        return [
            left * scale,
            top * scale,
            (left + width) * scale,
            (top + height) * scale,
        ]
    return [0.0, 0.0, float(model_width), float(model_height)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", required=True)
    parser.add_argument("--masks", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--salad-checkpoint", default=None)
    parser.add_argument("--dinov2-checkpoint", default=None)
    parser.add_argument("--frame-stride", type=int, default=6)
    parser.add_argument("--max-keyframes", type=int, default=240)
    parser.add_argument("--submap-size", type=int, default=16)
    parser.add_argument("--overlap-size", type=int, default=1)
    parser.add_argument("--min-disparity", type=float, default=50.0)
    parser.add_argument("--conf-threshold", type=float, default=25.0)
    parser.add_argument("--lc-thres", type=float, default=0.95)
    parser.add_argument("--max-loops", type=int, default=1)
    parser.add_argument("--confidence-percentile", type=float, default=35.0)
    parser.add_argument("--pixel-stride", type=int, default=2)
    parser.add_argument("--voxel-size", type=float, default=0.025)
    parser.add_argument(
        "--metric-room-height",
        type=float,
        default=2.6,
        help="Rescale the merged world so its gravity-axis extent (p98-p2 of "
        "the points) equals this many meters; 0 disables. VGGT depth is "
        "relative while every geometry-stage threshold (plane 0.04, TSDF "
        "voxel 0.03, ...) assumes meters, so a ~2.6 m room-height anchor "
        "keeps them meaningful.",
    )
    parser.add_argument(
        "--model-mode",
        choices=["square", "crop"],
        default="square",
        help="square: letterbox to 518x518 (matches vggt_direct); "
        "crop: resize width to 518 keeping aspect (upstream default)",
    )
    parser.add_argument(
        "--mask-aware-matching",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="exclude foreground pixels from submap scale estimation, "
        "submap point filtering and loop-closure retrieval embeddings",
    )
    args = parser.parse_args()

    repo = Path("external/VGGT-SLAM").resolve()
    _configure_paths(repo)
    _configure_offline_models(repo, args)

    import open3d as o3d
    import torch
    from scipy.spatial.transform import Rotation
    from vggt.models.vggt import VGGT
    from vggt.utils.geometry import depth_to_world_coords_points
    import vggt_slam.solver as solver_module

    if args.model_mode == "square":
        from vggt.utils.load_fn import load_and_preprocess_images_square

        solver_module.load_and_preprocess_images = (
            lambda image_names: load_and_preprocess_images_square(
                list(image_names), 518
            )[0]
        )
    solver_module.Viewer = _HeadlessViewer
    from vggt_slam.solver import Solver

    paths = sorted(Path(args.frames).glob("*.jpg"), key=lambda path: int(path.stem))
    if not paths:
        raise RuntimeError("No numeric JPEG frames found")

    first_image = cv2.imread(str(paths[0]))
    if first_image is None:
        raise RuntimeError(f"Cannot read frame {paths[0]}")
    original_height, original_width = first_image.shape[:2]

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

    solver = Solver(init_conf_threshold=args.conf_threshold, lc_thres=args.lc_thres)

    # Model-space geometry shared by masking, retrieval and the NPZ export.
    model_height, model_width = _model_space_size(
        args.model_mode, original_width, original_height
    )
    base_coords = _model_space_coords(
        args.model_mode, original_width, original_height, model_width, model_height
    )
    masks_root = Path(args.masks)

    def model_space_mask(frame_name: str) -> np.ndarray:
        frame_id = Path(frame_name).stem
        return mask_in_model_space(
            masks_root / f"{frame_id}.png",
            base_coords,
            model_height,
            model_width,
        )

    if args.mask_aware_matching:
        # Blank foreground pixels before SALAD retrieval so loop-closure
        # similarity is driven by the room, not by furniture that moved.
        original_retrieval = solver.image_retrieval.get_all_submap_embeddings

        def foreground_free_retrieval(submap):
            frames = submap.get_all_frames().clone()
            for index, name in enumerate(submap.img_names):
                mask = torch.from_numpy(model_space_mask(name) > 0).to(frames.device)
                frames[index].masked_fill_(mask, 0.5)
            return solver.image_retrieval.get_batch_descriptors(frames)

        solver.image_retrieval.get_all_submap_embeddings = foreground_free_retrieval

    # Capture the raw per-submap VGGT outputs (depth/confidence/intrinsics)
    # before they are consumed by add_points; Submap does not retain depth.
    stashes = []
    original_add_points = solver.add_points

    def stash_and_add_points(pred_dict):
        working = solver.current_working_submap
        stashes.append(
            {
                "submap_id": int(working.get_id()),
                "frame_names": [str(name) for name in working.img_names],
                "depth": np.asarray(pred_dict["depth"], dtype=np.float32),
                "depth_conf": np.asarray(pred_dict["depth_conf"], dtype=np.float32),
                "intrinsic": np.asarray(pred_dict["intrinsic"], dtype=np.float64),
            }
        )
        if args.mask_aware_matching:
            # Zero foreground confidence before the solver stores it: the
            # submap confidence threshold, the inter-submap scale estimation
            # (good_mask in Solver.add_edge) and the submap point filtering
            # then all ignore foreground pixels. The stash above keeps the
            # raw confidence for the NPZ/TSDF interface, which applies the
            # foreground masks itself.
            masks = np.stack(
                [model_space_mask(name) > 0 for name in working.img_names]
            )
            if masks.shape == pred_dict["depth_conf"].shape:
                pred_dict["depth_conf"] = np.where(
                    masks, np.float32(0.0), pred_dict["depth_conf"]
                ).astype(np.float32)
            else:
                print(
                    f"mask-aware matching skipped for submap {working.get_id()}: "
                    f"mask shape {masks.shape} != conf shape "
                    f"{pred_dict['depth_conf'].shape}"
                )
        return original_add_points(pred_dict)

    solver.add_points = stash_and_add_points

    # Phase 1: optical-flow keyframe selection over the strided candidates.
    stride = max(1, args.frame_stride)
    candidates = paths[::stride]
    if candidates[-1] != paths[-1]:
        candidates.append(paths[-1])
    keyframes = []
    for path in candidates:
        image = cv2.imread(str(path))
        if image is None:
            raise RuntimeError(f"Cannot read frame {path}")
        if solver.flow_tracker.compute_disparity(image, args.min_disparity):
            keyframes.append(path)
    keyframes_before_cap = len(keyframes)
    if len(keyframes) > args.max_keyframes:
        keep = np.unique(
            np.linspace(0, len(keyframes) - 1, args.max_keyframes, dtype=int)
        )
        keyframes = [keyframes[index] for index in keep]
    print(
        f"VGGT-SLAM keyframing: {len(candidates)} candidates, "
        f"{keyframes_before_cap} keyframes, {len(keyframes)} after cap"
    )

    # Phase 2: submap processing (VGGT -> pose graph -> optimize), mirroring
    # external/VGGT-SLAM/main.py.
    overlap = max(1, min(args.overlap_size, args.submap_size))
    batch = []
    processed_submaps = 0
    started = time.time()

    def process_submap(frame_names):
        nonlocal processed_submaps
        print(f"Submap {processed_submaps}: {len(frame_names)} frames")
        predictions = solver.run_predictions(frame_names, model, args.max_loops, None, None)
        solver.add_points(predictions)
        solver.graph.optimize()
        processed_submaps += 1

    for index, path in enumerate(keyframes):
        batch.append(str(path.resolve()))
        is_last = index == len(keyframes) - 1
        if len(batch) == args.submap_size + overlap or is_last:
            if len(batch) <= overlap:
                break
            process_submap(batch)
            batch = batch[-overlap:]

    elapsed = time.time() - started
    if not stashes:
        raise RuntimeError("VGGT-SLAM produced no submaps")

    # Phase 3: export optimized poses, scaled depth and masked background
    # points in the vggt_direct NPZ interface.
    stash_by_id = {stash["submap_id"]: stash for stash in stashes}
    extrinsics_parts = []
    intrinsics_parts = []
    depth_parts = []
    confidence_parts = []
    frame_path_parts = []
    frame_id_parts = []
    mask_parts = []
    coords_parts = []
    submap_scales = {}
    exported_frame_ids = set()

    stash_height = int(stashes[0]["depth"].shape[1])
    stash_width = int(stashes[0]["depth"].shape[2])
    if (stash_height, stash_width) != (model_height, model_width):
        raise RuntimeError(
            f"Model space mismatch: predictions are {stash_height}x{stash_width}, "
            f"expected {model_height}x{model_width} for {args.model_mode} mode"
        )

    for submap in solver.map.ordered_submaps_by_key():
        if submap.get_lc_status():
            continue
        submap_id = int(submap.get_id())
        stash = stash_by_id.get(submap_id)
        if stash is None:
            raise RuntimeError(f"No captured predictions for submap {submap_id}")
        homographies = [
            solver.graph.get_homography(submap_id + index)
            for index in range(len(submap.poses))
        ]
        scale = homography_scale(homographies[0])
        submap_scales[str(submap_id)] = scale
        frame_ids = [int(round(float(value))) for value in submap.get_frame_ids()]
        if len(frame_ids) != len(stash["frame_names"]):
            raise RuntimeError(
                f"Submap {submap_id}: {len(frame_ids)} frame ids vs "
                f"{len(stash['frame_names'])} captured frames"
            )
        for index, homography in enumerate(homographies):
            frame_id = frame_ids[index]
            if frame_id in exported_frame_ids:
                # Overlap frames close consecutive submaps; export each once.
                continue
            exported_frame_ids.add(frame_id)
            frame_name = stash["frame_names"][index]
            cam_to_world = rigid_from_similarity(homography, scale)
            world_to_cam = np.linalg.inv(cam_to_world)
            extrinsics_parts.append(world_to_cam[:3])
            intrinsics_parts.append(stash["intrinsic"][index])
            depth_parts.append(stash["depth"][index] * np.float32(scale))
            confidence_parts.append(stash["depth_conf"][index])
            frame_path_parts.append(frame_name)
            frame_id_parts.append(frame_id)
            mask_parts.append(
                mask_in_model_space(
                    masks_root / f"{Path(frame_name).stem}.png",
                    base_coords,
                    model_height,
                    model_width,
                )
            )
            coords_parts.append(
                [*base_coords, float(original_width), float(original_height)]
            )

    extrinsics = np.stack(extrinsics_parts)
    intrinsics = np.stack(intrinsics_parts)
    depth = np.stack(depth_parts).astype(np.float32)
    confidence = np.stack(confidence_parts).astype(np.float32)
    original_coords = np.asarray(coords_parts, dtype=np.float64)
    foreground_masks = np.stack(mask_parts).astype(bool)
    confidence_cutoff = float(np.percentile(confidence, args.confidence_percentile))

    points_parts = []
    color_parts = []
    raw_background_points = 0
    rejected_foreground_points = 0
    mask_coverages = []
    for index, frame_path in enumerate(frame_path_parts):
        world_xyz, _, valid_depth = depth_to_world_coords_points(
            depth[index, ..., 0], extrinsics[index], intrinsics[index]
        )
        mask = foreground_masks[index]
        mask_coverages.append(float(np.count_nonzero(mask)) / mask.size)
        confident = confidence[index] >= confidence_cutoff
        valid = valid_depth & confident & np.isfinite(world_xyz).all(axis=-1)
        rejected_foreground_points += int(np.count_nonzero(valid & mask))
        background = valid & (mask == 0)
        raw_background_points += int(np.count_nonzero(background))
        flat_indices = np.flatnonzero(background)[:: max(1, args.pixel_stride)]
        colors = frame_rgb_in_model_space(
            frame_path, base_coords, model_height, model_width
        )
        points_parts.append(world_xyz.reshape(-1, 3)[flat_indices])
        color_parts.append(colors.reshape(-1, 3)[flat_indices])

    points = np.concatenate(points_parts, axis=0)
    colors = np.concatenate(color_parts, axis=0)

    # VGGT depth is relative per submap; the merged world ends up at an
    # arbitrary scale (observed room heights 0.33-1.13 units for a ~2.6 m
    # room). Anchor it to meters via the gravity-axis extent so the geometry
    # stage's metric thresholds apply. Scaling points, depths and camera
    # translations by one factor keeps depth/extrinsics consistent.
    scale_factor = 1.0
    world_height = None
    if args.metric_room_height > 0 and len(points):
        from vbr.geometry import estimate_gravity

        gravity = estimate_gravity(extrinsics)
        low, high = np.percentile(points @ gravity, [2.0, 98.0])
        world_height = float(high - low)
        if world_height > 1e-6:
            scale_factor = float(args.metric_room_height) / world_height
            points = points * scale_factor
            depth *= np.float32(scale_factor)
            extrinsics = extrinsics.copy()
            extrinsics[:, :3, 3] *= scale_factor

    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points.astype(np.float64)))
    cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)
    if args.voxel_size > 0:
        # voxel_size <= 0 keeps the official dense output (no downsample,
        # no statistical outlier removal) for the geometry stage.
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
        depth=depth,
        confidence=confidence,
        confidence_cutoff=np.float32(confidence_cutoff),
        frame_paths=np.asarray([str(path) for path in frame_path_parts]),
        frame_ids=np.asarray(frame_id_parts, dtype=np.int32),
        original_coords=original_coords,
        foreground_masks=foreground_masks,
    )

    trajectory = output.parent / "trajectory_tum.txt"
    with trajectory.open("w", encoding="utf-8") as handle:
        for index in range(len(frame_id_parts)):
            world_to_cam = np.eye(4)
            world_to_cam[:3] = extrinsics[index]
            cam_to_world = np.linalg.inv(world_to_cam)
            translation = cam_to_world[:3, 3]
            quaternion = Rotation.from_matrix(cam_to_world[:3, :3]).as_quat()
            handle.write(
                f"{frame_id_parts[index]} {translation[0]:.8f} {translation[1]:.8f} "
                f"{translation[2]:.8f} {quaternion[0]:.8f} {quaternion[1]:.8f} "
                f"{quaternion[2]:.8f} {quaternion[3]:.8f}\n"
            )

    report = {
        "backend": "vggt_slam_masked",
        "selected_frames": len(frame_id_parts),
        "candidate_frames": len(candidates),
        "keyframes_after_flow": keyframes_before_cap,
        "keyframes_after_cap": len(keyframes),
        "submap_count": len(submap_scales),
        "loop_closures": int(solver.graph.get_num_loops()),
        "mask_aware_matching": bool(args.mask_aware_matching),
        "model_mode": args.model_mode,
        "submap_scales": submap_scales,
        "frame_ids": frame_id_parts,
        "confidence_percentile": args.confidence_percentile,
        "confidence_cutoff": confidence_cutoff,
        "mean_mask_coverage_model_space": float(np.mean(mask_coverages)),
        "foreground_points_rejected": rejected_foreground_points,
        "raw_background_points": raw_background_points,
        "output_points": len(cloud.points),
        "metric_scale_factor": scale_factor,
        "world_height_before_metric": world_height,
        "model_space": [model_height, model_width],
        "pointcloud": str(output),
        "reconstruction": str(output.with_suffix(".npz")),
        "trajectory": str(trajectory),
        "slam_elapsed_seconds": elapsed,
    }
    output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    mask_has_content = bool(np.any(foreground_masks))
    if rejected_foreground_points == 0 and mask_has_content:
        raise RuntimeError("No foreground pixels reached VGGT-SLAM point filtering")
    print(
        f"Masked VGGT-SLAM reconstruction: {len(submap_scales)} submaps, "
        f"{solver.graph.get_num_loops()} loop closures, rejected "
        f"{rejected_foreground_points} foreground samples, wrote "
        f"{len(cloud.points)} points to {output}"
    )


if __name__ == "__main__":
    main()
