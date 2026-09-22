"""Structural-prior geometry rebuild from the ORIGINAL frames (branch A).

Rebuilds the room geometry without any generative video in the loop:

1. Masked VGGT-SLAM on the original frames with the current foreground
   masks: poses, per-frame depth and background points in one metric-anchored
   world (the same SLAM configuration the production reconstruction stage
   uses). No diffusion-generated pixels enter the geometry.
2. The geometry stage then completes occluded regions with structural
   priors: gravity-aligned floor/ceiling planes, wall lines fitted in the
   plan view, and corner closure between adjacent walls. Regions the camera
   never observed behind furniture stay filled by these priors instead of
   hallucinated texture.
3. Masked depth is fused by TSDF (same settings as production) and merged
   with the prior mesh.

Output is written to a separate directory so the production artifacts in
``outputs/001_sam31_slam`` are not touched.

Usage (vbr environment):
    python -m tools.rebuild_geometry_prior \
        --out outputs/geometry_prior_a --config configs/vggt_slam.yaml
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from tools.rebuild_geometry_from_background import diagnose_submap_scales
from vbr.cli import PROJECT_ROOT, _load_opening_hints
from vbr.config import load_config
from vbr.geometry import build_mesh, save_pointcloud
from vbr.interactive import write_html
from vbr.models.slam import SLAMAdapter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path,
                        default=PROJECT_ROOT / "outputs/001_sam31_slam",
                        help="pipeline run directory holding frames_all/ and masks/")
    parser.add_argument("--out", type=Path,
                        default=PROJECT_ROOT / "outputs/geometry_prior_a")
    parser.add_argument("--config", default="configs/vggt_slam.yaml")
    parser.add_argument("--reuse-slam", action="store_true",
                        help="reuse an existing <out>/slam instead of re-running "
                             "masked VGGT-SLAM (only for re-checking the geometry stage)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_dir = (PROJECT_ROOT / args.run_dir).resolve() if not args.run_dir.is_absolute() else args.run_dir
    out_dir = (PROJECT_ROOT / args.out).resolve() if not args.out.is_absolute() else args.out
    frames_dir = run_dir / "frames_all"
    masks_dir = run_dir / "masks"
    for path in (frames_dir, masks_dir):
        if not path.exists():
            raise RuntimeError(f"Missing {path}; expected a completed pipeline run")
    out_dir.mkdir(parents=True, exist_ok=True)

    frame_count = len(list(frames_dir.glob("*.jpg")))
    mask_count = len(list(masks_dir.glob("*.png")))
    if frame_count != mask_count:
        raise RuntimeError(
            f"Frame/mask count mismatch: {frame_count} frames vs {mask_count} masks"
        )
    sample = cv2.imread(str(next(frames_dir.glob("*.jpg"))))
    height, width = sample.shape[:2]
    print(f"input: {frame_count} frames @ {width}x{height}, masks from {masks_dir}")

    slam_dir = out_dir / "slam"
    if args.reuse_slam and (slam_dir / "points_background.npz").exists():
        print(f"reusing existing SLAM run in {slam_dir}")
        report_path = slam_dir / "points_background.json"
        slam_report = json.loads(report_path.read_text(encoding="utf-8"))
        reconstruction = slam_dir / "points_background.npz"
        values = np.load(reconstruction)
        slam_result = {
            "pointcloud": slam_dir / "points_background.ply",
            "reconstruction": reconstruction,
            "extrinsics": values["extrinsics"],
            "intrinsics": values["intrinsics"],
            "report": slam_report,
        }
    else:
        slam_result = SLAMAdapter(cfg["slam"], PROJECT_ROOT).run(
            frames_dir, slam_dir, masks_dir
        )

    import open3d as o3d

    cloud = o3d.io.read_point_cloud(str(slam_result["pointcloud"]))
    points = np.asarray(cloud.points)
    colors = np.asarray(cloud.colors)
    if not len(points):
        raise RuntimeError("Masked reconstruction returned an empty point cloud")
    save_pointcloud(points, out_dir / "pointcloud_masked.ply", colors)
    print(f"masked background points: {len(points)}")

    geometry_cfg = cfg.get("geometry", {})
    opening_hints = _load_opening_hints(out_dir, cfg, frames_dir, slam_result)
    geometry_report = build_mesh(
        points,
        colors,
        out_dir,
        geometry_cfg,
        reconstruction_path=slam_result["reconstruction"],
        opening_hints=opening_hints,
    )
    write_html(
        points,
        colors,
        out_dir / "interactive.html",
        mesh_path=out_dir / "background_mesh.ply",
        extrinsics=slam_result["extrinsics"],
    )

    scale_diagnosis = diagnose_submap_scales(slam_result["report"])
    report = {
        "route": "masked VGGT-SLAM on original frames + structural priors (no generative video)",
        "input_frames": str(frames_dir),
        "input_masks": str(masks_dir),
        "frames": frame_count,
        "slam": slam_result["report"],
        "submap_scales": scale_diagnosis,
        "geometry": geometry_report,
        "opening_hints": len(opening_hints) if opening_hints else 0,
    }
    (out_dir / "geometry_prior_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(
        {
            "background_points": len(points),
            "room": {k: geometry_report["room"][k] for k in
                     ("room_height", "wall_count", "wall_source", "wall_openings")},
            "surface_vertices": geometry_report["surface_vertices"],
            "structural_vertices": geometry_report["structural_vertices"],
            "combined_triangles": geometry_report["combined_triangles"],
            "scale_outliers": scale_diagnosis.get("scale_outliers"),
        },
        indent=2, ensure_ascii=False,
    ))


if __name__ == "__main__":
    main()