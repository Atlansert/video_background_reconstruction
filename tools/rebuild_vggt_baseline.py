"""VGGT-SLAM baseline on the ORIGINAL input video (no inpainting in the loop).

Runs the exact production chain — SLAMAdapter (same config) on the original
001.mp4 frames with EMPTY masks, then build_mesh with opening carving and the
interactive viewer — into a separate output directory, so the result can be
compared 1:1 with the redepth GLB built from the regenerated background video.
Only the input frames differ; every SLAM/geometry parameter is identical.

Usage (vbr environment):
    python -m tools.rebuild_vggt_baseline \
        --frames-dir outputs/001_sam31_slam/frames_all \
        --out outputs/vggt_slam_baseline
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from tools.rebuild_geometry_from_background import (
    diagnose_submap_scales,
    write_empty_masks,
)
from vbr.cli import PROJECT_ROOT, _load_opening_hints
from vbr.config import load_config
from vbr.geometry import build_mesh, save_pointcloud
from vbr.interactive import write_html
from vbr.models.slam import SLAMAdapter
from vbr.video import video_info

import cv2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-dir", type=Path,
                        default=PROJECT_ROOT / "outputs/001_sam31_slam/frames_all")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "outputs/vggt_slam_baseline")
    parser.add_argument("--config", default="configs/vggt_slam.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    frames_dir = (PROJECT_ROOT / args.frames_dir).resolve() if not args.frames_dir.is_absolute() else args.frames_dir
    out_dir = (PROJECT_ROOT / args.out).resolve() if not args.out.is_absolute() else args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    frame_count = len(list(frames_dir.glob("*.jpg")))
    sample = cv2.imread(str(next(frames_dir.glob("*.jpg"))))
    height, width = sample.shape[:2]
    print(f"frames: {frame_count} @ {width}x{height}")

    masks_empty = write_empty_masks(out_dir / "masks_empty", frame_count)
    slam_result = SLAMAdapter(cfg["slam"], PROJECT_ROOT).run(
        frames_dir, out_dir / "slam", masks_empty
    )

    import open3d as o3d
    cloud = o3d.io.read_point_cloud(str(slam_result["pointcloud"]))
    points = np.asarray(cloud.points)
    colors = np.asarray(cloud.colors)
    if not len(points):
        raise RuntimeError("Baseline reconstruction returned an empty point cloud")
    save_pointcloud(points, out_dir / "pointcloud_original.ply", colors)

    opening_hints = _load_opening_hints(out_dir, cfg, frames_dir, slam_result)
    geometry_report = build_mesh(
        points,
        colors,
        out_dir,
        cfg.get("geometry", {}),
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
        "input_frames": str(frames_dir),
        "masks": "empty (no foreground rejection; raw VGGT-SLAM baseline)",
        "slam": slam_result["report"],
        "submap_scales": scale_diagnosis,
        "geometry": geometry_report,
    }
    (out_dir / "baseline_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "output_points": len(points),
        "submap_scales": scale_diagnosis,
        "geometry": {k: geometry_report[k] for k in
                     ("room", "surface_vertices", "combined_vertices", "combined_triangles")
                     if k in geometry_report},
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
