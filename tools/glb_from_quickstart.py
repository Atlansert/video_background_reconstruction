"""Build a GLB from the official VGGT-SLAM quick start point cloud.

Feeds poses_points.pcd (the dense output of the official main.py quick start)
through the same vbr geometry stage (build_mesh) used for the production GLB.
No reconstruction npz exists on this path, so the surface step falls back to
Poisson reconstruction instead of TSDF fusion, and wall-opening carving is
skipped (no per-frame depth association).

Usage (vbr environment):
    python -m tools.glb_from_quickstart \
        --pcd outputs/vggt_quickstart/001_official/poses_points.pcd
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d

from vbr.cli import PROJECT_ROOT
from vbr.config import load_config
from vbr.geometry import build_mesh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pcd", type=Path,
                        default=PROJECT_ROOT / "outputs/vggt_quickstart/001_official/poses_points.pcd")
    parser.add_argument("--out", type=Path,
                        default=PROJECT_ROOT / "outputs/vggt_quickstart/001_official/glb")
    parser.add_argument("--config", default="configs/vggt_slam.yaml")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="uniform scale applied to the cloud before meshing "
                             "(world units -> meters); 1.0 keeps the SLAM scale")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cloud = o3d.io.read_point_cloud(str(args.pcd.resolve()))
    points = np.asarray(cloud.points)
    colors = np.asarray(cloud.colors)
    if not len(points):
        raise RuntimeError(f"Empty point cloud {args.pcd}")
    if args.scale != 1.0:
        points = points * args.scale
    print(f"points: {len(points)}, extent: "
          f"{np.ptp(points, axis=0).round(3)}")

    report = build_mesh(
        points,
        colors,
        out_dir,
        dict(cfg.get("geometry", {})),
        reconstruction_path=None,
        opening_hints=None,
    )
    (out_dir / "glb_report.json").write_text(
        json.dumps({"source": str(args.pcd), "scale": args.scale,
                    "geometry": report}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({
        "room": report.get("room"),
        "surface_method": report.get("surface_method"),
        "combined_vertices": report.get("combined_vertices"),
        "combined_triangles": report.get("combined_triangles"),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
