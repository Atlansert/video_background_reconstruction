"""Rebuild only the geometry stage for an existing output directory.

Avoids re-running reconstruction when iterating on geometry parameters.
Reads the existing point cloud and reconstruction NPZ, then rewrites
structural_planes.ply, background_mesh.ply, geometry_report.json,
background_scene.glb and interactive.html.

Usage (vbr environment):
    python -m tools.rebuild_geometry --config configs/default.yaml
    python -m tools.rebuild_geometry --output-dir outputs/001_sam31_slam \
        --config configs/vggt_slam.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from vbr.cli import PROJECT_ROOT, _load_opening_hints
from vbr.config import load_config
from vbr.geometry import build_mesh, save_pointcloud
from vbr.interactive import write_html
from vbr.models.slam import SLAMAdapter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="defaults to the config's output_dir",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = (PROJECT_ROOT / (args.output_dir or cfg["output_dir"])).resolve()
    slam_dir = output_dir / "slam"
    pointcloud_path = slam_dir / "points_background.ply"
    reconstruction_path = slam_dir / "points_background.npz"
    if not pointcloud_path.exists() or not reconstruction_path.exists():
        raise SystemExit(f"Missing reconstruction outputs in {slam_dir}")

    import open3d as o3d

    cloud = o3d.io.read_point_cloud(str(pointcloud_path))
    points = np.asarray(cloud.points)
    colors = np.asarray(cloud.colors)
    if not len(points):
        raise SystemExit("Empty point cloud")

    slam_result = {
        "reconstruction": reconstruction_path,
        "pointcloud": pointcloud_path,
        "extrinsics": np.load(reconstruction_path)["extrinsics"],
        "intrinsics": np.load(reconstruction_path)["intrinsics"],
        "report": {},
    }
    opening_hints = _load_opening_hints(
        output_dir, cfg, output_dir / "frames_all", slam_result
    )
    print(f"opening hints: {len(opening_hints) if opening_hints else 0}")

    values = np.load(reconstruction_path)
    geometry_report = build_mesh(
        points,
        colors,
        output_dir,
        cfg.get("geometry", {}),
        reconstruction_path=reconstruction_path,
        opening_hints=opening_hints,
    )
    save_pointcloud(points, output_dir / "pointcloud_background.ply", colors)
    write_html(
        points,
        colors,
        output_dir / "interactive.html",
        mesh_path=output_dir / "background_mesh.ply",
        extrinsics=values["extrinsics"],
    )
    (output_dir / "geometry_report.json").write_text(
        json.dumps(geometry_report, indent=2), encoding="utf-8"
    )
    print(json.dumps(geometry_report, indent=2))


if __name__ == "__main__":
    main()
