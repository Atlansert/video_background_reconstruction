"""Denoise the raw VGGT-SLAM baseline surface (no foreground removal).

The baseline runs with ``confidence_percentile: 0`` (official dense output),
so ~40% of the fused samples sit at the lowest confidence and TSDF turns them
into thousands of floating micro-fragments ("sparkle" in the air). The fix
does NOT raise the confidence cutoff — the walls themselves are low-confidence
at grazing angles, and cutting them thins the room. Instead:

1. Re-fuse the TSDF surface exactly like the baseline (same voxel/truncation),
2. drop connected components smaller than ``--min-component-triangles``
   (micro-fragments in the air are tiny islands; furniture pieces and walls
   stay well above the threshold),
3. optionally raise the confidence cutoff only when explicitly requested.

Usage (vbr environment):
    python -m tools.denoise_baseline_mesh \
        --run-dir outputs/vggt_slam_baseline \
        --out outputs/vggt_slam_baseline/background_mesh_denoised.ply
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from vbr.cli import PROJECT_ROOT
from vbr.config import load_config
from vbr.geometry import _model_rgb


def build_tsdf_surface(data, cutoff, geometry_cfg):
    import open3d as o3d

    depth = data["depth"]
    confidence = data["confidence"]
    foreground = data["foreground_masks"]
    intrinsics = data["intrinsics"]
    extrinsics = data["extrinsics"]
    coords = data["original_coords"]
    frame_paths = data["frame_paths"]
    height = int(depth.shape[1])
    width = int(depth.shape[2])
    positive = depth[..., 0][depth[..., 0] > 0]
    depth_trunc = float(
        np.quantile(positive, geometry_cfg.get("depth_trunc_quantile", 0.995))
    )
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=geometry_cfg.get("voxel_size", 0.02),
        sdf_trunc=geometry_cfg.get("tsdf_trunc", 0.10),
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )
    for index, frame_path in enumerate(frame_paths):
        depth_image = depth[index, ..., 0].copy()
        depth_image[(confidence[index] < cutoff) | foreground[index]] = 0
        color_image = _model_rgb(str(frame_path), coords[index], width, height)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(color_image),
            o3d.geometry.Image(depth_image.astype(np.float32)),
            depth_scale=1.0,
            depth_trunc=depth_trunc,
            convert_rgb_to_intensity=False,
        )
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width,
            height,
            float(intrinsics[index, 0, 0]),
            float(intrinsics[index, 1, 1]),
            float(intrinsics[index, 0, 2]),
            float(intrinsics[index, 1, 2]),
        )
        extrinsic = np.eye(4)
        extrinsic[:3] = extrinsics[index]
        volume.integrate(rgbd, intrinsic, extrinsic)
    mesh = volume.extract_triangle_mesh()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_non_manifold_edges()
    return mesh


def filter_components(mesh, min_triangles):
    import open3d as o3d

    clusters, counts, _ = mesh.cluster_connected_triangles()
    clusters = np.asarray(clusters)
    sizes = np.bincount(clusters)
    keep = np.flatnonzero(sizes >= min_triangles)
    triangles = np.asarray(mesh.triangles)
    mask = np.isin(clusters, keep)
    subset = o3d.geometry.TriangleMesh()
    subset.vertices = mesh.vertices
    subset.triangles = o3d.utility.Vector3iVector(triangles[mask])
    if mesh.has_vertex_colors():
        subset.vertex_colors = mesh.vertex_colors
    subset.remove_unreferenced_vertices()
    subset.compute_vertex_normals()
    return subset, {
        "components_before": int(len(sizes)),
        "components_after": int(len(keep)),
        "triangles_before": int(len(triangles)),
        "triangles_after": int(len(subset.triangles)),
        "dropped_components": int(len(sizes) - len(keep)),
        "dropped_triangles": int(len(triangles) - len(subset.triangles)),
        "largest_component_fraction": float(sizes.max() / max(1, len(triangles))),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path,
                        default=PROJECT_ROOT / "outputs/vggt_slam_baseline")
    parser.add_argument("--npz", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--config", default="configs/vggt_slam.yaml")
    parser.add_argument("--confidence-cutoff", type=float, default=1.0,
                        help="fuse only samples at/above this depth confidence. "
                             "Default 1.0 keeps the baseline behavior (the "
                             "stored cutoff); raise to 2-3 for a more aggressive "
                             "clean at the cost of thinning grazing-angle walls.")
    parser.add_argument("--min-component-triangles", type=int, default=300,
                        help="drop connected components with fewer triangles "
                             "(airborne micro-fragments)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_dir = (
        (PROJECT_ROOT / args.run_dir).resolve()
        if not args.run_dir.is_absolute() else args.run_dir
    )
    npz_path = args.npz or (run_dir / "slam" / "points_background.npz")
    out_path = args.out or (run_dir / "background_mesh_denoised.ply")
    if not Path(npz_path).exists():
        raise FileNotFoundError(npz_path)

    import open3d as o3d

    data = np.load(npz_path)
    cutoff = float(args.confidence_cutoff)
    print(f"fusing TSDF: cutoff={cutoff} "
          f"voxel={cfg['geometry'].get('voxel_size')} "
          f"trunc={cfg['geometry'].get('tsdf_trunc')}")
    mesh = build_tsdf_surface(data, cutoff, cfg.get("geometry", {}))
    print(f"raw surface: {len(mesh.triangles)} triangles")
    mesh, stats = filter_components(mesh, int(args.min_component_triangles))
    print("component filter:", json.dumps(stats, indent=2))
    o3d.io.write_triangle_mesh(str(out_path), mesh)
    report = {
        "input": str(npz_path),
        "output": str(out_path),
        "confidence_cutoff": cutoff,
        "min_component_triangles": int(args.min_component_triangles),
        "stats": stats,
        "vertices": int(len(mesh.vertices)),
        "triangles": int(len(mesh.triangles)),
    }
    out_path.with_suffix(".json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"wrote {out_path} ({len(mesh.triangles)} triangles)")


if __name__ == "__main__":
    main()