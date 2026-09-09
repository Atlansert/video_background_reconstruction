"""Rebuild the room geometry from the final background video.

The original reconstruction fused depth only where the foreground masks were
absent, so the surfaces behind the removed furniture never get observations
and the walls fall back to a flat footprint box. Re-running the same VGGT-SLAM
backend on the inpainted background video recovers depth everywhere: the
inpainted texture replaces the hidden surfaces, and the multi-view TSDF fusion
turns them into a continuous shell. The geometry stage then re-fuses the new
point maps and re-exports the mesh/GLB.

The original slam/ directory and the pre-rebuild GLB/PLY are snapshotted
before anything is overwritten.

Usage (vbr environment):
    python -m tools.rebuild_geometry_from_background \
        --output-dir outputs/001_sam31_slam --config configs/vggt_slam.yaml
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

from vbr.cli import PROJECT_ROOT, _load_opening_hints
from vbr.config import load_config
from vbr.geometry import build_mesh, save_pointcloud
from vbr.interactive import write_html
from vbr.models.slam import SLAMAdapter
from vbr.video import video_info


def extract_background_frames(output_dir: Path) -> Path:
    background_frames = output_dir / "frames_background"
    background_frames.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / "background_video.mp4"
    info = video_info(video_path)
    capture = cv2.VideoCapture(str(video_path))
    written = 0
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        destination = background_frames / f"{index:06d}.jpg"
        if not destination.exists():
            cv2.imwrite(str(destination), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            written += 1
        index += 1
    capture.release()
    if index != info["frames"]:
        raise RuntimeError(
            f"Background video has {index} frames, expected {info['frames']}"
        )
    return background_frames


def write_empty_masks(mask_dir: Path, frame_count: int) -> Path:
    mask_dir = Path(mask_dir)
    mask_dir.mkdir(parents=True, exist_ok=True)
    blank = np.zeros((540, 960), dtype=np.uint8)
    for index in range(frame_count):
        path = mask_dir / f"{index:06d}.png"
        if not path.exists():
            cv2.imwrite(str(path), blank)
    return mask_dir


def diagnose_submap_scales(report: dict) -> dict:
    scales = report.get("submap_scales", {})
    values = [float(value) for value in scales.values()]
    if not values:
        return {"checked": False, "note": "no submap_scales in backend report"}
    median = float(np.median(values))
    outliers = {
        str(submap): float(value)
        for submap, value in scales.items()
        if abs(float(value) - median) / max(median, 1e-9) > 0.3
    }
    return {
        "checked": True,
        "submaps": len(values),
        "median_scale": round(median, 4),
        "min_scale": round(float(min(values)), 4),
        "max_scale": round(float(max(values)), 4),
        "scale_outliers": outliers,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--config", default="configs/vggt_slam.yaml")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    cfg = load_config(args.config)
    info = video_info(output_dir / "background_video.mp4")

    for snapshot in ("background_scene.glb", "background_mesh.ply"):
        source = output_dir / snapshot
        if source.exists() and not (
            output_dir / f"{Path(snapshot).stem}_pre_redepth{Path(snapshot).suffix}"
        ).exists():
            shutil.copy2(
                source,
                output_dir / f"{Path(snapshot).stem}_pre_redepth{Path(snapshot).suffix}",
            )

    background_frames = extract_background_frames(output_dir)
    masks_empty = write_empty_masks(output_dir / "masks_empty", info["frames"])

    slam_dir_bg = output_dir / "slam_bg"
    slam_result = SLAMAdapter(cfg["slam"], PROJECT_ROOT).run(
        background_frames, slam_dir_bg, masks_empty
    )
    cloud = o3d.io.read_point_cloud(str(slam_result["pointcloud"]))
    points = np.asarray(cloud.points)
    colors = np.asarray(cloud.colors)
    if not len(points):
        raise RuntimeError("Background reconstruction returned an empty point cloud")
    save_pointcloud(points, output_dir / "pointcloud_background.ply", colors)

    geometry_cfg = cfg.get("geometry", {})
    opening_hints = _load_opening_hints(output_dir, cfg, output_dir / "frames_all", slam_result)
    geometry_report = build_mesh(
        points,
        colors,
        output_dir,
        geometry_cfg,
        reconstruction_path=slam_result["reconstruction"],
        opening_hints=opening_hints,
    )
    write_html(
        points,
        colors,
        output_dir / "interactive.html",
        mesh_path=output_dir / "background_mesh.ply",
        extrinsics=slam_result["extrinsics"],
    )

    scale_diagnosis = diagnose_submap_scales(slam_result["report"])
    (output_dir / "geometry_redepth_report.json").write_text(
        json.dumps(
            {
                "input": "background_video.mp4",
                "frames_extracted": info["frames"],
                "slam": slam_result["report"],
                "submap_scales": scale_diagnosis,
                "geometry": geometry_report,
                "opening_hints": len(opening_hints) if opening_hints else 0,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(
        {
            "wall_source": geometry_report["room"]["wall_source"],
            "wall_count": geometry_report["room"]["wall_count"],
            "surface_vertices": geometry_report["surface_vertices"],
            "combined_triangles": geometry_report["combined_triangles"],
            "mesh_clean": geometry_report["mesh_clean"],
            "scale_outliers": scale_diagnosis["scale_outliers"],
            "background_points": len(points),
        },
        indent=2,
        ensure_ascii=False,
    ))


if __name__ == "__main__":
    main()