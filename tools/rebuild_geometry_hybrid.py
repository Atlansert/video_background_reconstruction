"""Hybrid geometry rebuild (branch B): route-A geometry + route-P texture.

Geometry comes from the same masked VGGT-SLAM run as route A (original
frames, current masks, structural-prior completion), so occluded surfaces
stay gravity-aligned planes without any diffusion-generated depth in the
loop. Texture is then projected from the SVOR-inpainted video: real pixels
where the camera saw the surface, generated pixels where furniture used to
be. Diffusion can therefore influence appearance but never geometry.

Usage (vbr environment):
    python -m tools.rebuild_geometry_hybrid \
        --out outputs/geometry_hybrid_b --config configs/vggt_slam.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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
                        help="pipeline run directory holding frames_all/, masks/ "
                             "and background_video.mp4")
    parser.add_argument("--out", type=Path,
                        default=PROJECT_ROOT / "outputs/geometry_hybrid_b")
    parser.add_argument("--config", default="configs/vggt_slam.yaml")
    parser.add_argument("--texture-max-frames", type=int, default=48,
                        help="cap on video frames used for texture projection")
    parser.add_argument("--reuse-slam", action="store_true",
                        help="reuse an existing <out>/slam instead of re-running "
                             "masked VGGT-SLAM (for re-checking the geometry stage)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_dir = (PROJECT_ROOT / args.run_dir).resolve() if not args.run_dir.is_absolute() else args.run_dir
    out_dir = (PROJECT_ROOT / args.out).resolve() if not args.out.is_absolute() else args.out
    frames_dir = run_dir / "frames_all"
    masks_dir = run_dir / "masks"
    video_path = run_dir / "background_video.mp4"
    for path in (frames_dir, masks_dir, video_path):
        if not path.exists():
            raise RuntimeError(f"Missing {path}; expected a completed pipeline run")
    out_dir.mkdir(parents=True, exist_ok=True)

    frame_count = len(list(frames_dir.glob("*.jpg")))
    mask_count = len(list(masks_dir.glob("*.png")))
    if frame_count != mask_count:
        raise RuntimeError(
            f"Frame/mask count mismatch: {frame_count} frames vs {mask_count} masks"
        )
    print(f"input: {frame_count} frames, masks from {masks_dir}, "
          f"texture from {video_path}")

    slam_dir = out_dir / "slam"
    if args.reuse_slam and (slam_dir / "points_background.npz").exists():
        print(f"reusing existing SLAM run in {slam_dir}")
        slam_report = json.loads(
            (slam_dir / "points_background.json").read_text(encoding="utf-8")
        )
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

    import cv2
    import open3d as o3d

    cloud = o3d.io.read_point_cloud(str(slam_result["pointcloud"]))
    points = np.asarray(cloud.points)
    colors = np.asarray(cloud.colors)
    if not len(points):
        raise RuntimeError("Masked reconstruction returned an empty point cloud")
    save_pointcloud(points, out_dir / "pointcloud_masked.ply", colors)
    print(f"masked background points: {len(points)}")

    # Texture frames: every frame of the inpainted video, mapped frame-id wise.
    video_frames_dir = out_dir / "frames_video"
    video_frames_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    written = 0
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        destination = video_frames_dir / f"{index:06d}.jpg"
        if not destination.exists():
            cv2.imwrite(str(destination), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            written += 1
        index += 1
    capture.release()
    texture_frames = sorted(video_frames_dir.glob("*.jpg"), key=lambda path: int(path.stem))
    print(f"texture frames extracted: {written} new, {len(texture_frames)} total "
          f"({index} in video)")

    geometry_cfg = cfg.get("geometry", {})
    opening_hints = _load_opening_hints(out_dir, cfg, frames_dir, slam_result)
    geometry_report = build_mesh(
        points,
        colors,
        out_dir,
        geometry_cfg,
        reconstruction_path=slam_result["reconstruction"],
        opening_hints=opening_hints,
        texture_frames=texture_frames,
        texture_max_frames=int(args.texture_max_frames),
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
        "route": "masked VGGT-SLAM geometry on original frames + structural priors, "
                 "texture projected from the SVOR-inpainted video",
        "input_frames": str(frames_dir),
        "input_masks": str(masks_dir),
        "texture_video": str(video_path),
        "frames": frame_count,
        "slam": slam_result["report"],
        "submap_scales": scale_diagnosis,
        "geometry": geometry_report,
        "opening_hints": len(opening_hints) if opening_hints else 0,
    }
    (out_dir / "geometry_hybrid_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(
        {
            "background_points": len(points),
            "room": {k: geometry_report["room"][k] for k in
                     ("room_height", "wall_count", "wall_source", "wall_openings")},
            "surface_vertices": geometry_report["surface_vertices"],
            "structural_vertices": geometry_report["structural_vertices"],
            "combined_vertices": geometry_report["combined_vertices"],
            "combined_triangles": geometry_report["combined_triangles"],
            "texture": geometry_report.get("texture"),
            "scale_outliers": scale_diagnosis.get("scale_outliers"),
        },
        indent=2, ensure_ascii=False,
    ))


if __name__ == "__main__":
    main()