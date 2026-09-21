"""Render a walkthrough video along the reconstructed camera path.

The mesh is rendered from the SLAM cameras — the same physical motion the
original video recorded — interpolated to every output frame, then encoded to
H.264 with the original audio track. Positions interpolate linearly and
rotations with slerp between consecutive keyframes; frames outside the
keyframe range hold the nearest keyframe pose.

Usage (vbr environment), route B as an example:
    python -m tools.render_trajectory_video \
        --run-dir outputs/geometry_hybrid_b \
        --out outputs/geometry_hybrid_b/trajectory_video.mp4

Route A / production differ only in the mesh + reconstruction NPZ paths:
    python -m tools.render_trajectory_video --run-dir outputs/geometry_prior_a \
        --out outputs/geometry_prior_a/trajectory_video.mp4
    python -m tools.render_trajectory_video --run-dir outputs/001_sam31_slam \
        --npz outputs/001_sam31_slam/slam_bg/points_background.npz \
        --out outputs/001_sam31_slam/trajectory_video_redepth.mp4
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from vbr.cli import PROJECT_ROOT
from vbr.config import load_config
from vbr.models.inpainting import _resolve_ffmpeg
from vbr.video import video_info


def cam_to_world_poses(extrinsics):
    """3x4 world-to-camera matrices -> 4x4 camera-to-world matrices."""
    extrinsics = np.asarray(extrinsics, dtype=np.float64)
    count = len(extrinsics)
    poses = np.tile(np.eye(4, dtype=np.float64), (count, 1, 1))
    poses[:, :3, :3] = np.transpose(extrinsics[:, :3, :3], (0, 2, 1))
    poses[:, :3, 3] = -np.einsum("nij,nj->ni", poses[:, :3, :3], extrinsics[:, :3, 3])
    return poses


def render_intrinsics(intrinsics, coords, width, height):
    """Model-space intrinsics -> intrinsics for the rendering resolution.

    ``coords`` is the per-frame letterbox content box [x1, y1, x2, y2, w, h]
    of the frame inside the model canvas, so this handles crop and square
    modes. Rows repeat the same box for a fixed input; per-row values keep it
    correct if a run ever mixes sizes.
    """
    intrinsics = np.asarray(intrinsics, dtype=np.float64)
    coords = np.asarray(coords, dtype=np.float64).reshape(-1, 6)
    x1, y1, x2, y2 = coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3]
    scale_x = width / np.maximum(x2 - x1, 1e-6)
    scale_y = height / np.maximum(y2 - y1, 1e-6)
    out = np.zeros_like(intrinsics)
    out[:, 0, 0] = intrinsics[:, 0, 0] * scale_x
    out[:, 0, 2] = (intrinsics[:, 0, 2] - x1) * scale_x
    out[:, 1, 1] = intrinsics[:, 1, 1] * scale_y
    out[:, 1, 2] = (intrinsics[:, 1, 2] - y1) * scale_y
    out[:, 2, 2] = 1.0
    return out


def interpolate_path(frame_ids, poses, intrinsics, frames):
    """Per-output-frame camera pose and (fx, fy, cx, cy) along the path."""
    from scipy.spatial.transform import Rotation, Slerp

    ids = np.asarray(frame_ids, dtype=np.float64)
    order = np.argsort(ids)
    ids = ids[order]
    poses = poses[order]
    intrinsics = intrinsics[order]
    query = np.clip(np.asarray(frames, dtype=np.float64), ids[0], ids[-1])
    positions = np.stack(
        [np.interp(query, ids, poses[:, axis, 3]) for axis in range(3)], axis=1
    )
    slerp = Slerp(ids, Rotation.from_matrix(poses[:, :3, :3]))
    rotations = slerp(query).as_matrix()
    fx = np.interp(query, ids, intrinsics[:, 0, 0])
    fy = np.interp(query, ids, intrinsics[:, 1, 1])
    cx = np.interp(query, ids, intrinsics[:, 0, 2])
    cy = np.interp(query, ids, intrinsics[:, 1, 2])
    return positions, rotations, np.stack([fx, fy, cx, cy], axis=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=None,
                        help="output directory holding the mesh and slam/ NPZ")
    parser.add_argument("--mesh", type=Path, default=None,
                        help="mesh to render (default <run-dir>/background_mesh.ply)")
    parser.add_argument("--npz", type=Path, default=None,
                        help="reconstruction NPZ with poses (default "
                             "<run-dir>/slam/points_background.npz)")
    parser.add_argument("--out", type=Path, default=None,
                        help="output mp4 (default <run-dir>/trajectory_video.mp4)")
    parser.add_argument("--input-video", default=None,
                        help="source video for timing/audio (default from config)")
    parser.add_argument("--config", default="configs/vggt_slam.yaml")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--fps", type=float, default=None,
                        help="output frame rate (default: source video fps)")
    parser.add_argument("--stride", type=int, default=1,
                        help="render every Nth frame (output keeps real-time "
                             "pace by dividing the frame rate)")
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--no-audio", action="store_true")
    parser.add_argument("--background", default="0.88,0.88,0.90",
                        help="background RGB in 0-1, comma separated")
    parser.add_argument("--near", type=float, default=0.03)
    parser.add_argument("--far", type=float, default=80.0)
    parser.add_argument("--limit", type=int, default=0,
                        help="render only the first N output frames (smoke test)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_dir = (
        (PROJECT_ROOT / args.run_dir).resolve()
        if args.run_dir and not args.run_dir.is_absolute()
        else args.run_dir
    )
    mesh_path = args.mesh or (run_dir / "background_mesh.ply")
    npz_path = args.npz or (run_dir / "slam" / "points_background.npz")
    out_path = args.out or (run_dir / "trajectory_video.mp4")
    mesh_path = Path(mesh_path)
    npz_path = Path(npz_path)
    out_path = Path(out_path)
    for path in (mesh_path, npz_path):
        if not path.exists():
            raise FileNotFoundError(path)

    input_video = args.input_video or cfg.get("input_video", "video/001.mp4")
    input_video = Path(input_video)
    if not input_video.is_absolute():
        input_video = (PROJECT_ROOT / input_video).resolve()
    has_video = input_video.exists()
    if has_video:
        info = video_info(input_video)
        total_frames = int(info["frames"])
        fps = float(args.fps if args.fps else info["fps"])
    else:
        total_frames = max(0, int(np.max(np.load(npz_path)["frame_ids"])) + 1)
        fps = float(args.fps if args.fps else 30.0)
        print(f"note: input video {input_video} not found; timing from keyframes")

    values = np.load(npz_path)
    frame_ids = np.asarray(values["frame_ids"])
    poses = cam_to_world_poses(values["extrinsics"])
    intrinsics = render_intrinsics(
        values["intrinsics"], values["original_coords"], args.width, args.height
    )
    stride = max(1, int(args.stride))
    frames = np.arange(0, total_frames, stride)
    if args.limit:
        frames = frames[: args.limit]
    positions, rotations, camera_intrinsics = interpolate_path(
        frame_ids, poses, intrinsics, frames
    )
    out_fps = fps / stride
    print(
        f"mesh={mesh_path} keyframes={len(frame_ids)} output_frames={len(frames)} "
        f"{args.width}x{args.height}@{out_fps:.3f} fps"
    )

    import open3d as o3d

    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if not mesh.has_vertex_colors():
        print("warning: mesh has no vertex colors; geometry renders gray")
    mesh.compute_vertex_normals()

    renderer = o3d.visualization.rendering.OffscreenRenderer(args.width, args.height)
    scene = renderer.scene
    background = [float(value) for value in args.background.split(",")]
    scene.set_background([*background, 1.0])
    scene.scene.set_sun_light([-0.35, -0.55, -0.75], [1.0, 1.0, 1.0], 65000.0)
    scene.scene.enable_sun_light(True)
    scene.scene.set_indirect_light_intensity(35000.0)
    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultLit"
    scene.add_geometry("mesh", mesh, material)
    camera = scene.camera

    out_path.parent.mkdir(parents=True, exist_ok=True)
    encode_audio = has_video and not args.no_audio
    ffmpeg = _resolve_ffmpeg()
    print(f"encoder: {ffmpeg}")
    command = [
        ffmpeg, "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pixel_format", "rgb24",
        "-video_size", f"{args.width}x{args.height}",
        "-framerate", f"{out_fps}", "-i", "-",
    ]
    if encode_audio:
        command += ["-i", str(input_video), "-map", "0:v:0", "-map", "1:a:0?"]
    command += [
        "-c:v", "libx264", "-preset", "medium", "-crf", str(args.crf),
        "-pix_fmt", "yuv420p",
    ]
    if encode_audio:
        command += ["-c:a", "aac", "-shortest"]
    command += [str(out_path)]
    log_path = out_path.with_suffix(".ffmpeg.log")
    with log_path.open("w", encoding="utf-8") as ffmpeg_log:
        process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
            stderr=ffmpeg_log,
        )
        started = time.time()
        try:
            for index in range(len(frames)):
                eye = positions[index].astype(np.float32)
                forward = rotations[index][:, 2].astype(np.float32)
                up = (-rotations[index][:, 1]).astype(np.float32)
                camera.look_at((eye + forward).astype(np.float32), eye, up)
                fx, fy, cx, cy = camera_intrinsics[index]
                camera.set_projection(
                    np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]),
                    float(args.near), float(args.far),
                    float(args.width), float(args.height),
                )
                image = np.asarray(renderer.render_to_image())
                process.stdin.write(image.tobytes())
                if (index + 1) % 100 == 0 or index + 1 == len(frames):
                    elapsed = time.time() - started
                    print(
                        f"  frame {index + 1}/{len(frames)} "
                        f"({(index + 1) / elapsed:.1f} fps, {elapsed:.0f}s)",
                        flush=True,
                    )
            process.stdin.close()
        except BrokenPipeError:
            pass
        return_code = process.wait()
    if return_code:
        tail = "\n".join(log_path.read_text(encoding="utf-8").splitlines()[-20:])
        raise RuntimeError(f"ffmpeg failed ({return_code}); see {log_path}\n{tail}")

    elapsed = time.time() - started
    (out_path.with_suffix(".json")).write_text(
        json.dumps(
            {
                "mesh": str(mesh_path),
                "reconstruction": str(npz_path),
                "input_video": str(input_video) if has_video else None,
                "frames": len(frames),
                "keyframes": int(len(frame_ids)),
                "stride": stride,
                "resolution": [args.width, args.height],
                "fps": out_fps,
                "seconds": float(len(frames) / out_fps),
                "render_elapsed_seconds": elapsed,
                "audio": bool(encode_audio),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB) in {elapsed:.0f}s")


if __name__ == "__main__":
    main()