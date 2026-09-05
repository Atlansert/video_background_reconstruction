"""Command-line entry point for foreground-free room reconstruction."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from .config import load_config
from .geometry import build_mesh, save_pointcloud
from .interactive import write_html
from .models.segmentation import SegmentationAdapter
from .models.slam import SLAMAdapter
from .models.inpainting import ProPainterAdapter, _resolve_ffmpeg
from .prompts import resolve_prompts
from .video import (
    extract_frame_sets,
    fill_enclosed_mask_holes,
    mask_statistics,
    video_info,
    write_background_video,
    write_mask_overlay,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Mask caches only need invalidation when segmentation-relevant code changes;
# hashing everything would needlessly discard usable masks on unrelated edits.
_SEGMENTATION_SOURCE_MODULES = (
    "cli.py",
    "config.py",
    "sam31_keyframes.py",
    "sam2_propagate.py",
    "prompts.py",
    "models/segmentation.py",
)


def _segmentation_source_hash() -> str:
    digest = hashlib.sha256()
    package_dir = Path(__file__).resolve().parent
    for relative in _SEGMENTATION_SOURCE_MODULES:
        path = package_dir / relative
        try:
            digest.update(path.read_bytes())
        except OSError:
            digest.update(f"missing:{relative}".encode())
    return digest.hexdigest()[:16]


def _fingerprint(video_path: Path, section: dict) -> str:
    stat = video_path.stat()
    payload = {
        "video": str(video_path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "config": section,
        "code": _segmentation_source_hash(),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _cached(manifest_path: Path, fingerprint: str, expected_files: int, files_dir: Path):
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        manifest.get("fingerprint") == fingerprint
        and len(list(files_dir.glob("*.png"))) == expected_files
    )


def _clear_generated(directory: Path, suffixes=(".png",)):
    directory.mkdir(parents=True, exist_ok=True)
    for path in directory.iterdir():
        if path.is_file() and path.suffix.lower() in suffixes:
            path.unlink()


def _build_opening_hints(values, preserved_dir: Path) -> list | None:
    """Pair preserved fixed-structure masks with reconstruction poses."""
    import cv2

    frame_paths = values["frame_paths"]
    coords = values["original_coords"]
    extrinsics = values["extrinsics"]
    intrinsics = values["intrinsics"]
    depth = values["depth"][..., 0]
    index_by_stem = {int(Path(path).stem): index for index, path in enumerate(frame_paths)}
    hints = []
    for mask_path in sorted(preserved_dir.glob("*.png")):
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None or not np.count_nonzero(mask):
            continue
        index = index_by_stem.get(int(mask_path.stem))
        if index is None:
            continue
        hints.append(
            {
                "extrinsic": extrinsics[index],
                "intrinsic": intrinsics[index],
                "original_coords": coords[index],
                "depth": depth[index],
                "mask": mask > 0,
            }
        )
    return hints or None


def _load_opening_hints(output_dir: Path, cfg, all_frames: Path, slam_result: dict):
    """Reuse or generate fixed-structure masks for wall-opening carving."""
    geometry_cfg = cfg.get("geometry", {})
    if not geometry_cfg.get("carve_openings", True):
        return None
    prompts = geometry_cfg.get(
        "opening_prompts",
        [
            "door",
            "window",
            "staircase",
            "stairs",
            "kitchen cabinet",
            "refrigerator",
            "sink",
        ],
    )
    values = np.load(slam_result["reconstruction"])
    frame_ids = [int(value) for value in values["frame_ids"]]
    expected = {"frame_ids": frame_ids, "prompts": prompts}
    masks_out = output_dir / "masks_openings"
    manifest_path = masks_out / "manifest.json"
    if manifest_path.exists():
        try:
            if json.loads(manifest_path.read_text(encoding="utf-8")) == expected:
                return _build_opening_hints(values, masks_out / "preserved")
        except (OSError, json.JSONDecodeError):
            pass
    SegmentationAdapter(cfg["segmentation"], PROJECT_ROOT).run_opening_masks(
        frame_ids, all_frames, masks_out, output_dir / "logs", prompts
    )
    manifest_path.write_text(json.dumps(expected, indent=2), encoding="utf-8")
    return _build_opening_hints(values, masks_out / "preserved")


def _ensure_frames(video_path, info, all_frames, keyframes, stride):
    expected_keyframes = len(set(range(0, info["frames"], stride)) | {info["frames"] - 1})
    all_count = len(list(all_frames.glob("*.jpg")))
    key_count = len(list(keyframes.glob("*.jpg")))
    if all_count == info["frames"] and key_count == expected_keyframes:
        return sorted(int(path.stem) for path in keyframes.glob("*.jpg"))
    _clear_generated(all_frames, (".jpg",))
    _clear_generated(keyframes, (".jpg",))
    return extract_frame_sets(video_path, all_frames, keyframes, stride)


def run(cfg, stop_after="all", force=False):
    started = time.time()
    input_video = (PROJECT_ROOT / cfg["input_video"]).resolve()
    output_dir = (PROJECT_ROOT / cfg["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "pipeline_status.json"
    status = {"state": "running", "started_at_unix": started, "stages": {}}
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")

    try:
        info = video_info(input_video)
        (output_dir / "video_info.json").write_text(
            json.dumps(info, indent=2), encoding="utf-8"
        )
        all_frames = output_dir / "frames_all"
        keyframes = output_dir / "frames_keyframes"
        key_stride = int(cfg.get("sampling", {}).get("segmentation_stride", 30))
        keyframe_ids = _ensure_frames(
            input_video, info, all_frames, keyframes, key_stride
        )
        status["stages"]["frames"] = {
            "state": "complete",
            "all": info["frames"],
            "keyframes": len(keyframe_ids),
        }
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")

        key_masks = output_dir / "masks_keyframes_sam31"
        masks = output_dir / "masks"
        segmentation_manifest = output_dir / "segmentation_manifest.json"
        segmentation_cfg = dict(cfg["segmentation"])
        segmentation_cfg["prompts"] = resolve_prompts(
            segmentation_cfg,
            cfg.get("gpt", {}),
            keyframes.glob("*.jpg"),
        )
        segmentation_fingerprint = _fingerprint(input_video, segmentation_cfg)
        if force or not _cached(
            segmentation_manifest, segmentation_fingerprint, info["frames"], masks
        ):
            _clear_generated(key_masks)
            _clear_generated(masks)
            segmentation_report = SegmentationAdapter(
                segmentation_cfg, PROJECT_ROOT
            ).run(
                all_frames,
                keyframes,
                key_masks,
                masks,
                output_dir / "logs",
            )
            segmentation_manifest.write_text(
                json.dumps(
                    {
                        "fingerprint": segmentation_fingerprint,
                        "report": segmentation_report,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        else:
            segmentation_report = {
                "sam31": json.loads(
                    (key_masks / "sam31_report.json").read_text(encoding="utf-8")
                ),
                "sam2": json.loads(
                    (masks / "sam2_report.json").read_text(encoding="utf-8")
                ),
            }
            segmentation_manifest.write_text(
                json.dumps(
                    {
                        "fingerprint": segmentation_fingerprint,
                        "report": segmentation_report,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

        filled_hole_pixels = fill_enclosed_mask_holes(masks)
        mask_report = mask_statistics(masks, info["frames"])
        mask_report["filled_enclosed_hole_pixels"] = filled_hole_pixels
        if mask_report["files"] != info["frames"] or mask_report["mean_coverage"] <= 0:
            raise RuntimeError(f"Invalid foreground masks: {mask_report}")
        status["stages"]["segmentation"] = {
            "state": "complete",
            **mask_report,
        }
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
        write_mask_overlay(input_video, masks, output_dir / "mask_overlay.mp4")
        if stop_after == "segmentation":
            status["state"] = "stopped_after_segmentation"
            status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
            print(f"Segmentation complete: {output_dir}")
            return

        slam_dir = output_dir / "slam"
        slam_result = SLAMAdapter(cfg["slam"], PROJECT_ROOT).run(
            all_frames, slam_dir, masks
        )
        import open3d as o3d

        cloud = o3d.io.read_point_cloud(str(slam_result["pointcloud"]))
        points = np.asarray(cloud.points)
        colors = np.asarray(cloud.colors)
        if not len(points):
            raise RuntimeError("VGGT returned an empty background point cloud")
        save_pointcloud(points, output_dir / "pointcloud_background.ply", colors)
        status["stages"]["reconstruction"] = {
            "state": "complete",
            **slam_result["report"],
        }
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
        if stop_after == "reconstruction":
            status["state"] = "stopped_after_reconstruction"
            status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
            print(f"Reconstruction complete: {output_dir}")
            return

        geometry_cfg = cfg.get("geometry", {})
        opening_hints = _load_opening_hints(output_dir, cfg, all_frames, slam_result)
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
        status["stages"]["geometry"] = {
            "state": "complete",
            "opening_hints": len(opening_hints) if opening_hints else 0,
            **geometry_report,
        }
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
        if stop_after == "geometry":
            status["state"] = "stopped_after_geometry"
            status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
            print(f"Geometry complete: {output_dir}")
            return

        completion_cfg = cfg.get("video_completion", {})
        try:
            if completion_cfg.get("backend", "propainter") != "propainter":
                raise RuntimeError("ProPainter disabled by configuration")
            video_report = ProPainterAdapter(completion_cfg, PROJECT_ROOT).run(
                input_video,
                all_frames,
                masks,
                output_dir / "background_video.mp4",
                info["fps"],
            )
        except Exception as error:
            if not completion_cfg.get("allow_temporal_fallback", True):
                raise
            video_report = write_background_video(
                input_video,
                all_frames,
                masks,
                output_dir / "background_video.mp4",
                dilation=cfg["segmentation"].get("mask_dilation_px", 7),
                temporal_offsets=completion_cfg.get(
                    "temporal_offsets", [15, -15, 30, -30, 60, -60]
                ),
            )
            video_report["propainter_error"] = str(error)
        status["stages"]["video"] = {"state": "complete", **video_report}
        status["state"] = "complete"
        status["elapsed_seconds"] = time.time() - started
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
        print(f"Pipeline complete: {output_dir}")
    except Exception as error:
        status["state"] = "failed"
        status["error"] = f"{type(error).__name__}: {error}"
        status["elapsed_seconds"] = time.time() - started
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
        raise


def doctor(cfg):
    checks = []
    required_files = [
        cfg["input_video"],
        cfg["segmentation"]["sam3_checkpoint"],
        cfg["segmentation"]["sam2_checkpoint"],
        cfg["slam"]["checkpoint"],
        "checkpoints/propainter/ProPainter.pth",
        "checkpoints/propainter/recurrent_flow_completion.pth",
        "checkpoints/propainter/raft-things.pth",
    ]
    for value in required_files:
        path = (PROJECT_ROOT / value).resolve()
        checks.append({"check": str(path), "ok": path.exists()})
    if cfg["slam"].get("backend") == "vggt_slam":
        for value in (
            cfg["slam"].get(
                "salad_checkpoint", "checkpoints/salad/dino_salad.ckpt"
            ),
            cfg["slam"].get(
                "dinov2_checkpoint", "checkpoints/dinov2/dinov2_vitb14_pretrain.pth"
            ),
        ):
            path = (PROJECT_ROOT / value).resolve()
            checks.append({"check": str(path), "ok": path.exists()})
    env_commands = {
        cfg["segmentation"].get("environment", "vbr-seg"): (
            "import av,cv2,scipy,skimage,torch,sam3,sam2; import sam2._C; "
            "assert torch.cuda.is_available(); print(torch.__version__)"
        ),
        cfg["slam"].get("environment", "vbr-slam"): (
            "import torch,vggt,vggt_slam,gtsam,open3d,salad; "
            "assert torch.cuda.is_available() and hasattr(gtsam,'SL4'); print(torch.__version__)"
        ),
    }
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(
        cfg["segmentation"].get("cuda_visible_devices", "7")
    )
    for env_name, code in env_commands.items():
        result = subprocess.run(
            ["conda", "run", "-n", env_name, "python", "-c", code],
            cwd=PROJECT_ROOT,
            env=environment,
            text=True,
            capture_output=True,
        )
        checks.append(
            {
                "check": f"conda:{env_name}",
                "ok": result.returncode == 0,
                "detail": (result.stdout + result.stderr).strip()[-1000:],
            }
        )
    try:
        ffmpeg = _resolve_ffmpeg(cfg.get("video_completion", {}).get("ffmpeg_bin"))
        checks.append({"check": "ffmpeg:libx264", "ok": True, "detail": ffmpeg})
    except RuntimeError as error:
        checks.append({"check": "ffmpeg:libx264", "ok": False, "detail": str(error)})
    print(json.dumps(checks, indent=2))
    if not all(item["ok"] for item in checks):
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--config", default="configs/default.yaml")
    run_parser.add_argument(
        "--stop-after",
        choices=["segmentation", "reconstruction", "geometry", "all"],
        default="all",
    )
    run_parser.add_argument("--force", action="store_true")
    doctor_parser = subparsers.add_parser("doctor")
    doctor_parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    if args.command == "run":
        run(load_config(args.config), stop_after=args.stop_after, force=args.force)
    elif args.command == "doctor":
        doctor(load_config(args.config))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
