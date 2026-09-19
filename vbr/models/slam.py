"""Subprocess adapter for masked VGGT reconstruction backends.

Two backends are supported:
- ``vggt_direct``: one-shot VGGT prediction over uniformly sampled frames.
- ``vggt_slam``: the full VGGT-SLAM solver (optical-flow keyframing,
  submaps, SL(4) pose graph optimization, loop closure).

Both run inside the vbr-slam environment and export the same NPZ/PLY
interface consumed by the geometry stage.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np


class SLAMAdapter:
    def __init__(self, cfg: dict, project_root: Path):
        self.cfg = cfg
        self.project_root = Path(project_root).resolve()

    def _command(
        self,
        module: str,
        frame_dir: Path,
        masks_dir: Path,
        pointcloud: Path,
        checkpoint: Path,
    ) -> list:
        command = [
            "conda",
            "run",
            "--no-capture-output",
            "-n",
            self.cfg.get("environment", "vbr-slam"),
            "python",
            "-m",
            module,
            "--frames",
            str(frame_dir.resolve()),
            "--masks",
            str(masks_dir.resolve()),
            "--output",
            str(pointcloud.resolve()),
            "--checkpoint",
            str(checkpoint.resolve()),
            "--confidence-percentile",
            str(self.cfg.get("confidence_percentile", 35)),
            "--pixel-stride",
            str(self.cfg.get("pixel_stride", 2)),
            "--voxel-size",
            str(self.cfg.get("voxel_size", 0.025)),
            "--metric-room-height",
            str(self.cfg.get("metric_room_height", 2.6)),
        ]
        if module == "vbr.vggt_direct":
            command += ["--max-frames", str(self.cfg.get("max_frames", 32))]
        else:
            command += [
                "--frame-stride",
                str(self.cfg.get("frame_stride", 6)),
                "--max-keyframes",
                str(self.cfg.get("max_keyframes", 240)),
                "--submap-size",
                str(self.cfg.get("submap_size", 16)),
                "--overlap-size",
                str(self.cfg.get("overlap_size", 1)),
                "--min-disparity",
                str(self.cfg.get("min_disparity", 50)),
                "--conf-threshold",
                str(self.cfg.get("conf_threshold", 25)),
                "--lc-thres",
                str(self.cfg.get("lc_thres", 0.95)),
                "--max-loops",
                str(self.cfg.get("max_loops", 1)),
                "--model-mode",
                str(self.cfg.get("model_mode", "square")),
                "--mask-aware-matching"
                if self.cfg.get("mask_aware_matching", True)
                else "--no-mask-aware-matching",
            ]
            for flag, key, fallback in (
                (
                    "--salad-checkpoint",
                    "salad_checkpoint",
                    "checkpoints/salad/dino_salad.ckpt",
                ),
                (
                    "--dinov2-checkpoint",
                    "dinov2_checkpoint",
                    "checkpoints/dinov2/dinov2_vitb14_pretrain.pth",
                ),
            ):
                value = self.cfg.get(key, fallback)
                if value:
                    path = (self.project_root / value).resolve()
                    if path.exists():
                        command += [flag, str(path)]
        return command

    def run(self, frame_dir: Path, output_dir: Path, masks_dir: Path) -> dict:
        backend = self.cfg.get("backend", "vggt_direct")
        if backend == "vggt_direct":
            module = "vbr.vggt_direct"
        elif backend == "vggt_slam":
            module = "vbr.vggt_slam_backend"
        else:
            raise ValueError(
                f"Unsupported reconstruction backend {backend!r}; "
                "use 'vggt_direct' or 'vggt_slam'"
            )

        output_dir.mkdir(parents=True, exist_ok=True)
        pointcloud = output_dir / "points_background.ply"
        checkpoint = self.project_root / self.cfg.get(
            "checkpoint", "checkpoints/vggt/model.pt"
        )
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)

        command = self._command(module, frame_dir, masks_dir, pointcloud, checkpoint)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(self.cfg.get("cuda_visible_devices", "7"))
        env["PYTHONUNBUFFERED"] = "1"
        log_path = output_dir / f"{backend}.log"
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(
                command,
                cwd=self.project_root,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
        if result.returncode:
            tail = "\n".join(log_path.read_text(encoding="utf-8").splitlines()[-30:])
            raise RuntimeError(
                f"Reconstruction backend {backend} failed ({result.returncode}); "
                f"see {log_path}\n{tail}"
            )

        reconstruction = pointcloud.with_suffix(".npz")
        values = np.load(reconstruction)
        report = json.loads(pointcloud.with_suffix(".json").read_text(encoding="utf-8"))
        return {
            "pointcloud": pointcloud,
            "reconstruction": reconstruction,
            "extrinsics": values["extrinsics"],
            "intrinsics": values["intrinsics"],
            "report": report,
        }
