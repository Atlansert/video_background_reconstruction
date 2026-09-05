"""Subprocess adapter for SAM 3.1 detection and SAM2 propagation."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path


class SegmentationAdapter:
    def __init__(self, cfg: dict, project_root: Path):
        self.cfg = cfg
        self.project_root = Path(project_root).resolve()

    def _environment(self) -> tuple[str, dict, Path]:
        env_name = self.cfg.get("environment", "vbr-seg")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(self.cfg.get("cuda_visible_devices", "7"))
        env["PYTHONUNBUFFERED"] = "1"
        checkpoint = self.project_root / self.cfg.get(
            "sam3_checkpoint", "checkpoints/sam3.1/sam3.1_multiplex.pt"
        )
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        return env_name, env, checkpoint

    def run_opening_masks(
        self,
        frame_ids: list[int],
        all_frames_dir: Path,
        output_dir: Path,
        logs_dir: Path,
        prompts: list[str],
    ) -> Path:
        """Run preserve-only SAM 3.1 prompts on the reconstruction frames.

        Produces ``<output_dir>/preserved/<frame_id>.png`` masks for the fixed
        structures (door, window, stairs, cabinets, ...) that the geometry
        stage projects onto wall grids to carve openings and occluded cells.
        """
        env_name, env, checkpoint = self._environment()
        output_dir = output_dir.resolve()
        subset_dir = output_dir / "frames_subset"
        subset_dir.mkdir(parents=True, exist_ok=True)
        for frame_id in frame_ids:
            source = all_frames_dir / f"{frame_id:06d}.jpg"
            if not source.exists():
                raise FileNotFoundError(source)
            destination = subset_dir / source.name
            if not destination.exists():
                shutil.copy(source, destination)

        logs_dir.mkdir(parents=True, exist_ok=True)
        command = [
            "conda",
            "run",
            "--no-capture-output",
            "-n",
            env_name,
            "python",
            "-m",
            "vbr.sam31_keyframes",
            "--frames",
            str(subset_dir.resolve()),
            "--output",
            str(output_dir),
            "--checkpoint",
            str(checkpoint),
            "--prompts-json",
            "[]",
            "--preserve-prompts-json",
            json.dumps(prompts),
            "--prompt-thresholds-json",
            json.dumps(self.cfg.get("prompt_thresholds", {})),
            "--threshold",
            str(self.cfg.get("sam3_threshold", 0.45)),
            "--max-objects",
            str(self.cfg.get("sam3_max_objects", 64)),
            "--allow-empty-union",
        ]
        self._run_logged(command, logs_dir / "sam31_openings.log", env)
        return output_dir / "preserved"

    def run(
        self,
        all_frames_dir: Path,
        keyframes_dir: Path,
        key_masks_dir: Path,
        masks_dir: Path,
        logs_dir: Path,
    ) -> dict:
        backend = self.cfg.get("backend", "sam31_sam2")
        if backend != "sam31_sam2":
            raise ValueError(
                f"Unsupported production segmentation backend {backend!r}; "
                "use 'sam31_sam2'"
            )

        env_name, env, checkpoint = self._environment()
        sam2_checkpoint = self.project_root / self.cfg.get(
            "sam2_checkpoint", "checkpoints/sam2/sam2.1_hiera_large.pt"
        )
        if not sam2_checkpoint.exists():
            raise FileNotFoundError(sam2_checkpoint)

        logs_dir.mkdir(parents=True, exist_ok=True)

        detect_cmd = [
            "conda",
            "run",
            "--no-capture-output",
            "-n",
            env_name,
            "python",
            "-m",
            "vbr.sam31_keyframes",
            "--frames",
            str(keyframes_dir.resolve()),
            "--output",
            str(key_masks_dir.resolve()),
            "--checkpoint",
            str(checkpoint),
            "--prompts-json",
            json.dumps(self.cfg.get("prompts", [])),
            "--preserve-prompts-json",
            json.dumps(self.cfg.get("preserve_prompts", [])),
            "--prompt-thresholds-json",
            json.dumps(self.cfg.get("prompt_thresholds", {})),
            "--threshold",
            str(self.cfg.get("sam3_threshold", 0.45)),
            "--max-objects",
            str(self.cfg.get("sam3_max_objects", 64)),
        ]
        self._run_logged(detect_cmd, logs_dir / "sam31.log", env)

        propagate_cmd = [
            "conda",
            "run",
            "--no-capture-output",
            "-n",
            env_name,
            "python",
            "-m",
            "vbr.sam2_propagate",
            "--frames",
            str(all_frames_dir.resolve()),
            "--key-masks",
            str(key_masks_dir.resolve()),
            "--output",
            str(masks_dir.resolve()),
            "--checkpoint",
            str(sam2_checkpoint),
            "--min-area",
            str(self.cfg.get("min_area_px", 100)),
        ]
        if self.cfg.get("sam2_offload_state", False):
            propagate_cmd.append("--offload-state")
        self._run_logged(propagate_cmd, logs_dir / "sam2.log", env)

        return {
            "sam31": json.loads((key_masks_dir / "sam31_report.json").read_text()),
            "sam2": json.loads((masks_dir / "sam2_report.json").read_text()),
        }

    def _run_logged(self, command: list[str], log_path: Path, env: dict) -> None:
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
            raise RuntimeError(f"Command failed ({result.returncode}); see {log_path}\n{tail}")
