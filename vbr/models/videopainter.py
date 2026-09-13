"""Adapter for VideoPainter (TencentARC, SIGGRAPH 2025) as an alternative inpainting backend.

VideoPainter is a dual-branch CogVideoX-5b-I2V pipeline: a lightweight context-control
branch conditions generation on the masked video, and an ID-resample LoRA enables
any-length input (the pipeline windows the clip internally and averages overlapping
window latents). Unlike SVOR — which regenerates whole frames and needs source
compositing — its `replace_gt` mode locks latents OUTSIDE the mask to the noised
ground truth at every denoising step, giving latent-level pass-through of walls and
floors for free.

CogVideoX is an 8fps 720x480 model, so the driver temporally downsamples the clip
(`down_sample_stride`), generates at model resolution, and this adapter rescales the
result back to source fps/size (`upsample: minterpolate` motion-compensates; the
geometry/evaluation contract expects 1799 frames @ source fps). The dedicated conda
env (repo-bundled diffusers fork) is addressed by NAME or absolute prefix path and
the driver runs via `conda run`; weights live under `external/videopainter/ckpt/`.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import cv2

from vbr.models.inpainting import _resolve_ffmpeg


def window_pad_count(model_frames, window_frames, window_stride):
    """Repeat-last-frame padding so the pipeline's internal windowing covers
    every frame.

    n_windows = ceil((M - window) / stride) + 1; the last window covers
    (n_windows - 1) * stride + window, which must reach M. Clipped clips
    shorter than one window pad up to a full window (the driver trims the
    repeats afterwards).
    """
    if model_frames <= 0:
        raise ValueError(f"Invalid model frame count {model_frames}")
    if model_frames <= window_frames:
        return max(window_frames - model_frames, 0)
    n_windows = -(-(model_frames - window_frames) // window_stride) + 1
    covered = (n_windows - 1) * window_stride + window_frames
    return covered - model_frames


class VideoPainterAdapter:
    def __init__(self, cfg: dict, project_root: Path):
        # The video_completion section carries the shared keys; backend-specific
        # values live in a nested `videopainter:` block and win on conflict.
        self.base_cfg = cfg
        nested = cfg.get("videopainter", {})
        self.cfg = {**cfg, **nested} if nested else cfg
        self.project_root = Path(project_root).resolve()

    def _count_frames(self, frames_dir, masks_dir):
        frame_paths = sorted(Path(frames_dir).glob("*.jpg"))
        mask_paths = sorted(Path(masks_dir).glob("*.png"))
        if not frame_paths:
            raise RuntimeError(f"No frames in {frames_dir}")
        if len(mask_paths) != len(frame_paths):
            raise RuntimeError(
                f"Mask/frame count mismatch: {len(mask_paths)} masks vs {len(frame_paths)} frames"
            )
        return frame_paths, mask_paths

    def run(self, video_path, frames_dir, masks_dir, output_path, fps):
        repo = self.project_root / self.cfg.get("repo_dir", "external/videopainter")
        ckpt = repo / "ckpt"
        model_path = self.project_root / self.cfg.get(
            "model_path", "external/videopainter/ckpt/CogVideoX-5b-I2V"
        )
        branch_path = self.project_root / self.cfg.get(
            "branch_path",
            "external/videopainter/ckpt/VideoPainter/VideoPainter/checkpoints/branch",
        )
        id_adapter_path = self.project_root / self.cfg.get(
            "id_adapter_path",
            "external/videopainter/ckpt/VideoPainter/VideoPainterID/checkpoints",
        )
        required = [model_path, branch_path / "config.json", id_adapter_path]
        for path in required:
            if not path.exists():
                raise FileNotFoundError(path)

        probe = cv2.VideoCapture(str(video_path))
        size = (int(probe.get(cv2.CAP_PROP_FRAME_WIDTH)), int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        capture_fps = probe.get(cv2.CAP_PROP_FPS)
        probe.release()
        fps_value = float(fps) if fps else capture_fps
        total = len(self._count_frames(frames_dir, masks_dir)[0])

        work_root = Path(output_path).parent / "videopainter"
        work_root.mkdir(parents=True, exist_ok=True)
        raw_output = work_root / "generated_modelfps.mp4"
        report_path = work_root / "driver_report.json"

        stride = max(1, int(self.cfg.get("down_sample_stride", 3)))
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(self.cfg.get("cuda_visible_devices", "4"))
        env["PYTHONPATH"] = str(self.project_root)
        env["PYTHONUNBUFFERED"] = "1"
        environment_name = self.cfg.get("environment", "videopainter")
        conda_flag = "-p" if "/" in environment_name else "-n"
        command = [
            "conda", "run", "--no-capture-output", conda_flag, environment_name,
            "python", "-m", "vbr.videopainter_driver",
            "--frames_dir", str(Path(frames_dir).resolve()),
            "--masks_dir", str(Path(masks_dir).resolve()),
            "--output", str(raw_output.resolve()),
            "--report", str(report_path.resolve()),
            "--model_path", str(model_path.resolve()),
            "--branch_path", str(branch_path.resolve()),
            "--id_adapter_path", str(id_adapter_path.resolve()),
            "--prompt", str(self.cfg.get("prompt", "an empty kitchen interior with white walls and a wooden floor")),
            "--source_fps", str(fps_value),
            "--down_sample_stride", str(stride),
            "--height", str(self.cfg.get("height", 480)),
            "--width", str(self.cfg.get("width", 720)),
            "--window_frames", str(self.cfg.get("window_frames", 49)),
            "--window_stride", str(self.cfg.get("window_stride", 49)),
            "--num_inference_steps", str(self.cfg.get("num_inference_steps", 50)),
            "--guidance_scale", str(self.cfg.get("guidance_scale", 6.0)),
            "--prev_clip_weight", str(self.cfg.get("prev_clip_weight", 0.5)),
            "--seed", str(self.cfg.get("seed", 42)),
            "--mask_dilate_px", str(self.cfg.get("mask_dilate_px", 16)),
            "--fill_mode", str(self.cfg.get("fill_mode", "telea")),
            "--first_frame_fill", str(self.cfg.get("first_frame_fill", "telea")),
            "--dtype", str(self.cfg.get("dtype", "bfloat16")),
        ]
        log_path = work_root / "driver.log"
        if not (
            self.cfg.get("resume", True)
            and raw_output.exists()
            and raw_output.stat().st_size > 0
            and report_path.exists()
        ):
            with log_path.open("w", encoding="utf-8") as log:
                result = subprocess.run(
                    command, cwd=self.project_root, env=env,
                    stdout=log, stderr=subprocess.STDOUT, text=True,
                )
            if result.returncode:
                tail = "\n".join(log_path.read_text(encoding="utf-8").splitlines()[-60:])
                raise RuntimeError(f"videopainter driver failed ({result.returncode}); see {log_path}\n{tail}")
            if not raw_output.exists() or raw_output.stat().st_size == 0:
                raise RuntimeError(f"driver produced no video, see {log_path}")

        # Rescale to the source contract: size + fps + frame count. minterpolate
        # motion-compensates the model's low-fps output back to source fps.
        upsample = self.cfg.get("upsample", "minterpolate")
        ffmpeg = _resolve_ffmpeg(self.base_cfg.get("ffmpeg_bin"))
        final = [
            ffmpeg, "-y", "-loglevel", "error",
            "-i", str(raw_output.resolve()),
            "-i", str(Path(video_path).resolve()),
            "-map", "0:v:0", "-map", "1:a?",
        ]
        if upsample == "minterpolate":
            final += [
                "-vf",
                f"minterpolate=fps={fps_value}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1,"
                f"tpad=stop_mode=clone:stop={total},scale={size[0]}:{size[1]}",
            ]
        elif upsample == "duplicate":
            final += [
                "-vf",
                f"minterpolate=fps={fps_value}:mi_mode=dup,tpad=stop_mode=clone:stop={total},scale={size[0]}:{size[1]}",
            ]
        else:
            final += [
                "-vf",
                f"tpad=stop_mode=clone:stop={total},scale={size[0]}:{size[1]}",
                "-r", str(fps_value),
            ]
        final += [
            "-c:v", "libx264",
            "-crf", str(self.cfg.get("crf", 18)),
            "-preset", self.cfg.get("preset", "medium"),
            "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-shortest",
            "-frames:v", str(total),  # drop the window-padding repeats
            str(output_path),
        ]
        subprocess.run(final, cwd=self.project_root, check=True)

        report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
        report.update(
            {
                "method": "videopainter",
                "environment": environment_name,
                "down_sample_stride": stride,
                "upsample": upsample,
                "output": str(Path(output_path).resolve()),
                "raw_output": str(raw_output.resolve()),
            }
        )
        Path(output_path).with_suffix(".json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        return report
