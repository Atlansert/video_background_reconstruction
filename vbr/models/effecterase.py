"""Adapter for EffectErase as an alternative inpainting backend.

EffectErase (FudanCVL/EffectErase, CVPR 2026, CC BY-NC 4.0) is a
Wan2.1-Fun-1.3B diffusion model fine-tuned to jointly erase objects and their
visual effects (shadows, reflections) — the shadow pass-through that motivated
`composite_source: false` on the SVOR backend is what this model removes by
design. Its inference entry (examples/remove_wan/infer_remove_wan.py) reads
exactly `--num_frames` frames from the START of a video + a mask video
(white = remove), so the chunking strategy is identical to SVOR's: synthesize
per-chunk videos, split into 4k+1-aligned chunks with overlap, run per chunk,
and blend-stitch.

Differences from SVOR handled here:
- inference goes through our wrapper `vbr/models/effecterase_infer.py`
  (upstream crashes when a chunk's first mask frame is empty);
- the model runs at a fixed resolution (default 544x960, both /16) and the
  final encode scales back to the source size (same as the SVOR tail);
- the post-processing defaults mirror the current SVOR recipe: no source
  composite (the model's whole point is removing shadows itself), flow-EMA
  stabilization in the fill and on the regenerated walls/floors.

Weight layout expected under `repo_dir/models/`:
- `Wan-AI/Wan2.1-Fun-1.3B-InP/`  base model (4 files used by the wrapper)
- `FudanCVL/EffectErase/EffectErase.ckpt`  method LoRA
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import cv2

from vbr.models.svor import (
    SVORAdapter,
    blend_stitch,
    dilate_masks,
    svor_chunk_ranges,
    _resolve_ffmpeg,
)


class EffectEraseAdapter(SVORAdapter):
    def __init__(self, cfg: dict, project_root: Path):
        nested = cfg.get("effecterase", {})
        merged = {**cfg, **nested} if nested else dict(cfg)
        # The video_completion section carries the sibling `svor:` block too;
        # dropping it keeps the base class's nested-merge from overriding the
        # effecterase values on shared keys (environment, repo_dir, ...).
        merged.pop("svor", None)
        super().__init__(merged, project_root)

    def run(self, video_path, frames_dir, masks_dir, output_path, fps):
        repo = self.project_root / self.cfg.get("repo_dir", "external/effecterase")
        wrapper = self.project_root / self.cfg.get(
            "infer_script", "vbr/models/effecterase_infer.py"
        )
        model_dir = repo / self.cfg.get("model_dir", "models/Wan-AI/Wan2.1-Fun-1.3B-InP")
        lora_path = repo / self.cfg.get(
            "lora_path", "models/FudanCVL/EffectErase/EffectErase.ckpt"
        )
        required = [
            wrapper,
            model_dir / "diffusion_pytorch_model.safetensors",
            model_dir / "models_t5_umt5-xxl-enc-bf16.pth",
            model_dir / "Wan2.1_VAE.pth",
            model_dir / "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
            lora_path,
        ]
        for path in required:
            if not path.exists():
                raise FileNotFoundError(path)

        work_root = Path(output_path).parent / "effecterase"
        work_root.mkdir(parents=True, exist_ok=True)
        probe = cv2.VideoCapture(str(video_path))
        size = (
            int(probe.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        capture_fps = probe.get(cv2.CAP_PROP_FPS)
        probe.release()
        fps_value = fps if fps else capture_fps
        frames, masks, _ = self._synthesize_inputs(frames_dir, masks_dir)
        total = len(frames)
        dilation_px = int(self.cfg.get("mask_dilation", 8))
        masks = dilate_masks(masks, dilation_px)
        pad = (4 - (total - 1) % 4) % 4  # pad the tail so the last chunk stays VAE-aligned
        if pad:
            frames += [frames[-1]] * pad
            masks += [masks[-1]] * pad
        ranges = list(
            svor_chunk_ranges(
                len(frames),
                int(self.cfg.get("chunk_frames", 81)),
                int(self.cfg.get("overlap", 20)),
            )
        )
        chunk_inputs = self._chunk_inputs(work_root, frames, masks, fps_value, ranges)

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(self.cfg.get("cuda_visible_devices", "5"))
        env["PYTHONUNBUFFERED"] = "1"
        environment_name = self.cfg.get("environment", "effecterase")
        height = int(self.cfg.get("height", 544))
        width = int(self.cfg.get("width", 960))
        chunk_outputs = []
        for chunk_index, (start, end, chunk_input, chunk_mask, chunk_dir) in enumerate(
            chunk_inputs
        ):
            produced = chunk_dir / "out.mp4"
            command = [
                "conda", "run", "--no-capture-output", "-n", environment_name,
                "python", str(wrapper.resolve()),
                "--fg_bg_path", str(chunk_input.resolve()),
                "--mask_path", str(chunk_mask.resolve()),
                "--output_path", str(produced.resolve()),
                "--model_dir", str(model_dir.resolve()),
                "--lora_path", str(lora_path.resolve()),
                "--num_frames", str(end - start),
                "--height", str(height),
                "--width", str(width),
                "--num_inference_steps", str(self.cfg.get("num_inference_steps", 50)),
                "--seed", str(self.cfg.get("seed", 2025)),
                "--cfg", str(self.cfg.get("cfg", 1.0)),
                "--lora_alpha", str(self.cfg.get("lora_alpha", 1.0)),
            ]
            if self.cfg.get("tiled", False):
                command.append("--tiled")
            log_path = (
                Path(output_path).parent / "logs" / f"effecterase_chunk_{chunk_index:03d}.log"
            )
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("w", encoding="utf-8") as log:
                result = subprocess.run(
                    command, cwd=self.project_root, env=env, stdout=log,
                    stderr=subprocess.STDOUT, text=True,
                )
            if result.returncode:
                tail = "\n".join(log_path.read_text(encoding="utf-8").splitlines()[-40:])
                raise RuntimeError(
                    f"EffectErase chunk {chunk_index} failed ({result.returncode}); "
                    f"see {log_path}\n{tail}"
                )
            if not produced.exists() or produced.stat().st_size == 0:
                raise RuntimeError(f"EffectErase did not create {produced}")
            chunk_outputs.append((start, end, produced))

        raw_output = work_root / "combined.mp4"
        stitched = blend_stitch(chunk_outputs, raw_output, fps_value)
        if stitched != len(frames):
            raise RuntimeError(f"Stitched {stitched} frames, expected {len(frames)}")

        # Same ordering as SVOR: stabilize the raw generation BEFORE any
        # compositing (an EMA after compositing drags real object edges into
        # the fill). Composite stays off by default: EffectErase's premise is
        # that it removes the shadows itself, so pass-through would re-paste
        # them; walls/floors get the light `wall_ema_alpha` stabilization.
        processed = raw_output
        ema_alpha = float(self.cfg.get("fill_ema_alpha", 0.65))
        outside_alpha = 0.0
        if not self.cfg.get("composite_source", False):
            outside_alpha = float(self.cfg.get("wall_ema_alpha", 0.3))
        if ema_alpha > 0 or outside_alpha > 0:
            stabilized = work_root / "stabilized.mp4"
            self._flow_ema(
                processed, masks, stabilized, fps_value, total, ema_alpha, outside_alpha
            )
            processed = stabilized
        if self.cfg.get("composite_source", False):
            composited = work_root / "composited.mp4"
            self._composite_source(processed, frames, masks, composited, fps_value, total)
            processed = composited
        if (self.cfg.get("temporal_smooth") or {}).get("enabled", False):
            smoothed = work_root / "smoothed.mp4"
            smooth_masks = masks_dir
            if dilation_px > 0:
                smooth_masks = work_root / "masks_dilated"
                smooth_masks.mkdir(parents=True, exist_ok=True)
                for index in range(total):
                    cv2.imwrite(str(smooth_masks / f"{index:06d}.png"), masks[index])
            self._temporal_smooth(processed, smooth_masks, smoothed, fps_value)
            processed = smoothed

        output_path = Path(output_path)
        encode = [
            _resolve_ffmpeg(self.base_cfg.get("ffmpeg_bin")),
            "-y", "-loglevel", "error",
            "-i", str(processed),
            "-i", str(Path(video_path).resolve()),
            "-map", "0:v:0", "-map", "1:a?",
            "-vf", f"scale={size[0]}:{size[1]}",
            "-c:v", "libx264",
            "-crf", str(self.cfg.get("crf", 18)),
            "-preset", self.cfg.get("preset", "medium"),
            "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-shortest",
            "-frames:v", str(total),  # drop the alignment padding frames
            str(output_path),
        ]
        subprocess.run(encode, cwd=self.project_root, check=True)
        report = {
            "method": "effecterase",
            "environment": environment_name,
            "chunks": len(ranges),
            "chunk_frames": int(self.cfg.get("chunk_frames", 81)),
            "overlap": int(self.cfg.get("overlap", 20)),
            "resolution": f"{height}x{width}",
            "num_inference_steps": int(self.cfg.get("num_inference_steps", 50)),
            "output": str(output_path.resolve()),
            "raw_output": str(raw_output.resolve()),
        }
        output_path.with_suffix(".json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        return report
