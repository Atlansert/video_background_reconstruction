"""Adapter for SVOR (Stable Video Object Removal) as an alternative inpainting backend.

SVOR (xiaomi-research/svor, Apache-2.0) is a Wan2.1-VACE-1.3B diffusion model with
two LoRA stages. Its `predict_SVOR.py` accepts an input video plus a mask video
(white = remove) and processes AT MOST `--video_length` frames (81 default, must
be 4k+1 after VAE alignment) in one pass — longer inputs are truncated. This
adapter therefore synthesizes chunk videos from the frame directories,
splits the clip into aligned chunks with overlap, runs SVOR per chunk, and
blend-stitches the chunk outputs. Because the model regenerates the whole
frame (preserved regions drift and wobble during camera motion), the stitch
is composited back over the source with feathered masks and — when
`video_completion.temporal_smooth.enabled` — flow-aligned median-smoothed
inside the masks before the final h264+aac encode.

Weights layout expected under `repo_dir/models/`:
- `Wan2.1-VACE-1.3B/`  base model (original Wan checkpoint layout from Wan-AI/Wan2.1-VACE-1.3B)
- `remove_model_stage1.safetensors`, `remove_model_stage2.safetensors`  (LoRAs)
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import cv2
import numpy as np

from vbr.models.inpainting import _resolve_ffmpeg, chunk_ranges


def _write_video(frames, path, fps, size):
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), size
    )
    for frame in frames:
        writer.write(frame)
    writer.release()


def _read_video_frames(video_path):
    capture = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    return frames


def svor_chunk_ranges(total, size, overlap):
    """Chunk ranges where every chunk length is 1 (mod 4) for the Wan VAE.

    SVOR aligns the clip length down to the VAE temporal ratio (4) plus one,
    so a chunk of length 4k+1 uses every frame; arbitrary lengths would
    silently drop trailing frames. The tail chunk therefore extends BACK
    past the stride (a longer overlap, never a shorter chunk): shrinking a
    misaligned tail would step the next start backwards and loop forever.
    Callers should pad `total` to 1 (mod 4); unaligned totals still
    terminate, with the last chunk repeating its final frames.
    """
    if size <= overlap or size <= 0 or overlap < 0:
        raise ValueError(f"Invalid chunk size/overlap ({size}/{overlap})")
    current = 0
    while current < total:
        remaining = total - current
        if remaining <= size:
            length = remaining + (4 - (remaining - 1) % 4) % 4  # next 4k+1 >= remaining
            yield max(total - length, 0), total
            break
        yield current, current + size
        current += size - overlap


def blend_stitch(chunk_outputs, output_path, fps, fade_frames=12):
    """Stitch chunk mp4s with a linear cross-fade inside each overlap.

    Diffusion outputs do not match pixel-wise across chunks; a hard switch at
    boundaries pops, so the first `fade_frames` of each later chunk blend
    linearly from the previous chunk's frames.
    """
    capture = cv2.VideoCapture(str(chunk_outputs[0][2]))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    total = chunk_outputs[-1][1]
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )
    buffers = {}
    for start, end, video_path in chunk_outputs:
        buffers[start] = _read_video_frames(video_path)
    for global_index in range(total):
        owning = [entry for entry in chunk_outputs if entry[0] <= global_index < entry[1]]
        start, end, _ = owning[-1]  # later chunk owns the frame
        local = global_index - start
        frames = buffers[start]
        frame = frames[local] if local < len(frames) else frames[-1]
        if len(owning) > 1:
            previous_start, previous_end, _ = owning[-2]
            previous_local = global_index - previous_start
            previous_frames = buffers[previous_start]
            if previous_local < len(previous_frames):
                fade = min(fade_frames, max(1, end - start - local - 1))
                if local < fade and len(previous_frames) > previous_local:
                    alpha = (local + 1) / (fade + 1)
                    frame = cv2.addWeighted(
                        previous_frames[previous_local], 1 - alpha, frame, alpha, 0
                    )
        writer.write(frame)
    writer.release()
    return total


class SVORAdapter:
    def __init__(self, cfg: dict, project_root: Path):
        # The video_completion section carries the shared keys of the active
        # backend (repo_dir/environment/...); backend-specific values live in
        # a nested `svor:` block and win on conflict.
        self.base_cfg = cfg
        nested = cfg.get("svor", {})
        self.cfg = {**cfg, **nested} if nested else cfg
        self.project_root = Path(project_root).resolve()

    def _composite_source(self, generated_video, frames, masks, output_path, fps, total):
        """Blend source pixels back into unmasked regions (feathered alpha).

        SVOR regenerates the whole frame, so preserved structure (walls,
        floors) drifts from the real footage and visibly wobbles during
        camera motion; this restores pass-through semantics for everything
        outside the inpaint masks.
        """
        ffmpeg = _resolve_ffmpeg(self.base_cfg.get("ffmpeg_bin"))
        generated = _read_video_frames(generated_video)
        height, width = frames[0].shape[:2]
        process = subprocess.Popen(
            [
                ffmpeg, "-y", "-loglevel", "error",
                "-f", "rawvideo", "-pix_fmt", "bgr24",
                "-s", f"{width}x{height}", "-r", str(fps),
                "-i", "-",
                "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", "-an",
                str(output_path),
            ],
            stdin=subprocess.PIPE,
        )
        for index in range(total):
            mask = masks[index]
            if mask.ndim == 3:
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            alpha = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (7, 7), 2.5)
            alpha = alpha[:, :, None]
            blended = (
                frames[index].astype(np.float32) * (1.0 - alpha)
                + cv2.resize(generated[index], (width, height)).astype(np.float32) * alpha
            )
            process.stdin.write(np.clip(blended, 0, 255).astype(np.uint8).tobytes())
        process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError(f"composite encode failed, see {output_path}")

    def _temporal_smooth(self, input_video, masks_dir, output_path, fps):
        """Flow-aligned 3-frame median inside the masks (RAFT, vbr-seg env)."""
        repo = (self.project_root / self.base_cfg.get("repo_dir", "external/ProPainter")).resolve()
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(self.cfg.get("cuda_visible_devices", "7"))
        env["PYTHONPATH"] = str(self.project_root)
        env["PYTHONUNBUFFERED"] = "1"
        log_path = output_path.parent / "temporal_smooth.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(
                [
                    "conda", "run", "--no-capture-output", "-n",
                    self.base_cfg.get("environment", "vbr-seg"),
                    "python", "-m", "vbr.temporal_smooth",
                    "--input", str(input_video.resolve()),
                    "--masks", str(Path(masks_dir).resolve()),
                    "--output", str(output_path.resolve()),
                    "--fps", str(fps),
                    "--repo-dir", str(repo),
                    "--raft-checkpoint", str(repo / "weights" / "raft-things.pth"),
                    "--log", str(log_path.resolve()),
                ],
                cwd=self.project_root, env=env, stdout=log, stderr=subprocess.STDOUT, text=True,
            )
        if result.returncode:
            tail = "\n".join(log_path.read_text(encoding="utf-8").splitlines()[-40:])
            raise RuntimeError(f"temporal_smooth failed; see {log_path}\n{tail}")

    def _synthesize_inputs(self, frames_dir, masks_dir):
        """Read frames and masks once; chunk videos are written directly."""
        frame_paths = sorted(Path(frames_dir).glob("*.jpg"), key=lambda p: int(p.stem))
        frames = [cv2.imread(str(path)) for path in frame_paths]
        missing = [i for i, f in enumerate(frames) if f is None]
        if missing:
            raise RuntimeError(f"Unreadable frames in {frames_dir}: {missing[:5]}")
        size = (frames[0].shape[1], frames[0].shape[0])
        mask_paths = sorted(Path(masks_dir).glob("*.png"), key=lambda p: int(p.stem))
        if len(mask_paths) != len(frame_paths):
            raise RuntimeError(
                f"Mask/frame count mismatch: {len(mask_paths)} masks vs {len(frame_paths)} frames"
            )
        masks = []
        for path in mask_paths:
            mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            masks.append(mask)
        return frames, masks, size

    def _chunk_inputs(self, work_root, frames, masks, fps, ranges):
        chunk_inputs = []
        for chunk_index, (start, end) in enumerate(ranges):
            chunk_dir = work_root / f"chunk_{chunk_index:03d}"
            chunk_dir.mkdir(parents=True, exist_ok=True)
            chunk_input = chunk_dir / "input.mp4"
            chunk_mask = chunk_dir / "mask.mp4"
            height, width = frames[start].shape[:2]
            _write_video(frames[start:end], chunk_input, fps, (width, height))
            _write_video(masks[start:end], chunk_mask, fps, (width, height))
            chunk_inputs.append((start, end, chunk_input, chunk_mask, chunk_dir))
        return chunk_inputs

    def run(self, video_path, frames_dir, masks_dir, output_path, fps):
        repo = self.project_root / self.cfg.get("repo_dir", "external/svor")
        script = repo / "predict_SVOR.py"
        model_name = self.cfg.get("model_name", "models/Wan2.1-VACE-1.3B")
        lora_paths = self.cfg.get(
            "lora_paths",
            [
                "models/remove_model_stage1.safetensors",
                "models/remove_model_stage2.safetensors",
            ],
        )
        required = [script] + [repo / lora for lora in lora_paths] + [repo / model_name]
        for path in required:
            if not path.exists():
                raise FileNotFoundError(path)

        work_root = Path(output_path).parent / "svor"
        work_root.mkdir(parents=True, exist_ok=True)
        probe = cv2.VideoCapture(str(video_path))
        size = (int(probe.get(cv2.CAP_PROP_FRAME_WIDTH)), int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        capture_fps = probe.get(cv2.CAP_PROP_FPS)
        probe.release()
        fps_value = fps if fps else capture_fps
        frames, masks, _ = self._synthesize_inputs(frames_dir, masks_dir)
        total = len(frames)
        pad = (4 - (total - 1) % 4) % 4  # pad the tail so the last chunk stays VAE-aligned
        if pad:
            frames += [frames[-1]] * pad
            masks += [masks[-1]] * pad
        ranges = list(
            svor_chunk_ranges(
                len(frames),
                int(self.cfg.get("chunk_frames", 77)),
                int(self.cfg.get("overlap", 20)),
            )
        )
        chunk_inputs = self._chunk_inputs(work_root, frames, masks, fps_value, ranges)

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(self.cfg.get("cuda_visible_devices", "7"))
        env["PYTHONUNBUFFERED"] = "1"
        environment_name = self.cfg.get("environment", "svor")
        sample_size = self.cfg.get("sample_size", "540,960")
        chunk_outputs = []
        for chunk_index, (start, end, chunk_input, chunk_mask, chunk_dir) in enumerate(
            chunk_inputs
        ):
            save_dir = chunk_dir / "out"
            command = [
                "conda", "run", "--no-capture-output", "-n", environment_name,
                "python", "predict_SVOR.py",
                "--input_video", str(chunk_input.resolve()),
                "--input_mask_video", str(chunk_mask.resolve()),
                "--save_dir", str(save_dir.resolve()),
                "--model_name", model_name,
                "--lora_path",
            ] + [str(repo / lora) for lora in lora_paths] + [
                "--sample_size", sample_size,
                "--video_length", str(end - start),
                "--fps", str(int(round(float(fps)))),  # predict_SVOR.py takes --fps as int
                "--num_inference_steps", str(self.cfg.get("num_inference_steps", 20)),
                "--gpu_memory_mode", self.cfg.get("gpu_memory_mode", "model_full_load"),
                "--guidance_scale", str(self.cfg.get("guidance_scale", 6.0)),
                "--seed", str(self.cfg.get("seed", 43)),
                "--dilation", str(self.cfg.get("dilation", 0)),
                "--weight_dtype", self.cfg.get("weight_dtype", "bfloat16"),
            ]
            log_path = Path(output_path).parent / "logs" / f"svor_chunk_{chunk_index:03d}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("w", encoding="utf-8") as log:
                result = subprocess.run(
                    command, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT, text=True
                )
            if result.returncode:
                tail = "\n".join(log_path.read_text(encoding="utf-8").splitlines()[-40:])
                raise RuntimeError(
                    f"SVOR chunk {chunk_index} failed ({result.returncode}); see {log_path}\n{tail}"
                )
            produced = save_dir / chunk_input.name
            if not produced.exists() or produced.stat().st_size == 0:
                raise RuntimeError(f"SVOR did not create {produced}")
            chunk_outputs.append((start, end, produced))

        raw_output = work_root / "combined.mp4"
        stitched = blend_stitch(chunk_outputs, raw_output, fps_value)
        if stitched != len(frames):
            raise RuntimeError(f"Stitched {stitched} frames, expected {len(frames)}")

        processed = raw_output
        if self.cfg.get("composite_source", True):
            composited = work_root / "composited.mp4"
            self._composite_source(raw_output, frames, masks, composited, fps_value, total)
            processed = composited
        if (self.cfg.get("temporal_smooth") or {}).get("enabled", False):
            smoothed = work_root / "smoothed.mp4"
            self._temporal_smooth(processed, masks_dir, smoothed, fps_value)
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
            "method": "svor",
            "environment": environment_name,
            "chunks": len(ranges),
            "chunk_frames": int(self.cfg.get("chunk_frames", 77)),
            "overlap": int(self.cfg.get("overlap", 20)),
            "sample_size": sample_size,
            "num_inference_steps": int(self.cfg.get("num_inference_steps", 20)),
            "output": str(output_path.resolve()),
            "raw_output": str(raw_output.resolve()),
        }
        output_path.with_suffix(".json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        return report