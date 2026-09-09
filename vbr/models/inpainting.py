"""Adapter for ProPainter video completion with audio-preserving final encoding."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import cv2


def _resolve_ffmpeg(configured: str | None = None) -> str:
    candidates = [configured, "/usr/bin/ffmpeg", shutil.which("ffmpeg")]
    checked = set()
    for candidate in candidates:
        if not candidate or candidate in checked:
            continue
        checked.add(candidate)
        try:
            result = subprocess.run(
                [candidate, "-hide_banner", "-encoders"],
                check=False,
                capture_output=True,
                text=True,
            )
        except OSError:
            continue
        if result.returncode == 0 and "libx264" in result.stdout:
            return candidate
    raise RuntimeError("No FFmpeg binary with the libx264 encoder is available")


def chunk_ranges(total, size, overlap):
    """Yield (start, end) frame-index ranges covering [0, total) with overlap.

    Each chunk except the last includes ``overlap`` trailing frames that the
    next chunk re-processes; the stitcher prefers the later chunk in overlap
    regions so every output frame has as much future context as possible.
    """
    if size <= overlap or size <= 0 or overlap < 0:
        raise ValueError(f"Invalid chunk size/overlap ({size}/{overlap})")
    current = 0
    while current < total:
        end = min(total, current + size)
        yield current, end
        if end == total:
            break
        current = end - overlap


def _reindexed_copy(source_dir: Path, target_dir: Path, start: int, end: int, suffix: str):
    """Copy ``source_dir`` frames [start, end) into a freshly reindexed dir.

    The target is wiped first: mask PNGs of equal byte size but different
    content (and chunk-size changes) otherwise turn into stale inputs on
    re-runs.
    """
    shutil.rmtree(target_dir, ignore_errors=True)
    target_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(source_dir.glob(f"*{suffix}"), key=lambda p: int(p.stem))
    for local, path in enumerate(paths[start:end]):
        shutil.copy2(path, target_dir / f"{local:06d}{suffix}")


def _stitch_chunks(chunk_outputs: list[tuple[int, int, Path]], output_path: Path, fps: float):
    """Merge per-chunk inpaint_out.mp4 files, preferring later-chunk frames."""
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
        capture = cv2.VideoCapture(str(video_path))
        frames = []
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
        capture.release()
        buffers[start] = frames
    for global_index in range(total):
        frame = None
        for start, end, _ in chunk_outputs:
            if start <= global_index < end:
                local = global_index - start
                if local < len(buffers[start]):
                    frame = buffers[start][local]
        if frame is None:
            raise RuntimeError(f"No chunk produced frame {global_index}")
        writer.write(frame)
    writer.release()
    return total


class ProPainterAdapter:
    def __init__(self, cfg: dict, project_root: Path):
        self.cfg = cfg
        self.project_root = Path(project_root).resolve()

    def run(self, video_path, frames_dir, masks_dir, output_path, fps):
        repo = self.project_root / self.cfg.get("repo_dir", "external/ProPainter")
        required = [
            repo / "inference_propainter.py",
            repo / "weights" / "ProPainter.pth",
            repo / "weights" / "recurrent_flow_completion.pth",
            repo / "weights" / "raft-things.pth",
        ]
        for path in required:
            if not path.exists():
                raise FileNotFoundError(path)

        process_width = int(self.cfg.get("width", 960))
        process_height = int(self.cfg.get("height", 536))
        result_root = Path(output_path).parent / "propainter"

        def build_command(video_arg: Path, masks_arg: Path, output_arg: Path) -> list:
            return [
                "conda",
                "run",
                "--no-capture-output",
                "-n",
                self.cfg.get("environment", "vbr-seg"),
                "python",
                "inference_propainter.py",
                "--video",
                str(video_arg.resolve()),
                "--mask",
                str(masks_arg.resolve()),
                "--output",
                str(output_arg.resolve()),
                "--width",
                str(process_width),
                "--height",
                str(process_height),
                "--save_fps",
                str(fps),
                "--mask_dilation",
                str(self.cfg.get("mask_dilation", 5)),
                "--ref_stride",
                str(self.cfg.get("ref_stride", 10)),
                "--neighbor_length",
                str(self.cfg.get("neighbor_length", 10)),
                "--subvideo_length",
                str(self.cfg.get("subvideo_length", 50)),
                "--raft_iter",
                str(self.cfg.get("raft_iter", 20)),
            ] + (["--fp16"] if self.cfg.get("fp16", True) else [])

        def run_inference(video_arg: Path, masks_arg: Path, output_arg: Path, log_name: str) -> Path:
            command = build_command(video_arg, masks_arg, output_arg)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(self.cfg.get("cuda_visible_devices", "7"))
            env["PYTHONUNBUFFERED"] = "1"
            log_path = Path(output_path).parent / "logs" / log_name
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("w", encoding="utf-8") as log:
                result = subprocess.run(
                    command,
                    cwd=repo,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            if result.returncode:
                tail = "\n".join(log_path.read_text(encoding="utf-8").splitlines()[-40:])
                raise RuntimeError(
                    f"ProPainter failed ({result.returncode}); see {log_path}\n{tail}"
                )
            produced = output_arg / Path(video_arg).name / "inpaint_out.mp4"
            if not produced.exists() or produced.stat().st_size == 0:
                raise RuntimeError(f"ProPainter did not create {produced}")
            return produced

        chunk_cfg = self.cfg.get("chunks", {})
        raw_output = None
        chunk_report = {"enabled": False}
        if chunk_cfg.get("enabled", False):
            frame_paths = sorted(
                Path(frames_dir).glob("*.jpg"), key=lambda p: int(p.stem)
            )
            ranges = list(
                chunk_ranges(
                    len(frame_paths),
                    int(chunk_cfg.get("size", 240)),
                    int(chunk_cfg.get("overlap", 60)),
                )
            )
            work_root = result_root / "chunks"
            shutil.rmtree(work_root, ignore_errors=True)
            raw_output = result_root / "combined.mp4"
            chunk_outputs = []
            for chunk_index, (start, end) in enumerate(ranges):
                chunk_frames = work_root / f"frames_{chunk_index:03d}"
                chunk_masks = work_root / f"masks_{chunk_index:03d}"
                _reindexed_copy(Path(frames_dir), chunk_frames, start, end, ".jpg")
                _reindexed_copy(Path(masks_dir), chunk_masks, start, end, ".png")
                chunk_output = run_inference(
                    chunk_frames,
                    chunk_masks,
                    work_root / f"out_{chunk_index:03d}",
                    f"propainter_chunk_{chunk_index:03d}.log",
                )
                chunk_outputs.append((start, end, chunk_output))
            stitched = _stitch_chunks(chunk_outputs, raw_output, fps)
            chunk_report = {
                "enabled": True,
                "chunks": len(ranges),
                "size": int(chunk_cfg.get("size", 240)),
                "overlap": int(chunk_cfg.get("overlap", 60)),
                "stitched_frames": stitched,
                "neighbor_length": int(self.cfg.get("neighbor_length", 40)),
            }
        else:
            # ProPainter writes <output>/<video-dir-name>/inpaint_out.mp4, so
            # pass result_root itself to get the flat frames_all/ layout.
            raw_output = run_inference(
                Path(frames_dir), Path(masks_dir), result_root, "propainter.log"
            )
        raw_output = Path(raw_output)
        output_path = Path(output_path)
        final_source = raw_output
        smooth_report = {}
        smooth_cfg = self.cfg.get("temporal_smooth", {})
        if smooth_cfg.get("enabled", False):
            smoothed_path = raw_output.with_name("smooth_out.mp4")
            smooth_env = os.environ.copy()
            smooth_env["CUDA_VISIBLE_DEVICES"] = str(
                self.cfg.get("cuda_visible_devices", "7")
            )
            smooth_env["PYTHONPATH"] = str(self.project_root)
            smooth_env["PYTHONUNBUFFERED"] = "1"
            smooth_log = Path(output_path).parent / "logs" / "temporal_smooth.log"
            smooth_log.parent.mkdir(parents=True, exist_ok=True)
            smooth_command = [
                "conda",
                "run",
                "--no-capture-output",
                "-n",
                self.cfg.get("environment", "vbr-seg"),
                "python",
                "-m",
                "vbr.temporal_smooth",
                "--input",
                str(raw_output.resolve()),
                "--masks",
                str(Path(masks_dir).resolve()),
                "--output",
                str(smoothed_path.resolve()),
                "--fps",
                str(fps),
                "--repo-dir",
                str(repo.resolve()),
                "--raft-checkpoint",
                str((repo / "weights" / "raft-things.pth").resolve()),
                "--log",
                str(smooth_log.resolve()),
            ]
            with smooth_log.open("w", encoding="utf-8") as log:
                smooth_result = subprocess.run(
                    smooth_command,
                    cwd=self.project_root,
                    env=smooth_env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            if smooth_result.returncode:
                raise RuntimeError(
                    f"temporal_smooth failed ({smooth_result.returncode}); "
                    f"see {smooth_log}"
                )
            final_source = smoothed_path
            smooth_report = {"enabled": True, "output": str(smoothed_path.resolve())}
        capture = cv2.VideoCapture(str(video_path))
        output_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        output_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        capture.release()
        encode = [
            _resolve_ffmpeg(self.cfg.get("ffmpeg_bin")),
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(final_source),
            "-i",
            str(Path(video_path).resolve()),
            "-map",
            "0:v:0",
            "-map",
            "1:a?",
            "-vf",
            f"scale={output_width}:{output_height}",
            "-c:v",
            "libx264",
            "-crf",
            str(self.cfg.get("crf", 18)),
            "-preset",
            self.cfg.get("preset", "medium"),
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(output_path),
        ]
        subprocess.run(encode, cwd=self.project_root, check=True)
        report = {
            "method": "propainter",
            "environment": self.cfg.get("environment", "vbr-seg"),
            "processing_resolution": [process_width, process_height],
            "output": str(output_path.resolve()),
            "raw_output": str(raw_output.resolve()),
            "audio_source": str(Path(video_path).resolve()),
            "chunks": chunk_report,
            "temporal_smooth": smooth_report,
        }
        output_path.with_suffix(".json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        return report
