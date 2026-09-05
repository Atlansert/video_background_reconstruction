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
        command = [
            "conda",
            "run",
            "--no-capture-output",
            "-n",
            self.cfg.get("environment", "vbr-seg"),
            "python",
            "inference_propainter.py",
            "--video",
            str(Path(frames_dir).resolve()),
            "--mask",
            str(Path(masks_dir).resolve()),
            "--output",
            str(result_root.resolve()),
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
        ]
        if self.cfg.get("fp16", True):
            command.append("--fp16")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(self.cfg.get("cuda_visible_devices", "7"))
        env["PYTHONUNBUFFERED"] = "1"
        log_path = Path(output_path).parent / "logs" / "propainter.log"
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
            raise RuntimeError(f"ProPainter failed ({result.returncode}); see {log_path}\n{tail}")

        raw_output = result_root / Path(frames_dir).name / "inpaint_out.mp4"
        if not raw_output.exists() or raw_output.stat().st_size == 0:
            raise RuntimeError(f"ProPainter did not create {raw_output}")
        output_path = Path(output_path)
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
            str(raw_output),
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
        }
        output_path.with_suffix(".json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        return report
