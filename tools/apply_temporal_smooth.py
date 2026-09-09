"""Apply flow-aligned temporal smoothing to an existing background video.

Standalone V2 entry: takes the current background_video.mp4, aligns t-1/t+1
with RAFT and takes the 3-frame median inside the inpaint masks, then
re-encodes h264+aac with the original audio. The pre-smooth video is kept as
a snapshot so versions stay comparable.

Usage (vbr environment):
    python -m tools.apply_temporal_smooth --output-dir outputs/001_sam31_slam \
        --config configs/vggt_slam.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

from vbr.cli import PROJECT_ROOT, _resolve_ffmpeg
from vbr.config import load_config
from vbr.models.inpainting import ProPainterAdapter
from vbr.video import video_info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--config", default="configs/vggt_slam.yaml")
    parser.add_argument(
        "--keep-version",
        action="store_true",
        help="keep the pre-smooth video as <output-dir>/background_video_unsmoothed.mp4",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = args.output_dir.resolve()
    video_path = output_dir / "background_video.mp4"
    if not video_path.exists():
        raise RuntimeError(f"No video at {video_path}")
    info = video_info(video_path)

    smooth_cfg = cfg.get("video_completion", {}).get("temporal_smooth", {})
    if args.keep_version and not (output_dir / "background_video_unsmoothed.mp4").exists():
        shutil.copy2(video_path, output_dir / "background_video_unsmoothed.mp4")

    repo = (PROJECT_ROOT / cfg["video_completion"].get(
        "repo_dir", "external/ProPainter")).resolve()
    smooth_out = output_dir / "smooth_raw.mp4"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(cfg["video_completion"].get("cuda_visible_devices", "7"))
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    env["PYTHONUNBUFFERED"] = "1"
    log_path = output_dir / "logs" / "temporal_smooth.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "conda", "run", "--no-capture-output", "-n",
        cfg["video_completion"].get("environment", "vbr-seg"),
        "python", "-m", "vbr.temporal_smooth",
        "--input", str(video_path.resolve()),
        "--masks", str((output_dir / "masks_inpaint").resolve()),
        "--output", str(smooth_out.resolve()),
        "--fps", str(info["fps"]),
        "--repo-dir", str(repo.resolve()),
        "--raft-checkpoint", str((repo / "weights" / "raft-things.pth").resolve()),
        "--log", str(log_path.resolve()),
    ]
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(command, cwd=PROJECT_ROOT, env=env,
                                stdout=log, stderr=subprocess.STDOUT, text=True)
    if result.returncode:
        tail = "\n".join(log_path.read_text(encoding="utf-8").splitlines()[-40:])
        raise RuntimeError(f"temporal_smooth failed ({result.returncode})\n{tail}")

    input_video = (PROJECT_ROOT / cfg["input_video"]).resolve()
    encode = [
        _resolve_ffmpeg(cfg["video_completion"].get("ffmpeg_bin")),
        "-y", "-loglevel", "error",
        "-i", str(smooth_out),
        "-i", str(input_video),
        "-map", "0:v:0", "-map", "1:a?",
        "-vf", f"scale={info['width']}:{info['height']}",
        "-c:v", "libx264", "-crf", str(cfg["video_completion"].get("crf", 18)),
        "-preset", cfg["video_completion"].get("preset", "medium"),
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
        str(video_path),
    ]
    subprocess.run(encode, cwd=PROJECT_ROOT, check=True)

    smooth_out.unlink(missing_ok=True)
    offset_report = {
        "kernel": int(smooth_cfg.get("kernel", 3)),
        "input": str(video_path.resolve()),
        "output": str(video_path.resolve()),
        "frames": info["frames"],
        "note": "flow-aligned 3-frame median inside masks_inpaint, h264+aac",
    }
    (output_dir / "post_temporal_smooth.json").write_text(
        json.dumps(offset_report, indent=2), encoding="utf-8"
    )
    print(json.dumps(offset_report, indent=2))


if __name__ == "__main__":
    main()