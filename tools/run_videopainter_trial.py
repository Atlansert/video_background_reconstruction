"""One-off: run the VideoPainterAdapter end-to-end on the full video (trial branch)."""

import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vbr.models.videopainter import VideoPainterAdapter

ROOT = Path(__file__).resolve().parent.parent
cfg = yaml.safe_load((ROOT / "configs" / "vggt_slam.yaml").read_text())
completion = cfg["video_completion"]
completion["videopainter"]["cuda_visible_devices"] = "5"

run_dir = ROOT / "outputs" / "001_sam31_slam"
adapter = VideoPainterAdapter(completion, ROOT)
started = time.time()
report = adapter.run(
    ROOT / "video" / "001.mp4",
    run_dir / "frames_all",
    run_dir / "masks_inpaint",
    run_dir / "background_video.mp4",
    29.97,
)
report["total_seconds"] = time.time() - started
(run_dir / "videopainter_trial_report.json").write_text(
    __import__("json").dumps(report, indent=2), encoding="utf-8"
)
print("DONE", report.get("output"), round(report["total_seconds"] / 60, 1), "min")
