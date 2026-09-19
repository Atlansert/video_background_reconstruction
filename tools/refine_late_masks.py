"""Forced late-section mask repair (fridge / wardrobe / glass cabinet doors).

Why this exists: the automatic miss refinement (`vbr.cli._refine_miss_windows`)
only fires on windows whose WHOLE-FRAME coverage stays under 0.20. In the late
section (1450-1799) the walls/floor carry large masks, so per-frame coverage
runs 0.29-0.58 and the detector never fires — while the fridge, the white
wardrobe right of the black pole and the glass cabinet door panels stay
unmasked. The configured `box_seeds` also aim at the left third of the frame,
but the fridge sits center-RIGHT around frame 1500 and only reaches the left
edge after ~1600 (the camera pans), so the previous refinement added wall and
cabinet area without ever touching the fridge.

This driver re-probes the late windows with per-segment corrected boxes
(matched to the object's on-screen position over time), selects seeds that
add real coverage, re-anchors SAM2 propagation from the existing seed set,
rebuilds masks_inpaint and re-renders both mask overlay videos. It does NOT
touch the video stage — SVOR re-running is a separate, later step.

Usage (vbr-seg environment):
    python -m tools.refine_late_masks            # full repair
    python -m tools.refine_late_masks --no-overlays
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from vbr.cli import PROJECT_ROOT
from vbr.config import load_config
from vbr.models.segmentation import SegmentationAdapter
from vbr.video import (
    fill_enclosed_mask_holes,
    prepare_inpainting_masks,
    select_miss_seeds,
    stabilize_foreground_masks,
    write_mask_overlay,
)

# Late-section windows kept wide on purpose: seeds that add nothing are
# rejected by select_miss_seeds, so over-probing only costs probe time.
WINDOWS = [
    {"start": 1440, "end": 1652},
    {"start": 1652, "end": 1799},
]

# Normalized xyxy boxes matched to where each missed object actually is over
# time (verified on grid-rendered frames 1500/1525/1550/1575/1600/1650/1700).
BOX_PROMPTS = [
    # Fridge pans right -> center-left -> left edge with the camera
    {"start": 1440, "end": 1570, "box": [0.46, 0.0, 0.86, 0.60], "prompt": "refrigerator"},
    {"start": 1570, "end": 1652, "box": [0.05, 0.05, 0.44, 0.64], "prompt": "refrigerator"},
    {"start": 1652, "end": 1799, "box": [0.08, 0.02, 0.38, 0.64], "prompt": "refrigerator"},
    # White wardrobe right of the black pole (drifts from x~0.65 to ~0.50)
    {"start": 1560, "end": 1595, "box": [0.58, 0.0, 0.80, 0.42], "prompt": "wardrobe"},
    {"start": 1595, "end": 1799, "box": [0.42, 0.0, 0.66, 0.48], "prompt": "wardrobe"},
    # Glass door panels of the upper kitchen cabinets
    {"start": 1440, "end": 1652, "box": [0.14, 0.0, 1.00, 0.42], "prompt": "kitchen wall cabinets"},
    {"start": 1652, "end": 1799, "box": [0.14, 0.0, 0.96, 0.45], "prompt": "kitchen wall cabinets"},
]

EXTRA_PROMPTS = ["refrigerator", "wardrobe"]
EXTRA_THRESHOLDS = {"refrigerator": 0.30, "wardrobe": 0.30}


def to_local(spec: dict, frame_ids: list[int]) -> dict | None:
    """Map a global [start, end] box spec onto positions in the probed subset."""
    local_start = next(
        (i for i, fid in enumerate(frame_ids) if fid >= int(spec["start"])), None
    )
    local_end = next(
        (i for i in range(len(frame_ids) - 1, -1, -1) if frame_ids[i] <= int(spec["end"])),
        None,
    )
    if local_start is None or local_end is None or local_start > local_end:
        return None
    return {**spec, "start": local_start, "end": local_end}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/vggt_slam.yaml")
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--no-overlays", action="store_true")
    parser.add_argument("--windows-json", default=None,
                        help="override WINDOWS, e.g. '[{\"start\":1440,\"end\":1592}]'")
    parser.add_argument("--boxes-json", default=None,
                        help="override BOX_PROMPTS with a JSON list of box specs")
    parser.add_argument("--union-existing", action="store_true", default=True,
                        help="union accepted seeds with any existing seed at the same "
                             "frame so re-anchoring can only grow coverage")
    parser.add_argument("--no-preserve", action="store_true",
                        help="run the SAM 3.1 worker with preserve_prompts disabled. "
                             "Needed for the late section: preserve classes door/window "
                             "match the wood-grain fridge body and the glass cabinet "
                             "panels, and the preserve subtraction runs AFTER box "
                             "prompts, carving those objects out of every mask.")
    args = parser.parse_args()

    import cv2
    import numpy as np

    cfg = load_config(args.config)
    windows = json.loads(args.windows_json) if args.windows_json else WINDOWS
    box_specs = json.loads(args.boxes_json) if args.boxes_json else BOX_PROMPTS
    output_dir = (PROJECT_ROOT / cfg["output_dir"]).resolve()
    all_frames = output_dir / "frames_all"
    masks = output_dir / "masks"
    seed_dir = output_dir / "masks_keyframes_sam31_onset"
    logs_dir = output_dir / "logs"
    for path in (all_frames, masks, seed_dir):
        if not path.exists():
            raise RuntimeError(f"Missing {path}; run the full pipeline first")

    # Keep a restorable pre-repair copy (idempotent: first run wins).
    backup_root = output_dir / "snapshots" / "masks"
    for name, source in (("masks_pre_latefix", masks), ("masks_inpaint_pre_latefix", output_dir / "masks_inpaint")):
        target = backup_root / name
        if source.exists() and not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source, target)

    frame_ids = sorted({
        fid
        for window in windows
        for fid in range(window["start"], window["end"] + 1, args.stride)
    })
    box_prompts = [s for s in (to_local(spec, frame_ids) for spec in box_specs) if s]

    adapter = SegmentationAdapter(dict(cfg["segmentation"]), PROJECT_ROOT)
    refinement_dir = output_dir / "masks_late_refinement"
    shutil.rmtree(refinement_dir, ignore_errors=True)
    if args.no_preserve:
        # Direct worker call, preserve disabled (see --no-preserve help).
        # Keep the FULL remove-prompt suite so candidates stay comparable to
        # the current masks (select_miss_seeds rejects candidates below
        # min_keep_fraction of the current coverage); only preserve is empty.
        seg_cfg = dict(cfg["segmentation"])
        prompts = list(seg_cfg.get("prompts", []))
        prompts += [p for p in EXTRA_PROMPTS if p not in prompts]
        thresholds = dict(seg_cfg.get("prompt_thresholds", {}))
        thresholds.update(EXTRA_THRESHOLDS)
        env_name, env, checkpoint = adapter._environment()
        subset = refinement_dir / "frames_subset"
        subset.mkdir(parents=True, exist_ok=True)
        for fid in frame_ids:
            source = all_frames / f"{fid:06d}.jpg"
            if not (subset / source.name).exists():
                shutil.copy(source, subset / source.name)
        command = [
            "conda", "run", "--no-capture-output", "-n", env_name,
            "python", "-m", "vbr.sam31_keyframes",
            "--frames", str(subset.resolve()),
            "--output", str(refinement_dir.resolve()),
            "--checkpoint", str(checkpoint),
            "--prompts-json", json.dumps(prompts),
            "--preserve-prompts-json", "[]",
            "--prompt-thresholds-json", json.dumps(thresholds),
            "--threshold", str(seg_cfg.get("sam3_threshold", 0.45)),
            "--max-objects", str(seg_cfg.get("sam3_max_objects", 64)),
            "--box-prompts-json", json.dumps(box_prompts),
        ]
        log_path = logs_dir / "sam31_late_nopreserve.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(command, cwd=PROJECT_ROOT, env=env,
                                    stdout=log, stderr=subprocess.STDOUT, text=True)
        if result.returncode:
            tail = "\n".join(log_path.read_text(encoding="utf-8").splitlines()[-40:])
            raise RuntimeError(f"sam31 no-preserve refinement failed; see {log_path}\n{tail}")
    else:
        adapter.run_refinement_masks(
            frame_ids,
            all_frames,
            refinement_dir,
            logs_dir,
            extra_prompts=EXTRA_PROMPTS,
            extra_prompt_thresholds=EXTRA_THRESHOLDS,
            box_prompts=box_prompts,
        )

    accepted, selection = select_miss_seeds(
        refinement_dir,
        masks,
        min_added_area_px=2000,
        min_keep_fraction=0.5,
        max_coverage_increase=float(
            cfg["segmentation"].get("miss_refinement", {}).get("max_coverage_increase", 0.45)
        ),
    )
    report = {
        "windows": windows,
        "box_prompts": box_specs,
        "probed_frames": len(frame_ids),
        "accepted_seed_frames": accepted,
        "rejected_frames": selection.get("rejected_frames"),
        "total_added_px": selection.get("total_added_px"),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not accepted:
        (output_dir / "late_mask_refine_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        print("No seeds accepted; masks unchanged.")
        return

    for stem in accepted:
        seed_path = refinement_dir / f"{stem:06d}.png"
        if not seed_path.exists():
            continue
        target = seed_dir / seed_path.name
        if args.union_existing and target.exists():
            existing = cv2.imread(str(target), cv2.IMREAD_GRAYSCALE)
            addition = cv2.imread(str(seed_path), cv2.IMREAD_GRAYSCALE)
            if existing is not None and addition is not None:
                merged = np.maximum(existing, addition)
                cv2.imwrite(str(target), merged)
                continue
        shutil.copy2(seed_path, target)

    # Re-anchor propagation from the full seed set (existing keyframe/onset
    # seeds plus the accepted late-section seeds); overwrites masks/ in place.
    sam2_report = adapter.propagate(
        all_frames, seed_dir, masks, logs_dir, log_name="sam2_late_refine.log"
    )
    report["sam2"] = sam2_report

    pin_stems = sorted({int(p.stem) for p in seed_dir.glob("*.png")})
    stabilize_report = stabilize_foreground_masks(
        masks,
        pin_stems=pin_stems,
        xor_threshold=float(cfg["segmentation"].get("temporal_xor_threshold", 0.06)),
        window=int(cfg["segmentation"].get("temporal_blend_window", 8)),
    )
    filled_holes = fill_enclosed_mask_holes(masks)

    completion_cfg = dict(cfg.get("video_completion", {}))
    inpaint_report = prepare_inpainting_masks(
        masks,
        output_dir / "masks_inpaint",
        close_px=int(completion_cfg.get("mask_close_px", 9)),
        temporal_radius=int(completion_cfg.get("mask_temporal_radius", 2)),
        dilate_px=int(completion_cfg.get("mask_expand_px", 1)),
        hull_min_area=int(completion_cfg.get("mask_hull_min_area", 0)),
        hull_max_extra=float(completion_cfg.get("mask_hull_max_extra", 0.35)),
        overlap=float(completion_cfg.get("mask_overlap", 0.12)),
    )

    report["stabilize"] = stabilize_report
    report["inpaint_masks"] = inpaint_report
    (output_dir / "late_mask_refine_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    if not args.no_overlays:
        input_video = (PROJECT_ROOT / cfg["input_video"]).resolve()
        write_mask_overlay(input_video, masks, output_dir / "mask_overlay.mp4")
        write_mask_overlay(input_video, output_dir / "masks_inpaint",
                           output_dir / "mask_overlay_inpaint.mp4")

    print(json.dumps({"accepted": len(accepted), "stabilize": stabilize_report,
                      "inpaint": inpaint_report}, indent=2))


if __name__ == "__main__":
    main()
