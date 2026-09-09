"""Re-run miss-window refinement and the video stage on cached segmentation.

Reuses the baked keyframe seeds (masks_keyframes_sam31_onset) and the existing
foreground masks, so the base SAM3.1 keyframe detection, the SAM2 base
propagation and the onset refinement are NOT repeated. The current pipeline
entry simply cannot be targeted cheaply because any code change invalidates
the segmentation fingerprint; this driver calls the same building blocks.

Usage (vbr environment):
    python -m tools.refine_misses --config configs/vggt_slam.yaml \
        [--no-video] [--no-smooth]
"""

from __future__ import annotations

import argparse
import json

from vbr.cli import PROJECT_ROOT, _refine_miss_windows, update_pipeline_status
from vbr.config import load_config
from vbr.models.inpainting import ProPainterAdapter
from vbr.video import (
    extract_frame_sets,
    fill_enclosed_mask_holes,
    prepare_inpainting_masks,
    stabilize_foreground_masks,
    video_info,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/vggt_slam.yaml")
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="stop after the refined masks (skip inpaint masks and ProPainter)",
    )
    parser.add_argument(
        "--no-smooth",
        action="store_true",
        help="disable the flow-aligned temporal smoothing in the video stage",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = (PROJECT_ROOT / cfg["output_dir"]).resolve()
    input_video = (PROJECT_ROOT / cfg["input_video"]).resolve()
    info = video_info(input_video)
    all_frames = output_dir / "frames_all"
    keyframes = output_dir / "frames_keyframes"
    key_stride = int(cfg.get("sampling", {}).get("segmentation_stride", 30))
    keyframe_ids = extract_frame_sets(input_video, all_frames, keyframes, key_stride)

    masks = output_dir / "masks"
    if not masks.exists() or not any(masks.glob("*.png")):
        raise RuntimeError(
            f"No foreground masks in {masks}; run the full pipeline first"
        )
    shared_seed_dir = output_dir / "masks_keyframes_sam31_onset"
    if not shared_seed_dir.exists():
        raise RuntimeError(
            f"No augmented seed dir {shared_seed_dir}; run the full pipeline first"
        )

    segmentation_cfg = dict(cfg["segmentation"])
    miss_seed_ids, miss_report = _refine_miss_windows(
        segmentation_cfg,
        all_frames,
        shared_seed_dir,
        masks,
        output_dir,
        output_dir / "logs",
        info["frames"],
        original_video=input_video,
    )
    if not miss_seed_ids:
        print("No miss-refinement seeds accepted; masks unchanged, skipping video stage.")
        return

    pin_stems = sorted(
        set(keyframe_ids)
        | {int(path.stem) for path in shared_seed_dir.glob("*.png")}
        | set(miss_seed_ids)
    )
    stabilize_report = stabilize_foreground_masks(
        masks,
        pin_stems=pin_stems,
        xor_threshold=float(segmentation_cfg.get("temporal_xor_threshold", 0.06)),
        window=int(segmentation_cfg.get("temporal_blend_window", 8)),
    )
    fill_enclosed_mask_holes(masks)
    print(f"stabilize: {stabilize_report}")

    (output_dir / "miss_windows.json").write_text(
        json.dumps(miss_report, indent=2), encoding="utf-8"
    )
    if args.no_video:
        return

    completion_cfg = dict(cfg.get("video_completion", {}))
    if args.no_smooth:
        completion_cfg.setdefault("temporal_smooth", {})["enabled"] = False
    inpaint_masks = output_dir / "masks_inpaint"
    inpaint_report = prepare_inpainting_masks(
        masks,
        inpaint_masks,
        close_px=int(completion_cfg.get("mask_close_px", 9)),
        temporal_radius=int(completion_cfg.get("mask_temporal_radius", 2)),
        dilate_px=int(completion_cfg.get("mask_expand_px", 1)),
        hull_min_area=int(completion_cfg.get("mask_hull_min_area", 0)),
        hull_max_extra=float(completion_cfg.get("mask_hull_max_extra", 0.35)),
        overlap=float(completion_cfg.get("mask_overlap", 0.12)),
    )
    (output_dir / "inpaint_refine_report.json").write_text(
        json.dumps({"inpaint_masks": inpaint_report}, indent=2), encoding="utf-8"
    )
    video_report = ProPainterAdapter(completion_cfg, PROJECT_ROOT).run(
        input_video,
        all_frames,
        inpaint_masks,
        output_dir / "background_video.mp4",
        info["fps"],
    )

    def record(status):
        status["stages"]["inpainting_masks"] = {"state": "complete", **inpaint_report}
        status["stages"]["video"] = {"state": "complete", **video_report}
        status["miss_refinement_driver"] = {
            "seeds": miss_seed_ids,
            "report": miss_report,
            "stabilize": stabilize_report,
        }
        status["state"] = "complete"

    update_pipeline_status(output_dir, record)
    print(json.dumps({"miss": miss_report, "stabilize": stabilize_report,
                      "inpaint_masks": inpaint_report, "video": video_report},
                     indent=2, ensure_ascii=False)[-2500:])


if __name__ == "__main__":
    main()