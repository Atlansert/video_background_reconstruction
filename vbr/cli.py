"""Command-line entry point for foreground-free room reconstruction."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from .config import load_config
from .geometry import build_mesh, save_pointcloud
from .interactive import write_html
from .models.segmentation import SegmentationAdapter
from .models.slam import SLAMAdapter
from .models.inpainting import ProPainterAdapter, _resolve_ffmpeg
from .models.svor import SVORAdapter
from .models.effecterase import EffectEraseAdapter
from .prompts import resolve_prompts
from .video import (
    copy_through_evidence,
    derive_box_seeds,
    detect_mask_onsets,
    detect_persistent_misses,
    extract_frame_sets,
    fill_enclosed_mask_holes,
    mask_statistics,
    onset_latency_metrics,
    prepare_inpainting_masks,
    select_miss_seeds,
    stabilize_foreground_masks,
    video_info,
    write_background_video,
    write_mask_overlay,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Mask caches only need invalidation when segmentation-relevant code changes;
# hashing everything would needlessly discard usable masks on unrelated edits.
_SEGMENTATION_SOURCE_MODULES = (
    "cli.py",
    "config.py",
    "sam31_keyframes.py",
    "sam2_propagate.py",
    "prompts.py",
    "models/segmentation.py",
    "video.py",
)


def _segmentation_source_hash() -> str:
    digest = hashlib.sha256()
    package_dir = Path(__file__).resolve().parent
    for relative in _SEGMENTATION_SOURCE_MODULES:
        path = package_dir / relative
        try:
            digest.update(path.read_bytes())
        except OSError:
            digest.update(f"missing:{relative}".encode())
    return digest.hexdigest()[:16]


def _fingerprint(video_path: Path, section: dict) -> str:
    stat = video_path.stat()
    payload = {
        "video": str(video_path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "config": section,
        "code": _segmentation_source_hash(),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _cached(manifest_path: Path, fingerprint: str, expected_files: int, files_dir: Path):
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        manifest.get("fingerprint") == fingerprint
        and len(list(files_dir.glob("*.png"))) == expected_files
    )


def _clear_generated(directory: Path, suffixes=(".png",)):
    directory.mkdir(parents=True, exist_ok=True)
    for path in directory.iterdir():
        if path.is_file() and path.suffix.lower() in suffixes:
            path.unlink()


def _build_opening_hints(values, preserved_dir: Path) -> list | None:
    """Pair preserved fixed-structure masks with reconstruction poses."""
    import cv2

    frame_paths = values["frame_paths"]
    coords = values["original_coords"]
    extrinsics = values["extrinsics"]
    intrinsics = values["intrinsics"]
    depth = values["depth"][..., 0]
    index_by_stem = {int(Path(path).stem): index for index, path in enumerate(frame_paths)}
    hints = []
    for mask_path in sorted(preserved_dir.glob("*.png")):
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None or not np.count_nonzero(mask):
            continue
        index = index_by_stem.get(int(mask_path.stem))
        if index is None:
            continue
        hints.append(
            {
                "extrinsic": extrinsics[index],
                "intrinsic": intrinsics[index],
                "original_coords": coords[index],
                "depth": depth[index],
                "mask": mask > 0,
            }
        )
    return hints or None


def _load_opening_hints(output_dir: Path, cfg, all_frames: Path, slam_result: dict):
    """Reuse or generate fixed-structure masks for wall-opening carving."""
    geometry_cfg = cfg.get("geometry", {})
    if not geometry_cfg.get("carve_openings", True):
        return None
    prompts = geometry_cfg.get(
        "opening_prompts",
        [
            "door",
            "window",
            "staircase",
            "stairs",
            "kitchen cabinet",
            "refrigerator",
            "sink",
        ],
    )
    values = np.load(slam_result["reconstruction"])
    frame_ids = [int(value) for value in values["frame_ids"]]
    expected = {"frame_ids": frame_ids, "prompts": prompts}
    masks_out = output_dir / "masks_openings"
    manifest_path = masks_out / "manifest.json"
    if manifest_path.exists():
        try:
            if json.loads(manifest_path.read_text(encoding="utf-8")) == expected:
                return _build_opening_hints(values, masks_out / "preserved")
        except (OSError, json.JSONDecodeError):
            pass
    SegmentationAdapter(cfg["segmentation"], PROJECT_ROOT).run_opening_masks(
        frame_ids, all_frames, masks_out, output_dir / "logs", prompts
    )
    manifest_path.write_text(json.dumps(expected, indent=2), encoding="utf-8")
    return _build_opening_hints(values, masks_out / "preserved")


def _refine_onset_seeds(
    segmentation_cfg, all_frames, key_masks, masks, output_dir, logs_dir, frame_count
):
    """Add dense SAM 3.1 seeds just before persistent foreground onsets.

    Note: this rebuilds the augmented seed dir from the base keyframes, so
    miss-window seeds merged by earlier runs are dropped here; the pipeline
    re-derives them in _refine_miss_windows right after this stage whenever
    refinement runs, so the final seed set stays complete.
    """
    onset_cfg = segmentation_cfg.get("onset_refinement", {})
    if not onset_cfg.get("enabled", True):
        return [], {"enabled": False, "events": []}, None
    events = detect_mask_onsets(
        masks,
        min_new_area_px=int(onset_cfg.get("min_new_area_px", 3500)),
        persistence_frames=int(onset_cfg.get("persistence_frames", 3)),
        dilation_px=int(onset_cfg.get("dilation_px", 9)),
        cooldown_frames=int(onset_cfg.get("cooldown_frames", 15)),
    )
    max_events = int(onset_cfg.get("max_events", 8))
    events = sorted(events, key=lambda event: -event["new_area_px"])[:max_events]
    events.sort(key=lambda event: event["frame_id"])
    if not events:
        return [], {"enabled": True, "events": [], "refined_seed_frames": []}, None

    lookback = int(onset_cfg.get("lookback_frames", 12))
    forward = int(onset_cfg.get("forward_frames", 6))
    frame_ids = sorted(
        {
            frame_id
            for event in events
            for frame_id in range(
                max(0, event["frame_id"] - lookback),
                min(frame_count, event["frame_id"] + forward + 1),
            )
        }
    )
    refinement_dir = output_dir / "masks_onset_refinement"
    augmented_dir = output_dir / "masks_keyframes_sam31_onset"
    shutil.rmtree(refinement_dir, ignore_errors=True)
    shutil.rmtree(augmented_dir, ignore_errors=True)
    shutil.copytree(key_masks, augmented_dir)
    adapter = SegmentationAdapter(segmentation_cfg, PROJECT_ROOT)
    adapter.run_refinement_masks(frame_ids, all_frames, refinement_dir, logs_dir)

    refined_seed_frames = []
    for path in refinement_dir.glob("*.png"):
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None or not np.any(mask):
            continue
        shutil.copy2(path, augmented_dir / path.name)
        refined_seed_frames.append(int(path.stem))
    if not refined_seed_frames:
        return [], {
            "enabled": True,
            "events": events,
            "candidate_frames": frame_ids,
            "refined_seed_frames": [],
            "warning": "SAM 3.1 refinement returned no non-empty masks",
        }, augmented_dir

    baseline_dir = output_dir / "masks_onset_baseline"
    shutil.rmtree(baseline_dir, ignore_errors=True)
    shutil.copytree(masks, baseline_dir)
    shutil.rmtree(masks, ignore_errors=True)
    masks.mkdir(parents=True, exist_ok=True)
    sam2_report = adapter.propagate(
        all_frames, augmented_dir, masks, logs_dir, log_name="sam2_onset_refinement.log"
    )
    latency = onset_latency_metrics(
        baseline_dir,
        masks,
        events,
        dilation_px=int(onset_cfg.get("dilation_px", 9)),
        overlap_fraction=float(onset_cfg.get("overlap_fraction", 0.5)),
        window_back=int(onset_cfg.get("lookback_frames", 12)),
        window_forward=int(onset_cfg.get("forward_frames", 6)),
    )
    improved_event_ids = {
        record["frame_id"]
        for record in latency["events"]
        if record["latency_frames"] > 0
    }
    improved_seed_frames = sorted(
        {
            seed
            for event in events
            if event["frame_id"] in improved_event_ids
            for seed in refined_seed_frames
            if event["frame_id"] - lookback <= seed <= event["frame_id"] + forward
        }
    )
    report = {
        "enabled": True,
        "events": events,
        "candidate_frames": frame_ids,
        "refined_seed_frames": refined_seed_frames,
        "improved_seed_frames": improved_seed_frames,
        "sam2": sam2_report,
        "latency": latency,
    }
    (output_dir / "onset_events.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return improved_seed_frames, report, augmented_dir


def _refine_miss_windows(
    segmentation_cfg, all_frames, augmented_dir, masks, output_dir, logs_dir, frame_count,
    original_video=None,
):
    """Add dense SAM 3.1 seeds inside sustained low-coverage windows.

    Windows where the mask stays nearly empty for many frames let ProPainter
    copy the missed object back from unmasked neighbors (the 746-970 sofa).
    Dense probing with extra sofa-family prompts re-segments only those
    windows; seeds that add substantial area are pinned and SAM2 repropagates
    the full clip. Other frames keep the precise masks untouched.
    """
    miss_cfg = segmentation_cfg.get("miss_refinement", {})
    if not miss_cfg.get("enabled", True):
        return [], {"enabled": False, "windows": []}
    windows = detect_persistent_misses(
        masks,
        frame_count,
        min_coverage=float(miss_cfg.get("min_coverage", 0.20)),
        min_window_frames=int(miss_cfg.get("min_window_frames", 30)),
        merge_gap_frames=int(miss_cfg.get("merge_gap_frames", 12)),
        max_windows=int(miss_cfg.get("max_windows", 3)),
    )
    if not windows:
        return [], {"enabled": True, "windows": [], "probed_frames": 0}

    stride = max(1, int(miss_cfg.get("window_stride", 2)))
    frame_ids = sorted(
        {
            frame_id
            for window in windows
            for frame_id in range(window["start"], window["end"] + 1, stride)
        }
    )
    configured_boxes = miss_cfg.get("box_seeds")
    if configured_boxes is None:
        configured_boxes = derive_box_seeds(
            masks,
            windows,
            subwindow_frames=int(miss_cfg.get("box_subwindow_frames", 25)),
            expansion=float(miss_cfg.get("box_expansion", 0.4)),
        )
    # The refinement subset holds strided frame ids; box specs index that
    # subset locally, so global [start, end] ranges map to subset positions.
    def to_local(spec: dict) -> dict | None:
        local_start = next(
            (position for position, frame_id in enumerate(frame_ids)
             if frame_id >= int(spec["start"])),
            None,
        )
        local_end = next(
            (position for position in range(len(frame_ids) - 1, -1, -1)
             if frame_ids[position] <= int(spec["end"])),
            None,
        )
        if local_start is None or local_end is None or local_start > local_end:
            return None
        return {**spec, "start": local_start, "end": local_end}

    box_prompts = [
        spec
        for spec in (to_local(entry) for entry in configured_boxes or [])
        if spec is not None
    ]
    refinement_dir = output_dir / "masks_miss_refinement"
    shutil.rmtree(refinement_dir, ignore_errors=True)
    adapter = SegmentationAdapter(segmentation_cfg, PROJECT_ROOT)
    adapter.run_refinement_masks(
        frame_ids,
        all_frames,
        refinement_dir,
        logs_dir,
        extra_prompts=list(miss_cfg.get("extra_prompts", [])),
        extra_prompt_thresholds=dict(miss_cfg.get("extra_prompt_thresholds", {})),
        box_prompts=box_prompts,
    )
    evidence = None
    current_video = output_dir / "background_video.mp4"
    current_inpaint_masks = output_dir / "masks_inpaint"
    if (
        miss_cfg.get("evidence_enabled", True)
        and original_video is not None
        and current_video.exists()
        and current_inpaint_masks.exists()
    ):
        evidence = copy_through_evidence(
            original_video,
            current_video,
            current_inpaint_masks,
            frame_ids,
            dilate_px=int(miss_cfg.get("evidence_dilate_px", 5)),
        )
    accepted_seed_frames, selection = select_miss_seeds(
        refinement_dir,
        masks,
        min_added_area_px=int(miss_cfg.get("min_added_area_px", 2000)),
        min_keep_fraction=float(miss_cfg.get("min_keep_fraction", 0.5)),
        max_coverage_increase=miss_cfg.get("max_coverage_increase"),
        evidence=evidence,
        evidence_min_overlap=float(miss_cfg.get("evidence_min_overlap", 0.5)),
    )
    report = {
        "enabled": True,
        "windows": windows,
        "probed_frames": len(frame_ids),
        "candidate_frames": frame_ids,
        "box_prompts": box_prompts or [],
        "evidence_frames": len(evidence) if evidence else 0,
        "accepted_seed_frames": accepted_seed_frames,
        **selection,
    }
    if accepted_seed_frames:
        for stem in accepted_seed_frames:
            seed_path = refinement_dir / f"{stem:06d}.png"
            if seed_path.exists():
                shutil.copy2(seed_path, augmented_dir / seed_path.name)
        baseline_dir = output_dir / "masks_miss_baseline"
        shutil.rmtree(baseline_dir, ignore_errors=True)
        shutil.copytree(masks, baseline_dir)
        shutil.rmtree(masks, ignore_errors=True)
        masks.mkdir(parents=True, exist_ok=True)
        sam2_report = adapter.propagate(
            all_frames,
            augmented_dir,
            masks,
            logs_dir,
            log_name="sam2_miss_refinement.log",
        )
        report["sam2"] = sam2_report
    (output_dir / "miss_windows.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return accepted_seed_frames, report


def _ensure_frames(video_path, info, all_frames, keyframes, stride):
    """Return keyframe ids, regenerating cached frames when the source moved.

    Frame caches previously only checked counts, so replacing the input video
    with another of the same length silently reused stale frames. A small
    marker file records the source stat + shape and forces re-extraction on
    any mismatch.
    """
    marker_path = all_frames.parent / "video_source.json"
    marker = {
        "path": str(Path(video_path).resolve()),
        "size": Path(video_path).stat().st_size,
        "mtime_ns": Path(video_path).stat().st_mtime_ns,
        "frames": info["frames"],
        "width": info["width"],
        "height": info["height"],
        "fps": info["fps"],
        "stride": int(stride),
    }
    cached_marker = None
    try:
        cached_marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        cached_marker = None
    expected_keyframes = len(set(range(0, info["frames"], stride)) | {info["frames"] - 1})
    all_count = len(list(all_frames.glob("*.jpg")))
    key_count = len(list(keyframes.glob("*.jpg")))
    if cached_marker == marker and all_count == info["frames"] and key_count == expected_keyframes:
        return sorted(int(path.stem) for path in keyframes.glob("*.jpg"))
    _clear_generated(all_frames, (".jpg",))
    _clear_generated(keyframes, (".jpg",))
    keyframe_ids = extract_frame_sets(video_path, all_frames, keyframes, stride)
    marker_path.write_text(json.dumps(marker, indent=2), encoding="utf-8")
    return keyframe_ids


def update_pipeline_status(output_dir: Path, patch: callable) -> None:
    """Rewrite pipeline_status.json through ``patch(status)``.

    The tool-stage drivers (refine_misses, apply_temporal_smooth,
    rebuild_geometry_from_background) run outside ``cli.run``; this keeps the
    status file honest about which stages the on-disk artifacts came from.
    """
    status_path = Path(output_dir) / "pipeline_status.json"
    status = {}
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        status = {"state": "unknown", "stages": {}}
    patch(status)
    status_path.write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")


def run(cfg, stop_after="all", force=False, refine_onsets=False, refine_misses=False):
    started = time.time()
    input_video = (PROJECT_ROOT / cfg["input_video"]).resolve()
    output_dir = (PROJECT_ROOT / cfg["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "pipeline_status.json"
    status = {"state": "running", "started_at_unix": started, "stages": {}}
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")

    try:
        info = video_info(input_video)
        (output_dir / "video_info.json").write_text(
            json.dumps(info, indent=2), encoding="utf-8"
        )
        all_frames = output_dir / "frames_all"
        keyframes = output_dir / "frames_keyframes"
        key_stride = int(cfg.get("sampling", {}).get("segmentation_stride", 30))
        keyframe_ids = _ensure_frames(
            input_video, info, all_frames, keyframes, key_stride
        )
        status["stages"]["frames"] = {
            "state": "complete",
            "all": info["frames"],
            "keyframes": len(keyframe_ids),
        }
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")

        key_masks = output_dir / "masks_keyframes_sam31"
        masks = output_dir / "masks"
        segmentation_manifest = output_dir / "segmentation_manifest.json"
        segmentation_cfg = dict(cfg["segmentation"])
        segmentation_cfg["prompts"] = resolve_prompts(
            segmentation_cfg,
            cfg.get("gpt", {}),
            keyframes.glob("*.jpg"),
        )
        segmentation_fingerprint = _fingerprint(input_video, segmentation_cfg)
        segmentation_ran = force or not _cached(
            segmentation_manifest, segmentation_fingerprint, info["frames"], masks
        )
        if segmentation_ran:
            _clear_generated(key_masks)
            _clear_generated(masks)
            segmentation_report = SegmentationAdapter(
                segmentation_cfg, PROJECT_ROOT
            ).run(
                all_frames,
                keyframes,
                key_masks,
                masks,
                output_dir / "logs",
            )
            segmentation_manifest.write_text(
                json.dumps(
                    {
                        "fingerprint": segmentation_fingerprint,
                        "report": segmentation_report,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        else:
            segmentation_report = {
                "sam31": json.loads(
                    (key_masks / "sam31_report.json").read_text(encoding="utf-8")
                ),
                "sam2": json.loads(
                    (masks / "sam2_report.json").read_text(encoding="utf-8")
                ),
            }
            segmentation_manifest.write_text(
                json.dumps(
                    {
                        "fingerprint": segmentation_fingerprint,
                        "report": segmentation_report,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

        onset_seed_ids = []
        miss_seed_ids = []
        onset_report = {"enabled": False, "events": [], "refined_seed_frames": []}
        miss_report = {"enabled": False, "windows": []}
        if segmentation_ran or refine_onsets:
            onset_seed_ids, onset_report, onset_augmented_dir = _refine_onset_seeds(
                segmentation_cfg,
                all_frames,
                key_masks,
                masks,
                output_dir,
                output_dir / "logs",
                info["frames"],
            )
        else:
            onset_augmented_dir = None
        if segmentation_ran or refine_onsets or refine_misses:
            shared_seed_dir = onset_augmented_dir
            if shared_seed_dir is None:
                shared_seed_dir = output_dir / "masks_keyframes_sam31_onset"
                if not shared_seed_dir.exists():
                    shutil.copytree(key_masks, shared_seed_dir)
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
            if not onset_seed_ids and not segmentation_ran:
                # Cached path: the augmented dir already holds the onset
                # seeds, so pin everything it contains rather than losing
                # them from the stabilization pin set.
                onset_seed_ids = sorted(
                    int(path.stem) for path in shared_seed_dir.glob("*.png")
                )
        segmentation_seed_ids = sorted(
            set(keyframe_ids) | set(onset_seed_ids) | set(miss_seed_ids)
        )
        filled_hole_pixels = fill_enclosed_mask_holes(masks)
        stabilize_report = {"changed_frames": 0, "changed_pixels": 0, "pinned_frames": 0}
        if segmentation_cfg.get("temporal_stabilize", True):
            stabilize_report = stabilize_foreground_masks(
                masks,
                pin_stems=segmentation_seed_ids,
                xor_threshold=float(
                    segmentation_cfg.get("temporal_xor_threshold", 0.06)
                ),
                window=int(segmentation_cfg.get("temporal_blend_window", 8)),
            )
            filled_hole_pixels += fill_enclosed_mask_holes(masks)
        mask_report = mask_statistics(
            masks,
            info["frames"],
            keyframe_stride=key_stride,
            keyframe_ids=segmentation_seed_ids,
        )
        mask_report["filled_enclosed_hole_pixels"] = filled_hole_pixels
        mask_report["temporal_stabilize"] = stabilize_report
        mask_report["onset_refinement"] = onset_report
        mask_report["miss_refinement"] = miss_report
        if mask_report["files"] != info["frames"] or mask_report["mean_coverage"] <= 0:
            raise RuntimeError(f"Invalid foreground masks: {mask_report}")
        status["stages"]["segmentation"] = {
            "state": "complete",
            **mask_report,
        }
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
        write_mask_overlay(input_video, masks, output_dir / "mask_overlay.mp4")
        if stop_after == "segmentation":
            status["state"] = "stopped_after_segmentation"
            status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
            print(f"Segmentation complete: {output_dir}")
            return

        slam_dir = output_dir / "slam"
        slam_result = SLAMAdapter(cfg["slam"], PROJECT_ROOT).run(
            all_frames, slam_dir, masks
        )
        import open3d as o3d

        cloud = o3d.io.read_point_cloud(str(slam_result["pointcloud"]))
        points = np.asarray(cloud.points)
        colors = np.asarray(cloud.colors)
        if not len(points):
            raise RuntimeError("VGGT returned an empty background point cloud")
        save_pointcloud(points, output_dir / "pointcloud_background.ply", colors)
        status["stages"]["reconstruction"] = {
            "state": "complete",
            **slam_result["report"],
        }
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
        if stop_after == "reconstruction":
            status["state"] = "stopped_after_reconstruction"
            status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
            print(f"Reconstruction complete: {output_dir}")
            return

        geometry_cfg = cfg.get("geometry", {})
        opening_hints = _load_opening_hints(output_dir, cfg, all_frames, slam_result)
        geometry_report = build_mesh(
            points,
            colors,
            output_dir,
            geometry_cfg,
            reconstruction_path=slam_result["reconstruction"],
            opening_hints=opening_hints,
        )
        write_html(
            points,
            colors,
            output_dir / "interactive.html",
            mesh_path=output_dir / "background_mesh.ply",
            extrinsics=slam_result["extrinsics"],
        )
        status["stages"]["geometry"] = {
            "state": "complete",
            "opening_hints": len(opening_hints) if opening_hints else 0,
            **geometry_report,
        }
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
        if stop_after == "geometry":
            status["state"] = "stopped_after_geometry"
            status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
            print(f"Geometry complete: {output_dir}")
            return

        completion_cfg = cfg.get("video_completion", {})
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
        status["stages"]["inpainting_masks"] = {
            "state": "complete",
            **inpaint_report,
        }
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
        try:
            backend = completion_cfg.get("backend", "propainter")
            if backend == "propainter":
                video_report = ProPainterAdapter(completion_cfg, PROJECT_ROOT).run(
                    input_video,
                    all_frames,
                    inpaint_masks,
                    output_dir / "background_video.mp4",
                    info["fps"],
                )
            elif backend == "svor":
                video_report = SVORAdapter(completion_cfg, PROJECT_ROOT).run(
                    input_video,
                    all_frames,
                    inpaint_masks,
                    output_dir / "background_video.mp4",
                    info["fps"],
                )
            elif backend == "effecterase":
                video_report = EffectEraseAdapter(completion_cfg, PROJECT_ROOT).run(
                    input_video,
                    all_frames,
                    inpaint_masks,
                    output_dir / "background_video.mp4",
                    info["fps"],
                )
            else:
                raise RuntimeError(f"Unsupported video completion backend {backend!r}")
        except Exception as error:
            if not completion_cfg.get("allow_temporal_fallback", True):
                raise
            video_report = write_background_video(
                input_video,
                all_frames,
                inpaint_masks,
                output_dir / "background_video.mp4",
                dilation=cfg["segmentation"].get("mask_dilation_px", 7),
                temporal_offsets=completion_cfg.get(
                    "temporal_offsets", [15, -15, 30, -30, 60, -60]
                ),
            )
            video_report[f"{backend}_error"] = str(error)
        status["stages"]["video"] = {"state": "complete", **video_report}
        status["state"] = "complete"
        status["elapsed_seconds"] = time.time() - started
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
        print(f"Pipeline complete: {output_dir}")
    except Exception as error:
        status["state"] = "failed"
        status["error"] = f"{type(error).__name__}: {error}"
        status["elapsed_seconds"] = time.time() - started
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
        raise


def doctor(cfg):
    checks = []
    required_files = [
        cfg["input_video"],
        cfg["segmentation"]["sam3_checkpoint"],
        cfg["segmentation"]["sam2_checkpoint"],
        cfg["slam"]["checkpoint"],
    ]
    for value in required_files:
        path = (PROJECT_ROOT / value).resolve()
        checks.append({"check": str(path), "ok": path.exists()})
    # ProPainterAdapter actually loads weights from the repo_dir, not from
    # checkpoints/, so doctor must validate the same locations the stage uses.
    repo_dir = (PROJECT_ROOT / cfg["video_completion"].get(
        "repo_dir", "external/ProPainter")).resolve()
    for value in (
        repo_dir / "inference_propainter.py",
        repo_dir / "weights" / "ProPainter.pth",
        repo_dir / "weights" / "recurrent_flow_completion.pth",
        repo_dir / "weights" / "raft-things.pth",
    ):
        checks.append({"check": str(value), "ok": value.exists()})
    if cfg["slam"].get("backend") == "vggt_slam":
        for value in (
            cfg["slam"].get(
                "salad_checkpoint", "checkpoints/salad/dino_salad.ckpt"
            ),
            cfg["slam"].get(
                "dinov2_checkpoint", "checkpoints/dinov2/dinov2_vitb14_pretrain.pth"
            ),
        ):
            path = (PROJECT_ROOT / value).resolve()
            checks.append({"check": str(path), "ok": path.exists()})
    env_commands = {
        cfg["segmentation"].get("environment", "vbr-seg"): (
            "import av,cv2,scipy,skimage,torch,sam3,sam2; import sam2._C; "
            "assert torch.cuda.is_available(); print(torch.__version__)"
        ),
        cfg["slam"].get("environment", "vbr-slam"): (
            "import torch,vggt,vggt_slam,gtsam,open3d,salad; "
            "assert torch.cuda.is_available() and hasattr(gtsam,'SL4'); print(torch.__version__)"
        ),
    }
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(
        cfg["segmentation"].get("cuda_visible_devices", "7")
    )
    for env_name, code in env_commands.items():
        result = subprocess.run(
            ["conda", "run", "-n", env_name, "python", "-c", code],
            cwd=PROJECT_ROOT,
            env=environment,
            text=True,
            capture_output=True,
        )
        checks.append(
            {
                "check": f"conda:{env_name}",
                "ok": result.returncode == 0,
                "detail": (result.stdout + result.stderr).strip()[-1000:],
            }
        )
    try:
        ffmpeg = _resolve_ffmpeg(cfg.get("video_completion", {}).get("ffmpeg_bin"))
        checks.append({"check": "ffmpeg:libx264", "ok": True, "detail": ffmpeg})
    except RuntimeError as error:
        checks.append({"check": "ffmpeg:libx264", "ok": False, "detail": str(error)})
    print(json.dumps(checks, indent=2))
    if not all(item["ok"] for item in checks):
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--config", default="configs/default.yaml")
    run_parser.add_argument(
        "--stop-after",
        choices=["segmentation", "reconstruction", "geometry", "all"],
        default="all",
    )
    run_parser.add_argument("--force", action="store_true")
    run_parser.add_argument(
        "--refine-onsets",
        action="store_true",
        help="Re-run first-appearance refinement on existing segmentation masks",
    )
    run_parser.add_argument(
        "--refine-misses",
        action="store_true",
        help="Re-run sustained low-coverage window refinement on existing masks",
    )
    doctor_parser = subparsers.add_parser("doctor")
    doctor_parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    if args.command == "run":
        run(
            load_config(args.config),
            stop_after=args.stop_after,
            force=args.force,
            refine_onsets=args.refine_onsets,
            refine_misses=args.refine_misses,
        )
    elif args.command == "doctor":
        doctor(load_config(args.config))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
