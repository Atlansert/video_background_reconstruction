"""Generate semantic foreground masks on sparse keyframes with SAM 3.1."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import cv2
import numpy as np


def _numeric_paths(frame_dir: Path) -> list[Path]:
    paths = list(frame_dir.glob("*.jpg")) + list(frame_dir.glob("*.jpeg"))
    return sorted(paths, key=lambda path: int(path.stem))


def _slug(text: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
    return value or "prompt"


def _union_output_masks(outputs: dict, shape: tuple[int, int]) -> np.ndarray:
    masks = outputs.get("out_binary_masks")
    if masks is None:
        return np.zeros(shape, dtype=np.uint8)
    if hasattr(masks, "detach"):
        masks = masks.detach().cpu().numpy()
    masks = np.asarray(masks)
    while masks.ndim > 3 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.size == 0:
        return np.zeros(shape, dtype=np.uint8)
    if masks.ndim == 2:
        masks = masks[None]
    union = np.any(masks.astype(bool), axis=0).astype(np.uint8) * 255
    if union.shape != shape:
        union = cv2.resize(union, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return union


def _box_to_xywh(normalized_box: list) -> list:
    """Normalized [x1, y1, x2, y2] to the SAM xywh (0..1) convention."""
    x1, y1, x2, y2 = (float(value) for value in normalized_box)
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    return [x1, y1, min(1.0 - x1, max(0.0, x2 - x1)), min(1.0 - y1, max(0.0, y2 - y1))]


def run(args: argparse.Namespace) -> None:
    from sam3.model_builder import build_sam3_predictor

    frame_dir = Path(args.frames).resolve()
    out_dir = Path(args.output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    prompt_dir = out_dir / "by_prompt"
    prompt_dir.mkdir(parents=True, exist_ok=True)

    frame_paths = _numeric_paths(frame_dir)
    if args.max_frames:
        frame_paths = frame_paths[: args.max_frames]
    if not frame_paths:
        raise RuntimeError(f"No numeric JPEG frames found in {frame_dir}")

    prompts = json.loads(args.prompts_json)
    preserve_prompts = json.loads(args.preserve_prompts_json)
    prompt_thresholds = json.loads(args.prompt_thresholds_json)
    box_prompts = json.loads(args.box_prompts_json) if args.box_prompts_json else []
    if not isinstance(prompts, list) or (not prompts and not args.allow_empty_union):
        raise ValueError("--prompts-json must contain a non-empty JSON list")
    if not isinstance(preserve_prompts, list):
        raise ValueError("--preserve-prompts-json must contain a JSON list")
    if not isinstance(prompt_thresholds, dict):
        raise ValueError("--prompt-thresholds-json must contain a JSON object")

    first = cv2.imread(str(frame_paths[0]))
    if first is None:
        raise RuntimeError(f"Cannot read {frame_paths[0]}")
    height, width = first.shape[:2]
    union_masks = [np.zeros((height, width), dtype=np.uint8) for _ in frame_paths]
    preserve_masks = [np.zeros_like(mask) for mask in union_masks]
    prior_report = {}
    if args.append_existing:
        report_path = out_dir / "sam31_report.json"
        if report_path.exists():
            prior_report = json.loads(report_path.read_text(encoding="utf-8"))
        for index, frame_path in enumerate(frame_paths):
            existing = cv2.imread(
                str(out_dir / f"{frame_path.stem}.png"), cv2.IMREAD_GRAYSCALE
            )
            preserved = cv2.imread(
                str(out_dir / "preserved" / f"{frame_path.stem}.png"),
                cv2.IMREAD_GRAYSCALE,
            )
            if existing is not None:
                union_masks[index] = existing
            if preserved is not None:
                preserve_masks[index] = preserved

    started = time.time()
    predictor = build_sam3_predictor(
        checkpoint_path=str(Path(args.checkpoint).resolve()),
        version="sam3.1",
        compile=False,
        warm_up=False,
        max_num_objects=args.max_objects,
        multiplex_count=16,
        use_fa3=False,
        use_rope_real=True,
        async_loading_frames=False,
    )

    def process_prompts(
        prompt_values: list[str], target_masks: list[np.ndarray], kind: str
    ) -> list[dict]:
        prompt_stats = []
        for prompt in prompt_values:
            threshold = float(prompt_thresholds.get(str(prompt), args.threshold))
            per_prompt = prompt_dir / kind / _slug(str(prompt))
            per_prompt.mkdir(parents=True, exist_ok=True)
            response = predictor.handle_request(
                {
                    "type": "start_session",
                    "resource_path": str(frame_dir),
                    "offload_video_to_cpu": True,
                }
            )
            session_id = response["session_id"]
            prompt_pixels = 0
            try:
                for local_idx, frame_path in enumerate(frame_paths):
                    response = predictor.handle_request(
                        {
                            "type": "add_prompt",
                            "session_id": session_id,
                            "frame_index": local_idx,
                            "text": str(prompt),
                            "output_prob_thresh": threshold,
                        }
                    )
                    mask = _union_output_masks(response["outputs"], (height, width))
                    target_masks[local_idx] = cv2.bitwise_or(target_masks[local_idx], mask)
                    prompt_pixels += int(np.count_nonzero(mask))
                    if args.keep_prompt_masks:
                        cv2.imwrite(str(per_prompt / f"{frame_path.stem}.png"), mask)
            finally:
                predictor.handle_request(
                    {
                        "type": "close_session",
                        "session_id": session_id,
                        "run_gc_collect": True,
                    }
                )
            prompt_stats.append(
                {
                    "prompt": str(prompt),
                    "threshold": threshold,
                    "masked_pixels": prompt_pixels,
                    "mean_coverage": prompt_pixels / (len(frame_paths) * height * width),
                }
            )
            print(
                f"SAM 3.1 {kind} prompt {prompt!r}: "
                f"{prompt_stats[-1]['mean_coverage']:.3%} coverage"
            )
        return prompt_stats

    def process_box_prompts(box_values: list[dict], target_masks: list[np.ndarray]) -> list[dict]:
        """Segment objects inside normalized boxes over explicit frame ranges.

        Each entry: {"start", "end", "box": [x1, y1, x2, y2] (0..1), optional
        "prompt" text hint while the box constrains the region}. Box prompts
        are the fallback for objects text prompts persistently miss (the
        sofa family in this room), and their masks still refine to the model's
        silhouette rather than the raw rectangle.
        """
        box_stats = []
        for spec in box_values:
            try:
                start = int(spec["start"])
                end = int(spec["end"])
                box_xywh = _box_to_xywh(spec["box"])
            except (KeyError, TypeError, ValueError) as error:
                raise RuntimeError(f"Invalid box prompt spec {spec!r}: {error}") from error
            text_hint = spec.get("prompt")
            threshold = float(
                prompt_thresholds.get(str(text_hint), args.threshold)
                if text_hint else args.threshold
            )
            response = predictor.handle_request(
                {
                    "type": "start_session",
                    "resource_path": str(frame_dir),
                    "offload_video_to_cpu": True,
                }
            )
            session_id = response["session_id"]
            prompt_pixels = 0
            try:
                for local_idx, frame_path in enumerate(frame_paths):
                    if not (start <= local_idx <= end):
                        continue
                    request = {
                        "type": "add_prompt",
                        "session_id": session_id,
                        "frame_index": local_idx,
                        "bounding_boxes": [box_xywh],
                        "bounding_box_labels": [1],
                        "output_prob_thresh": threshold,
                    }
                    if text_hint:
                        request["text"] = str(text_hint)
                    response = predictor.handle_request(request)
                    mask = _union_output_masks(response["outputs"], (height, width))
                    if np.any(mask):
                        target_masks[local_idx] = cv2.bitwise_or(
                            target_masks[local_idx], mask
                        )
                        prompt_pixels += int(np.count_nonzero(mask))
            finally:
                predictor.handle_request(
                    {
                        "type": "close_session",
                        "session_id": session_id,
                        "run_gc_collect": True,
                    }
                )
            box_stats.append(
                {
                    "prompt": str(text_hint or "box"),
                    "start": start,
                    "end": end,
                    "box": [round(value, 3) for value in box_xywh],
                    "masked_pixels": prompt_pixels,
                }
            )
            print(
                f"SAM 3.1 box prompt {'(hint: ' + str(text_hint) + ')' if text_hint else ''} "
                f"frames {start}-{end}: {prompt_pixels} pixels"
            )
        return box_stats

    added_stats = process_prompts(prompts, union_masks, "remove")
    added_preserve_stats = process_prompts(preserve_prompts, preserve_masks, "preserve")
    added_box_stats = process_box_prompts(box_prompts, union_masks)
    stats = prior_report.get("remove_prompts", []) + added_stats
    preserve_stats = prior_report.get("preserve_prompts", []) + added_preserve_stats

    preserve_dir = out_dir / "preserved"
    preserve_dir.mkdir(parents=True, exist_ok=True)
    protect_kernel = np.ones((args.preserve_dilation, args.preserve_dilation), np.uint8)
    for index in range(len(union_masks)):
        protected = cv2.dilate(preserve_masks[index], protect_kernel)
        union_masks[index][protected > 0] = 0

    coverages = []
    for frame_path, mask in zip(frame_paths, union_masks):
        cv2.imwrite(str(out_dir / f"{frame_path.stem}.png"), mask)
        cv2.imwrite(
            str(preserve_dir / f"{frame_path.stem}.png"),
            preserve_masks[len(coverages)],
        )
        coverages.append(float(np.count_nonzero(mask)) / mask.size)

    report = {
        "backend": "sam3.1",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "frames": len(frame_paths),
        "remove_prompts": stats,
        "preserve_prompts": preserve_stats,
        "box_prompts": added_box_stats,
        "mean_union_coverage": float(np.mean(coverages)),
        "max_union_coverage": float(np.max(coverages)),
        "nonempty_frames": int(np.count_nonzero(np.asarray(coverages) > 0)),
        "elapsed_seconds": prior_report.get("elapsed_seconds", 0) + time.time() - started,
        "incremental_runs": prior_report.get("incremental_runs", 0)
        + int(args.append_existing),
    }
    (out_dir / "sam31_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if report["nonempty_frames"] == 0 and not args.allow_empty_union:
        raise RuntimeError("SAM 3.1 produced no foreground masks for any configured prompt")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompts-json", required=True)
    parser.add_argument("--preserve-prompts-json", default="[]")
    parser.add_argument("--prompt-thresholds-json", default="{}")
    parser.add_argument("--box-prompts-json", default=None)
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--max-objects", type=int, default=64)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--preserve-dilation", type=int, default=7)
    parser.add_argument("--keep-prompt-masks", action="store_true")
    parser.add_argument("--append-existing", action="store_true")
    parser.add_argument(
        "--allow-empty-union",
        action="store_true",
        help="succeed even when the remove-prompt union is empty (used when "
        "only preserve prompts run, e.g. for wall-opening carving)",
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
