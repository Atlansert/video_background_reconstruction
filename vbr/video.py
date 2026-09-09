"""Video extraction, mask previews, and temporally assisted background filling."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


def video_info(path):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(path)
    info = {
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": float(capture.get(cv2.CAP_PROP_FPS)),
        "frames": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    capture.release()
    return info


def extract_frame_sets(path, all_dir, keyframe_dir, keyframe_stride=30, quality=95):
    all_dir = Path(all_dir)
    keyframe_dir = Path(keyframe_dir)
    all_dir.mkdir(parents=True, exist_ok=True)
    keyframe_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(path))
    frame_ids = []
    index = 0
    last_frame = None
    encode = [cv2.IMWRITE_JPEG_QUALITY, quality]
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        cv2.imwrite(str(all_dir / f"{index:06d}.jpg"), frame, encode)
        if index % keyframe_stride == 0:
            cv2.imwrite(str(keyframe_dir / f"{index:06d}.jpg"), frame, encode)
            frame_ids.append(index)
        last_frame = frame
        index += 1
    capture.release()
    if index and (index - 1) not in frame_ids:
        cv2.imwrite(str(keyframe_dir / f"{index - 1:06d}.jpg"), last_frame, encode)
        frame_ids.append(index - 1)
    return frame_ids


def _load_binary_mask(path: Path):
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    return mask > 0


def mask_statistics(
    masks_dir, expected_frames, keyframe_stride=None, keyframe_ids=None
):
    paths = sorted(Path(masks_dir).glob("*.png"), key=lambda path: int(path.stem))
    coverages = []
    xor_all = []
    xor_key = []
    xor_intra = []
    flicker_off = 0
    flicker_on = 0
    flicker_pixels = 0
    previous = None
    previous2 = None
    keyframe_set = {int(value) for value in keyframe_ids} if keyframe_ids is not None else None
    for path in paths:
        current = _load_binary_mask(path)
        if current is None:
            continue
        coverages.append(float(np.count_nonzero(current)) / current.size)
        if previous is not None:
            frac = float(np.logical_xor(previous, current).mean())
            xor_all.append(frac)
            frame_id = int(path.stem)
            at_key = (
                frame_id in keyframe_set
                if keyframe_set is not None
                else bool(keyframe_stride and frame_id % int(keyframe_stride) == 0)
            )
            if at_key:
                xor_key.append(frac)
            else:
                xor_intra.append(frac)
        if previous2 is not None:
            flicker_pixels += int(previous.size)
            flicker_off += int((previous2 & current & ~previous).sum())
            flicker_on += int((~previous2 & ~current & previous).sum())
        previous2 = previous
        previous = current
    mean_intra = float(np.mean(xor_intra)) if xor_intra else 0.0
    mean_key = float(np.mean(xor_key)) if xor_key else 0.0
    return {
        "files": len(paths),
        "expected_frames": expected_frames,
        "nonempty_frames": int(np.count_nonzero(np.asarray(coverages) > 0)),
        "mean_coverage": float(np.mean(coverages)) if coverages else 0.0,
        "max_coverage": float(np.max(coverages)) if coverages else 0.0,
        "mean_frame_xor": float(np.mean(xor_all)) if xor_all else 0.0,
        "max_frame_xor": float(np.max(xor_all)) if xor_all else 0.0,
        "keyframe_boundary_xor": mean_key,
        "intra_interval_xor": mean_intra,
        "keyframe_xor_ratio": mean_key / max(mean_intra, 1e-9),
        "flicker_off_fraction": flicker_off / flicker_pixels if flicker_pixels else 0.0,
        "flicker_on_fraction": flicker_on / flicker_pixels if flicker_pixels else 0.0,
    }


def detect_mask_onsets(
    masks_dir,
    min_new_area_px=1500,
    persistence_frames=3,
    dilation_px=9,
    cooldown_frames=10,
):
    """Find persistent, newly visible foreground components.

    These events identify where a sparse SAM 3.1 keyframe seed arrived too
    late. They are intentionally based on connected components rather than
    total coverage, so ordinary camera motion and small edge drift do not
    trigger a local semantic re-segmentation.
    """
    paths = sorted(Path(masks_dir).glob("*.png"), key=lambda path: int(path.stem))
    if len(paths) < persistence_frames + 1:
        return []
    masks = [_load_binary_mask(path) for path in paths]
    if any(mask is None for mask in masks):
        raise RuntimeError(f"Cannot read one or more masks in {masks_dir}")
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (max(1, int(dilation_px)), max(1, int(dilation_px)))
    )
    events = []
    index = 1
    while index < len(paths) - persistence_frames:
        previous = cv2.dilate(masks[index - 1].astype(np.uint8), kernel) > 0
        emerging = masks[index] & ~previous
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            emerging.astype(np.uint8), connectivity=8
        )
        candidates = []
        for component in range(1, count):
            area = int(stats[component, cv2.CC_STAT_AREA])
            if area < min_new_area_px:
                continue
            component_mask = labels == component
            component_dilated = cv2.dilate(component_mask.astype(np.uint8), kernel) > 0
            persistent = sum(
                bool((component_dilated & masks[future]).any())
                for future in range(index + 1, index + 1 + persistence_frames)
            )
            if persistent >= max(1, persistence_frames - 1):
                candidates.append((area, persistent))
        if candidates:
            new_area, persistent = max(candidates)
            events.append(
                {
                    "frame_id": int(paths[index].stem),
                    "frame_index": index,
                    "new_area_px": new_area,
                    "persistent_frames": persistent,
                }
            )
            index += max(1, int(cooldown_frames))
        else:
            index += 1
    return events


def detect_persistent_misses(
    masks_dir,
    expected_frames,
    min_coverage=0.20,
    min_window_frames=30,
    merge_gap_frames=12,
    max_windows=3,
):
    """Find sustained windows where foreground mask coverage stays low.

    A persistent miss means an object stays visible across many frames while
    the mask barely covers it (e.g. the 746-970 sofa), which lets ProPainter
    copy the object back from unmasked neighbors. Single-frame dips, brief
    occlusions and legitimately empty views must not trigger. Windows closer
    than ``merge_gap_frames`` are joined, then the largest windows are kept.
    """
    frame_count = expected_frames or 0
    coverages = np.zeros(frame_count, dtype=np.float64)
    by_name = {}
    for path in Path(masks_dir).glob("*.png"):
        frame_id = int(path.stem)
        if 0 <= frame_id < frame_count:
            by_name[frame_id] = path
    for frame_id in range(frame_count):
        path = by_name.get(frame_id)
        mask = _load_binary_mask(path) if path is not None else None
        if mask is not None:
            coverages[frame_id] = float(np.count_nonzero(mask)) / mask.size
    low = coverages < min_coverage

    runs = []
    start = None
    for frame_id in range(frame_count):
        if low[frame_id] and start is None:
            start = frame_id
        elif not low[frame_id] and start is not None:
            runs.append((start, frame_id - 1))
            start = None
    if start is not None:
        runs.append((start, frame_count - 1))

    merged = []
    for run in runs:
        if merged and run[0] - merged[-1][1] <= merge_gap_frames:
            merged[-1] = (merged[-1][0], run[1])
        else:
            merged.append(run)

    windows = []
    for start, end in merged:
        length = end - start + 1
        if length < min_window_frames:
            continue
        segment = coverages[start : end + 1]
        windows.append(
            {
                "start": int(start),
                "end": int(end),
                "frames": int(length),
                "mean_coverage": float(segment.mean()),
                "min_coverage": float(segment.min()),
            }
        )
    windows.sort(key=lambda window: -window["frames"])
    return windows[:max_windows]


def select_miss_seeds(
    refinement_dir,
    masks_dir,
    min_added_area_px=2000,
    min_keep_fraction=0.5,
    max_coverage_increase=None,
    evidence=None,
    evidence_min_overlap=0.5,
):
    """Choose refined window masks worth pinning as propagation seeds.

    A refined mask is kept only when it adds substantial new foreground area
    beyond the current propagated mask and does not collapse that frame's
    existing coverage (candidate smaller than ``min_keep_fraction`` of the
    current mask is rejected, as is a near-duplicate of it). With
    ``max_coverage_increase`` set, candidates whose coverage grows more than
    the current mask plus that fraction are rejected too, UNLESS most of the
    new area overlaps ``evidence`` (copy-through residual regions where the
    current video demonstrably still shows the object) — then the growth was
    justified and the precision cap is waived.
    """
    refinement_dir = Path(refinement_dir)
    masks_dir = Path(masks_dir)
    accepted = []
    rejected = 0
    total_added = 0
    waived = 0
    for path in sorted(refinement_dir.glob("*.png"), key=lambda p: int(p.stem)):
        frame_id = int(path.stem)
        candidate = _load_binary_mask(path)
        if candidate is None or not np.any(candidate):
            continue
        current = _load_binary_mask(masks_dir / path.name)
        current = current if current is not None else np.zeros_like(candidate)
        new_area = candidate & ~current
        added = int(np.count_nonzero(new_area))
        if added < min_added_area_px:
            rejected += 1
            continue
        if np.any(current) and np.count_nonzero(candidate) < min_keep_fraction * np.count_nonzero(current):
            rejected += 1
            continue
        if max_coverage_increase is not None:
            current_coverage = float(np.count_nonzero(current)) / current.size
            candidate_coverage = float(np.count_nonzero(candidate)) / candidate.size
            growth = candidate_coverage - current_coverage
            if growth > float(max_coverage_increase):
                residual = evidence.get(frame_id) if evidence else None
                if residual is not None:
                    overlap = float((new_area & residual).sum()) / max(1, added)
                    if overlap >= float(evidence_min_overlap):
                        waived += 1
                    else:
                        rejected += 1
                        continue
                else:
                    rejected += 1
                    continue
        accepted.append(int(path.stem))
        total_added += added
    return accepted, {
        "accepted_stems": accepted,
        "rejected_frames": rejected,
        "total_added_px": total_added,
        "coverage_cap_waived": waived,
    }


def derive_box_seeds(
    masks_dir,
    windows,
    subwindow_frames=25,
    expansion=0.4,
    min_component_fraction=0.02,
):
    """Build normalized box prompts from the masks inside miss windows.

    For each subwindow of each persistent-miss window the largest foreground
    component's bounding box is expanded (the detected sliver is usually one
    edge of the missed object) and rescaled to normalized [x1, y1, x2, y2].
    Subwindows keep the box tracking the camera pan; components too small to
    matter are skipped.
    """
    frame_size = None
    by_frame = {}
    for path in Path(masks_dir).glob("*.png"):
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        by_frame[int(path.stem)] = mask
        if frame_size is None:
            frame_size = mask.shape
    if frame_size is None:
        return []
    height, width = frame_size
    min_area = min_component_fraction * height * width
    seeds = []
    for window in windows:
        start, end = int(window["start"]), int(window["end"])
        subwindow_start = start
        while subwindow_start <= end:
            subwindow_end = min(end, subwindow_start + subwindow_frames - 1)
            union = np.zeros((height, width), dtype=bool)
            for frame_id in range(subwindow_start, subwindow_end + 1):
                mask = by_frame.get(frame_id)
                if mask is not None:
                    union |= mask > 0
            count, labels, stats, _ = cv2.connectedComponentsWithStats(
                union.astype(np.uint8), connectivity=8
            )
            largest = None
            for component in range(1, count):
                area = int(stats[component, cv2.CC_STAT_AREA])
                if area < min_area:
                    continue
                if largest is None or area > largest[0]:
                    largest = (
                        area,
                        int(stats[component, cv2.CC_STAT_LEFT]),
                        int(stats[component, cv2.CC_STAT_TOP]),
                        int(stats[component, cv2.CC_STAT_WIDTH]),
                        int(stats[component, cv2.CC_STAT_HEIGHT]),
                    )
            if largest is not None:
                _, left, top, box_width, box_height = largest
                x1 = max(0.0, (left - expansion * box_width) / width)
                y1 = max(0.0, (top - expansion * box_height) / height)
                x2 = min(1.0, (left + box_width + expansion * box_width) / width)
                y2 = min(1.0, (top + box_height + expansion * box_height) / height)
                seeds.append(
                    {
                        "start": subwindow_start,
                        "end": subwindow_end,
                        "box": [round(x1, 3), round(y1, 3), round(x2, 3), round(y2, 3)],
                    }
                )
            subwindow_start = subwindow_end + 1
    return seeds


def copy_through_evidence(
    original_video,
    background_video,
    masks_dir,
    frame_ids,
    threshold=11.0,
    dilate_px=5,
):
    """Copy-through residual regions for the queried frames.

    Pixels INSIDE the (eroded) inpaint masks where the current background
    video is still nearly identical to the original frame mark parts of the
    object that survived inpainting. These are the regions a new seed must
    remove, used to waive the coverage cap for seed candidates that
    demonstrably target them.
    """
    wanted = set(int(frame_id) for frame_id in frame_ids)
    originals = {}
    for index, frame in enumerate(_iter_video_frames(original_video)):
        if index in wanted:
            originals[index] = frame
        if len(originals) >= len(wanted):
            break
    backgrounds = {}
    for index, frame in enumerate(_iter_video_frames(background_video)):
        if index in wanted:
            backgrounds[index] = frame
        if len(backgrounds) >= len(wanted):
            break
    kernel = np.ones((max(1, dilate_px), max(1, dilate_px)), np.uint8)
    evidence = {}
    for frame_id in wanted:
        if frame_id not in originals or frame_id not in backgrounds:
            continue
        mask = cv2.imread(
            str(Path(masks_dir) / f"{frame_id:06d}.png"), cv2.IMREAD_GRAYSCALE
        )
        if mask is None:
            continue
        inner = cv2.erode((mask > 0).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        if not inner.any():
            continue
        diff = cv2.absdiff(originals[frame_id], backgrounds[frame_id]).astype(
            np.float32
        ).mean(axis=2)
        residual = (diff < threshold) & inner
        if residual.any():
            evidence[frame_id] = cv2.dilate(
                residual.astype(np.uint8), kernel
            ) > 0
    return evidence


def _iter_video_frames(video_path):
    capture = cv2.VideoCapture(str(video_path))
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            yield frame
    finally:
        capture.release()


def onset_latency_metrics(
    baseline_dir,
    refined_dir,
    events,
    dilation_px=9,
    overlap_fraction=0.5,
    window_back=12,
    window_forward=6,
):
    """Measure how many frames earlier the refined masks first cover each onset.

    The event region is the emerging component at the event frame in the
    baseline masks; latency is the gap between the baseline first hit (the
    event frame itself) and the first refined frame covering the region.
    """
    baseline_paths = sorted(Path(baseline_dir).glob("*.png"), key=lambda p: int(p.stem))
    if not baseline_paths:
        return {"events": [], "mean_latency_frames": 0.0, "improved_events": 0}
    baseline = {int(path.stem): _load_binary_mask(path) for path in baseline_paths}
    refined_paths = sorted(Path(refined_dir).glob("*.png"), key=lambda p: int(p.stem))
    refined = {int(path.stem): _load_binary_mask(path) for path in refined_paths}
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (max(1, int(dilation_px)), max(1, int(dilation_px)))
    )
    records = []
    for event in events:
        frame_id = int(event["frame_id"])
        current = baseline.get(frame_id)
        previous = baseline.get(frame_id - 1)
        if current is None or previous is None:
            continue
        emerging = current & ~cv2.dilate(previous.astype(np.uint8), kernel)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            emerging.astype(np.uint8), connectivity=8
        )
        if count <= 1:
            continue
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        region = cv2.dilate((labels == largest).astype(np.uint8), kernel) > 0
        target = overlap_fraction * max(1, int(region.sum()))

        def first_hit(masks):
            for candidate in range(
                max(0, frame_id - int(window_back)),
                min(frame_id + int(window_forward) + 1, max(masks) + 1 if masks else 0),
            ):
                mask = masks.get(candidate)
                if mask is not None and int((region & mask).sum()) >= target:
                    return candidate
            return frame_id

        refined_first = first_hit(refined)
        records.append(
            {
                "frame_id": frame_id,
                "baseline_first_frame": frame_id,
                "refined_first_frame": refined_first,
                "latency_frames": frame_id - refined_first,
            }
        )
    latencies = [record["latency_frames"] for record in records]
    return {
        "events": records,
        "mean_latency_frames": float(np.mean(latencies)) if latencies else 0.0,
        "improved_events": int(sum(value > 0 for value in latencies)),
    }


def _as_binary_stack(masks):
    return np.stack([np.asarray(mask) > 0 for mask in masks], axis=0)


def _to_mask_images(stacked):
    return [(frame.astype(np.uint8) * 255) for frame in stacked]


def stabilize_mask_sequence(masks, pin_indices=None):
    """Remove 1-frame on/off flicker. Pinned frames keep their original pixels."""
    if len(masks) < 3:
        return [np.where(mask > 0, 255, 0).astype(np.uint8) for mask in masks], {
            "changed_frames": 0,
            "changed_pixels": 0,
        }
    stacked = _as_binary_stack(masks)
    original = stacked.copy()
    pin = {int(index) for index in (pin_indices or [])}
    previous = stacked[:-2]
    current = stacked[1:-1]
    nxt = stacked[2:]
    updated = current.copy()
    updated[current & ~previous & ~nxt] = False
    updated[~current & previous & nxt] = True
    for offset in range(updated.shape[0]):
        frame_index = offset + 1
        if frame_index in pin:
            continue
        stacked[frame_index] = updated[offset]
    changed = stacked != original
    report = {
        "changed_frames": int(np.any(changed.reshape(len(masks), -1), axis=1).sum()),
        "changed_pixels": int(changed.sum()),
    }
    return _to_mask_images(stacked), report


def _signed_distance(mask):
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return np.full(mask.shape, -1.0e6, dtype=np.float32)
    if mask.all():
        return np.full(mask.shape, 1.0e6, dtype=np.float32)
    foreground = mask.astype(np.uint8)
    inside = cv2.distanceTransform(foreground, cv2.DIST_L2, 5)
    outside = cv2.distanceTransform(1 - foreground, cv2.DIST_L2, 5)
    return inside - outside


def _morph_interval(result, start, end, pin):
    """Interpolate unpinned frames in (start, end) toward the endpoint masks."""
    if end - start < 2:
        return 0
    sdf_start = _signed_distance(result[start])
    sdf_end = _signed_distance(result[end])
    span = float(end - start)
    changed = 0
    for index in range(start + 1, end):
        if index in pin:
            continue
        alpha = (index - start) / span
        morphed = ((1.0 - alpha) * sdf_start + alpha * sdf_end) >= 0
        if np.any(morphed != result[index]):
            result[index] = morphed
            changed += 1
    return changed


def blend_mask_jumps(masks, pin_indices=None, xor_threshold=0.06, window=8):
    """Spread large frame-to-frame mask pops over a short temporal window.

    SAM 3.1 keyframes stay pinned. A jump into a pinned frame is ramped in from
    the previous unpinned frames; a jump out of a pinned frame is ramped out;
    an interior snap is blended from both sides. Small motion below
    ``xor_threshold`` is left untouched.
    """
    if len(masks) < 3 or window < 2 or xor_threshold <= 0:
        return [np.where(mask > 0, 255, 0).astype(np.uint8) for mask in masks], {
            "jumps": 0,
            "changed_frames": 0,
        }
    result = _as_binary_stack(masks)
    original = result.copy()
    pin = {int(index) for index in (pin_indices or [])}
    jumps = 0
    index = 1
    n_frames = len(result)
    while index < n_frames:
        xor_frac = float(np.logical_xor(result[index - 1], result[index]).mean())
        if xor_frac < xor_threshold:
            index += 1
            continue
        left, right = index - 1, index
        left_pinned = left in pin
        right_pinned = right in pin
        prev_pin = max((item for item in pin if item < right), default=0)
        next_pin = min((item for item in pin if item > left), default=n_frames - 1)
        if right_pinned and not left_pinned:
            start = max(0, prev_pin, right - window)
            _morph_interval(result, start, right, pin)
        elif left_pinned and not right_pinned:
            end = min(n_frames - 1, next_pin, left + window)
            _morph_interval(result, left, end, pin)
        elif not left_pinned and not right_pinned:
            start = max(0, prev_pin, left - window // 2)
            end = min(n_frames - 1, next_pin, right + window // 2)
            _morph_interval(result, start, end, pin)
        jumps += 1
        index = right + 1
    changed = result != original
    report = {
        "jumps": jumps,
        "changed_frames": int(np.any(changed.reshape(n_frames, -1), axis=1).sum()),
        "changed_pixels": int(changed.sum()),
        "xor_threshold": float(xor_threshold),
        "window": int(window),
    }
    return _to_mask_images(result), report


def stabilize_foreground_masks(
    masks_dir,
    pin_stems=None,
    xor_threshold=0.06,
    window=8,
):
    """Drop 1-frame flashes and ramp large pops, keeping SAM 3.1 keyframes pinned."""
    paths = sorted(Path(masks_dir).glob("*.png"), key=lambda path: int(path.stem))
    if len(paths) < 3:
        return {
            "changed_frames": 0,
            "changed_pixels": 0,
            "pinned_frames": 0,
            "jumps": 0,
        }
    pin_stems = {int(stem) for stem in (pin_stems or [])}
    masks = []
    pin_indices = []
    for index, path in enumerate(paths):
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Cannot read mask {path}")
        masks.append(mask)
        if int(path.stem) in pin_stems:
            pin_indices.append(index)
    updated, flicker = stabilize_mask_sequence(masks, pin_indices=pin_indices)
    updated, jumps = blend_mask_jumps(
        updated,
        pin_indices=pin_indices,
        xor_threshold=xor_threshold,
        window=window,
    )
    for path, mask in zip(paths, updated):
        cv2.imwrite(str(path), mask)
    return {
        "changed_frames": int(flicker["changed_frames"] + jumps["changed_frames"]),
        "changed_pixels": int(flicker["changed_pixels"] + jumps.get("changed_pixels", 0)),
        "pinned_frames": len(pin_indices),
        "flicker_frames": flicker["changed_frames"],
        "jumps": jumps["jumps"],
        "jump_frames": jumps["changed_frames"],
    }


def fill_mask_convex_hulls(mask, min_area=2500, max_extra=0.35):
    """Fill concave interiors of large components without swallowing the room."""
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    filled = binary.copy()
    added = 0
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        ys, xs = np.where(labels == index)
        hull = cv2.convexHull(np.column_stack([xs, ys]))
        layer = np.zeros_like(binary)
        cv2.fillConvexPoly(layer, hull, 1)
        extra = (layer == 1) & (filled == 0)
        extra_count = int(extra.sum())
        if extra_count / max(area, 1) > max_extra:
            continue
        filled[layer == 1] = 1
        added += extra_count
    return filled, added


def _temporal_union(
    current,
    neighbors,
    radius,
    gated=False,
    min_overlap=0.12,
    dilate_px=21,
):
    """OR-merge neighbor masks into current.

    Neighbors must be the pre-union detections. Using already-unioned frames
    as neighbors lets empty in-between frames bridge unrelated objects.
    """
    if radius <= 0:
        return current
    union = current.copy()
    if not gated:
        for offset in range(1, radius + 1):
            union[:-offset] = np.maximum(union[:-offset], neighbors[offset:])
            union[offset:] = np.maximum(union[offset:], neighbors[:-offset])
        return union
    kernel = (
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px, dilate_px))
        if dilate_px
        else None
    )
    frame_area = union.shape[1] * union.shape[2]
    dilated = np.empty_like(union)
    for index, frame in enumerate(union):
        dilated[index] = cv2.dilate(frame, kernel) if kernel is not None else frame
    for offset in range(1, radius + 1):
        for neighbor_slice, current_slice, dilated_slice in (
            (neighbors[offset:], union[:-offset], dilated[:-offset]),
            (neighbors[:-offset], union[offset:], dilated[offset:]),
        ):
            overlap = (dilated_slice & neighbor_slice).reshape(len(neighbor_slice), -1).sum(1)
            area_current = current_slice.reshape(len(current_slice), -1).sum(1)
            area_neighbor = neighbor_slice.reshape(len(neighbor_slice), -1).sum(1)
            denom = np.minimum(np.maximum(area_current, 1), np.maximum(area_neighbor, 1))
            accept = overlap >= (min_overlap * denom)
            accept |= (area_current == 0) & (area_neighbor < 0.55 * frame_area)
            accept &= area_current < 0.55 * frame_area
            if not np.any(accept):
                continue
            current_slice[accept] = np.maximum(
                current_slice[accept], neighbor_slice[accept]
            )
    return union


def prepare_inpainting_masks(
    source_dir,
    output_dir,
    close_px=9,
    temporal_radius=2,
    dilate_px=1,
    hull_min_area=0,
    hull_max_extra=0.35,
    overlap=0.12,
):
    """Build conservative masks for video completion.

    Reconstruction masks stay tight. Inpainting needs the opposite: if an
    object is detected in a nearby frame, hide it here too, otherwise
    ProPainter copies the original object through the hole and it flickers.
    """
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(source_dir.glob("*.png"), key=lambda path: int(path.stem))
    if not paths:
        raise RuntimeError(f"No masks to expand in {source_dir}")
    close_px = max(0, int(close_px))
    temporal_radius = max(0, int(temporal_radius))
    dilate_px = max(0, int(dilate_px))
    hull_min_area = max(0, int(hull_min_area))
    hull_max_extra = float(hull_max_extra)
    close_kernel = (
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px, close_px))
        if close_px
        else None
    )
    dilate_kernel = (
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px, dilate_px))
        if dilate_px
        else None
    )
    stacked = []
    hull_pixels = 0
    source_sum = 0.0
    for path in paths:
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Cannot read mask {path}")
        source = (mask > 0).astype(np.uint8)
        source_sum += float(source.mean())
        binary = source
        if close_kernel is not None:
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel)
        if hull_min_area:
            binary, added = fill_mask_convex_hulls(
                binary, min_area=hull_min_area, max_extra=hull_max_extra
            )
            hull_pixels += added
        stacked.append(binary)
    stacked = np.stack(stacked, axis=0)
    local_radius = min(8, temporal_radius)
    union = _temporal_union(stacked, stacked, local_radius, gated=False)
    if temporal_radius > local_radius:
        union = _temporal_union(
            union,
            stacked,
            temporal_radius,
            gated=True,
            min_overlap=float(overlap),
        )
    mean_source = source_sum / len(paths)
    coverages = []
    extra_coverages = []
    for index, path in enumerate(paths):
        mask = union[index]
        if dilate_kernel is not None:
            mask = cv2.dilate(mask, dilate_kernel)
        source = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) > 0
        coverages.append(float(mask.mean()))
        extra_coverages.append(float((mask.astype(bool) & ~source).mean()))
        cv2.imwrite(str(output_dir / path.name), mask.astype(np.uint8) * 255)
    mean_after = float(np.mean(coverages))
    mean_extra = float(np.mean(extra_coverages))
    return {
        "frames": len(paths),
        "close_px": close_px,
        "temporal_radius": temporal_radius,
        "dilate_px": dilate_px,
        "hull_min_area": hull_min_area,
        "hull_pixels": hull_pixels,
        "overlap": float(overlap),
        "mean_coverage_source": mean_source,
        "mean_coverage": mean_after,
        "mean_extra_coverage": mean_extra,
        "p90_coverage": float(np.percentile(coverages, 90)),
        "max_coverage": float(np.max(coverages)),
        "extra_coverage_ratio": mean_extra / max(mean_source, 1e-9),
    }


def fill_enclosed_mask_holes(masks_dir):
    """Fill background islands fully enclosed by a foreground mask."""
    changed_pixels = 0
    paths = sorted(Path(masks_dir).glob("*.png"), key=lambda path: int(path.stem))
    for path in paths:
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        binary = np.where(mask > 0, 255, 0).astype(np.uint8)
        padded = cv2.copyMakeBorder(binary, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        flood = padded.copy()
        cv2.floodFill(flood, None, (0, 0), 255)
        holes = (flood[1:-1, 1:-1] == 0) & (binary == 0)
        changed_pixels += int(np.count_nonzero(holes))
        binary[holes] = 255
        cv2.imwrite(str(path), binary)
    return changed_pixels


def write_mask_overlay(video_path, masks_dir, output_path, opacity=0.55):
    capture = cv2.VideoCapture(str(video_path))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        mask = cv2.imread(
            str(Path(masks_dir) / f"{index:06d}.png"), cv2.IMREAD_GRAYSCALE
        )
        if mask is not None:
            red = np.zeros_like(frame)
            red[:, :, 2] = 255
            selected = mask > 0
            frame[selected] = (
                frame[selected] * (1 - opacity) + red[selected] * opacity
            ).astype(np.uint8)
        writer.write(frame)
        index += 1
    capture.release()
    writer.release()


class _FeatureCache:
    def __init__(self, frame_paths, masks_dir, scale=0.5):
        self.frame_paths = frame_paths
        self.masks_dir = Path(masks_dir)
        self.scale = scale
        self.detector = cv2.ORB_create(nfeatures=1400, fastThreshold=12)
        self.cache = {}

    def get(self, index):
        if index in self.cache:
            return self.cache[index]
        image = cv2.imread(str(self.frame_paths[index]), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(
            str(self.masks_dir / f"{int(self.frame_paths[index].stem):06d}.png"),
            cv2.IMREAD_GRAYSCALE,
        )
        small = cv2.resize(image, None, fx=self.scale, fy=self.scale)
        valid = None
        if mask is not None:
            valid = cv2.resize(mask, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST)
            valid = np.where(valid > 0, 0, 255).astype(np.uint8)
        keypoints, descriptors = self.detector.detectAndCompute(small, valid)
        points = (
            np.asarray([keypoint.pt for keypoint in keypoints], dtype=np.float32) / self.scale
            if keypoints
            else np.empty((0, 2), dtype=np.float32)
        )
        self.cache[index] = (points, descriptors)
        return self.cache[index]


def _estimate_homography(cache, source_index, target_index):
    source_points, source_descriptors = cache.get(source_index)
    target_points, target_descriptors = cache.get(target_index)
    if source_descriptors is None or target_descriptors is None:
        return None
    matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(
        source_descriptors, target_descriptors, k=2
    )
    good = [
        pair[0]
        for pair in matches
        if len(pair) == 2 and pair[0].distance < 0.72 * pair[1].distance
    ]
    if len(good) < 12:
        return None
    source = np.asarray([source_points[item.queryIdx] for item in good])
    target = np.asarray([target_points[item.trainIdx] for item in good])
    homography, inliers = cv2.findHomography(source, target, cv2.RANSAC, 4.0)
    if homography is None or inliers is None or int(inliers.sum()) < 10:
        return None
    return homography


def write_background_video(
    video_path,
    frames_dir,
    masks_dir,
    output_path,
    dilation=7,
    temporal_offsets=(15, -15, 30, -30, 60, -60),
):
    frame_paths = sorted(Path(frames_dir).glob("*.jpg"), key=lambda path: int(path.stem))
    if not frame_paths:
        raise RuntimeError("No extracted frames for background video")
    info = video_info(video_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        info["fps"] or 30,
        (info["width"], info["height"]),
    )
    cache = _FeatureCache(frame_paths, masks_dir)
    kernel = np.ones((max(1, dilation), max(1, dilation)), dtype=np.uint8)
    total_masked = 0
    temporal_filled = 0
    homographies = 0

    for target_index, frame_path in enumerate(frame_paths):
        frame = cv2.imread(str(frame_path))
        mask = cv2.imread(
            str(Path(masks_dir) / f"{frame_path.stem}.png"), cv2.IMREAD_GRAYSCALE
        )
        if mask is None or not np.any(mask):
            writer.write(frame)
            continue
        mask = cv2.dilate(mask, kernel)
        missing = mask > 0
        total_masked += int(np.count_nonzero(missing))
        result = frame.copy()
        for offset in temporal_offsets:
            source_index = target_index + int(offset)
            if source_index < 0 or source_index >= len(frame_paths) or not np.any(missing):
                continue
            homography = _estimate_homography(cache, source_index, target_index)
            if homography is None:
                continue
            source = cv2.imread(str(frame_paths[source_index]))
            source_mask = cv2.imread(
                str(Path(masks_dir) / f"{frame_paths[source_index].stem}.png"),
                cv2.IMREAD_GRAYSCALE,
            )
            source_valid = np.full(source.shape[:2], 255, dtype=np.uint8)
            if source_mask is not None:
                source_valid[source_mask > 0] = 0
            warped = cv2.warpPerspective(source, homography, (frame.shape[1], frame.shape[0]))
            valid = cv2.warpPerspective(
                source_valid,
                homography,
                (frame.shape[1], frame.shape[0]),
                flags=cv2.INTER_NEAREST,
            )
            fill = missing & (valid > 0)
            result[fill] = warped[fill]
            temporal_filled += int(np.count_nonzero(fill))
            missing[fill] = False
            homographies += 1
        if np.any(missing):
            residual = missing.astype(np.uint8) * 255
            result = cv2.inpaint(result, residual, inpaintRadius=7, flags=cv2.INPAINT_TELEA)
        writer.write(result)
    writer.release()

    report = {
        "method": "temporal_homography_then_telea",
        "frames": len(frame_paths),
        "masked_pixels": total_masked,
        "temporally_filled_pixels": temporal_filled,
        "temporal_fill_fraction": temporal_filled / total_masked if total_masked else 0.0,
        "successful_homographies": homographies,
    }
    output_path.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
