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


def mask_statistics(masks_dir, expected_frames):
    paths = sorted(Path(masks_dir).glob("*.png"), key=lambda path: int(path.stem))
    coverages = []
    for path in paths:
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            coverages.append(float(np.count_nonzero(mask)) / mask.size)
    return {
        "files": len(paths),
        "expected_frames": expected_frames,
        "nonempty_frames": int(np.count_nonzero(np.asarray(coverages) > 0)),
        "mean_coverage": float(np.mean(coverages)) if coverages else 0.0,
        "max_coverage": float(np.max(coverages)) if coverages else 0.0,
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
