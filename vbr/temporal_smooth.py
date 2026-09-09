"""Flow-aligned temporal median smoothing for the inpainted background video.

A plain 3-frame pixel median smears texture wherever the camera moves: the
neighbors are sampled out of alignment. RAFT (from the ProPainter repo
weights) aligns t-1 and t+1 into t first, so the median only blends truly
corresponding pixels.

The warp/median core is pure NumPy and unit-testable with synthetic flows;
RAFT runs only inside the vbr-seg environment (torch):

    python -m vbr.temporal_smooth --input inpaint_out.mp4 --masks masks_inpaint \
        --output smooth_out.mp4 --fps 30 --repo-dir external/ProPainter \
        --raft-checkpoint weights/raft-things.pth
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


def warp_flow(frame, flow):
    """Backward-warp ``frame`` so its content lines up with the flow grid.

    ``flow(q)`` is the displacement from pixel q to the position in ``frame``
    that shows the same scene content (RAFT's negative of the forward motion
    for the previous frame, forward motion for the next frame).
    """
    height, width = frame.shape[:2]
    flow = np.asarray(flow, dtype=np.float32)
    map_x, map_y = np.meshgrid(
        np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32)
    )
    return cv2.remap(
        frame,
        map_x + flow[..., 0],
        map_y + flow[..., 1],
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def aligned_median(target, neighbor_frames, neighbor_flows, mask):
    """Median of the target and its flow-aligned neighbors, inside ``mask``.

    ``neighbor_frames[i]`` is aligned into the target grid using
    ``neighbor_flows[i]``; entries that are None are skipped. Returns the
    smoothed copy of ``target`` and the number of aligned frames used.
    """
    aligned = [target]
    for frame, flow in zip(neighbor_frames, neighbor_flows):
        if frame is None or flow is None:
            continue
        aligned.append(warp_flow(frame, flow))
    stack = np.stack(aligned, axis=0)
    median = np.median(stack, axis=0)
    result = target.copy()
    if mask is not None and np.any(mask):
        result[mask] = median[mask].astype(np.uint8)
    return result, len(aligned)


def _to_raft_input(frame_bgr, height, width):
    """BGR uint8 -> [1,3,H,W] float tensor in [-1,1], padded to 16 multiples."""
    import torch

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
    pad_height = int(np.ceil(height / 16) * 16)
    pad_width = int(np.ceil(width / 16) * 16)
    if pad_height != height or pad_width != width:
        padding = (0, pad_width - width, 0, pad_height - height)
        tensor = torch.nn.functional.pad(tensor, padding, mode="replicate")
    return tensor * 2.0 - 1.0


def initialize_raft(repo_dir, checkpoint, device="cuda"):
    """Load the RAFT model exactly as ProPainter's compute_flow script does."""
    import torch

    repo = Path(repo_dir).resolve()
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from RAFT import RAFT

    args = argparse.Namespace(
        raft_model=str(checkpoint),
        small=False,
        mixed_precision=False,
        alternate_corr=False,
    )
    model = torch.nn.DataParallel(RAFT(args))
    state = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model = model.module.to(device).eval()
    return model


def estimate_forward_flow(raft, frame_source, frame_target, device, height, width):
    """RAFT flow from ``frame_source`` to ``frame_target``, on the grid of the source."""
    import torch

    img1 = _to_raft_input(frame_source, height, width).to(device)
    img2 = _to_raft_input(frame_target, height, width).to(device)
    with torch.no_grad():
        _, flow = raft(img1, img2, iters=20, test_mode=True)
    flow = flow[0].permute(1, 2, 0).cpu().numpy().astype(np.float32)
    if flow.shape[0] != height or flow.shape[1] != width:
        flow = flow[:height, :width]
    return flow


def smooth_video(frames, masks, output_path, fps, repo_dir, raft_checkpoint, device="cuda"):
    """Apply flow-aligned 3-frame median smoothing inside the mask regions.

    ``frames`` are BGR arrays (inpainted video), ``masks`` binary arrays.
    Writes an mp4v intermediate (the caller re-encodes with audio).
    """
    output_path = Path(output_path)
    height, width = frames[0].shape[:2]
    raft = initialize_raft(repo_dir, raft_checkpoint, device=device)

    flows_prev = [None] * len(frames)  # t -> t-1
    flows_next = [None] * len(frames)  # t -> t+1
    for index in range(1, len(frames)):
        flows_prev[index] = estimate_forward_flow(
            raft, frames[index], frames[index - 1], device, height, width
        )
        flows_next[index - 1] = estimate_forward_flow(
            raft, frames[index - 1], frames[index], device, height, width
        )

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )
    total_changed = 0
    for index, frame in enumerate(frames):
        neighbors = [None, None]
        flows = [None, None]
        if index > 0:
            neighbors[0] = frames[index - 1]
            flows[0] = flows_prev[index]
        if index < len(frames) - 1:
            neighbors[1] = frames[index + 1]
            flows[1] = flows_next[index]
        mask = masks[index] if index < len(masks) else None
        smoothed, used = aligned_median(frame, neighbors, flows, mask)
        if mask is not None and np.any(mask):
            total_changed += int(np.count_nonzero(mask))
        writer.write(smoothed)
    writer.release()
    return {
        "kernel": 3,
        "frames": len(frames),
        "flow_aligned_frames_used": "2" if len(frames) > 2 else "1",
        "changed_pixels": total_changed,
        "output": str(output_path.resolve()),
    }


def _read_video_frames(video_path):
    capture = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if not frames:
        raise RuntimeError(f"No frames read from {video_path}")
    return frames


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--masks", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--repo-dir", required=True)
    parser.add_argument("--raft-checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--log", default=None)
    args = parser.parse_args()

    frames = _read_video_frames(args.input)
    height, width = frames[0].shape[:2]
    masks = []
    for index in range(len(frames)):
        mask_path = Path(args.masks) / f"{index:06d}.png"
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            masks.append(np.zeros((height, width), dtype=bool))
            continue
        if mask.shape[:2] != (height, width):
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        masks.append(mask > 0)

    report = smooth_video(
        frames,
        masks,
        args.output,
        args.fps,
        args.repo_dir,
        args.raft_checkpoint,
        device=args.device,
    )
    report_path = Path(args.output).with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.log:
        Path(args.log).parent.mkdir(parents=True, exist_ok=True)
        Path(args.log).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"temporal_smooth: {report}")


if __name__ == "__main__":
    main()