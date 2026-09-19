"""Re-run only the SVOR chunks whose frame ranges intersect mask changes.

Chunk outputs are cached under <output-dir>/svor/chunk_NNN/out/. Chunks whose
range does not intersect any changed range keep their cached output, so a
mask repair in two sections costs ~N_affected chunks instead of the full 44.
Stitching and the finalize tail (flow EMA -> source composite -> encode) are
the adapter's own code paths, so output is bit-compatible with a full run.

Usage (vbr environment):
    python -m tools.rerun_svor_partial --output-dir outputs/001_sam31_slam \
        --config configs/vggt_slam.yaml --ranges-json '[[622,1038],[1432,1799]]'
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vbr.cli import PROJECT_ROOT
from vbr.config import load_config
from vbr.models.svor import SVORAdapter, blend_stitch, dilate_masks, svor_chunk_ranges
from vbr.video import video_info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--config", default="configs/vggt_slam.yaml")
    parser.add_argument("--ranges-json", required=True,
                        help="frame ranges [start, end] whose chunks must re-run")
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = args.output_dir.resolve()
    input_video = (PROJECT_ROOT / cfg["input_video"]).resolve()
    info = video_info(input_video)
    changed = [(int(a), int(b)) for a, b in json.loads(args.ranges_json)]

    adapter = SVORAdapter(dict(cfg.get("video_completion", {})), PROJECT_ROOT)
    work_root = output_dir / "svor"
    work_root.mkdir(parents=True, exist_ok=True)
    fps_value = info["fps"]
    frames, masks, size = adapter._synthesize_inputs(
        output_dir / "frames_all", output_dir / "masks_inpaint"
    )
    total = len(frames)
    masks = dilate_masks(masks, int(adapter.cfg.get("mask_dilation", 0)))
    pad = (4 - (total - 1) % 4) % 4
    if pad:
        frames += [frames[-1]] * pad
        masks += [masks[-1]] * pad
    ranges = list(
        svor_chunk_ranges(
            len(frames),
            int(adapter.cfg.get("chunk_frames", 81)),
            int(adapter.cfg.get("overlap", 41)),
        )
    )
    chunk_inputs = adapter._chunk_inputs(work_root, frames, masks, fps_value, ranges)

    def is_affected(start: int, end: int) -> bool:
        return any(start < b and end > a for a, b in changed)

    logs_dir = output_dir / "logs"
    chunk_outputs = []
    rerun_indices = []
    for index, (start, end, chunk_input, chunk_mask, chunk_dir) in enumerate(chunk_inputs):
        produced = chunk_dir / "out" / chunk_input.name
        if is_affected(start, end) or not produced.exists() or produced.stat().st_size == 0:
            produced = adapter._predict_chunk(
                index, chunk_input, chunk_mask, chunk_dir, info["fps"], start, end, logs_dir
            )
            rerun_indices.append(index)
        chunk_outputs.append((start, end, produced))
    print(json.dumps({"chunks_total": len(ranges), "rerun": rerun_indices}))

    raw_output = work_root / "combined.mp4"
    stitched = blend_stitch(
        chunk_outputs, raw_output, fps_value,
        fade_frames=int(adapter.cfg.get("fade_frames", 12)),
    )
    if stitched != len(frames):
        raise RuntimeError(f"Stitched {stitched} frames, expected {len(frames)}")

    report = adapter._finalize(
        raw_output, output_dir / "background_video.mp4", input_video, frames, masks,
        total, fps_value, size, len(ranges), output_dir / "masks_inpaint",
    )
    report["rerun_chunks"] = rerun_indices
    (output_dir / "background_video.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
