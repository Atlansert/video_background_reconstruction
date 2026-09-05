#!/usr/bin/env bash
# Fast end-to-end smoke test for the vggt_slam backend.
#
# Builds a 60-frame subset of the already-extracted frames, then runs the
# backend directly in the vbr-slam environment (small submaps, ~2 minutes on
# GPU 7). Verifies NPZ/PLY/trajectory outputs are produced and consistent.
#
# Usage: bash tools/smoke_test_slam.sh [frames_dir masks_dir]
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

FRAMES_SRC="${1:-outputs/001_sam31/frames_all}"
MASKS_SRC="${2:-outputs/001_sam31/masks}"
WORK="$(mktemp -d /tmp/vbr_slam_smoke.XXXXXX)"
mkdir -p "$WORK/frames" "$WORK/masks"

count=0
for src in "$FRAMES_SRC"/*.jpg; do
  id="$(basename "$src" .jpg)"
  index=$((10#$id))
  if (( index % 8 == 0 )) && (( index < 480 )); then
    cp "$src" "$WORK/frames/$id.jpg"
    cp "$MASKS_SRC/$id.png" "$WORK/masks/$id.png"
    count=$((count + 1))
  fi
done
echo "Smoke frames: $count"

CUDA_VISIBLE_DEVICES=7 PYTHONUNBUFFERED=1 conda run --no-capture-output -n vbr-slam \
  python -m vbr.vggt_slam_backend \
  --frames "$WORK/frames" \
  --masks "$WORK/masks" \
  --output "$WORK/points_background.ply" \
  --checkpoint checkpoints/vggt/model.pt \
  --frame-stride 1 --max-keyframes 60 --submap-size 8 --max-loops 1 \
  --model-mode square

conda run --no-capture-output -n vbr python - "$WORK" <<'PY'
import sys
from pathlib import Path

import numpy as np
import open3d as o3d

work = Path(sys.argv[1])
data = np.load(work / "points_background.npz")
required = {
    "extrinsics", "intrinsics", "depth", "confidence", "confidence_cutoff",
    "frame_paths", "frame_ids", "original_coords", "foreground_masks",
}
missing = required - set(data.files)
assert not missing, f"NPZ missing keys: {missing}"
assert np.isfinite(data["extrinsics"]).all(), "non-finite extrinsics"
assert len(data["frame_ids"]) == len(np.unique(data["frame_ids"])), "duplicate frame ids"
cloud = o3d.io.read_point_cloud(str(work / "points_background.ply"))
assert len(cloud.points) >= 500, f"too few points: {len(cloud.points)}"
assert (work / "trajectory_tum.txt").exists(), "trajectory missing"
print(f"Smoke test OK: {len(data['frame_ids'])} frames, {len(cloud.points)} points")
PY

rm -rf "$WORK"
