"""Carve foreground (furniture) out of a reconstructed mesh using poses + masks.

Why this exists: the VGGT-SLAM baseline (`tools/rebuild_vggt_baseline.py`) runs
with EMPTY masks, so the furniture the original camera saw is fused into the
mesh. Route P/A/B avoid that by never integrating those pixels, but any mesh
built from unmasked depth keeps the furniture. This tool removes it afterwards,
on the mesh, using the SLAM poses and the per-frame SAM masks.

Method. Every vertex is projected into every exported keyframe with the exact
convention `vbr.geometry.texture_mesh_from_frames` uses (letterbox box in
`original_coords`). A sample is accepted only when the projected pixel carries
this vertex's OWN surface -- stored depth within a tight ratio of the vertex
depth (``--min-depth-ratio``/``--max-depth-ratio``). Crucially, a vertex is
carved only when it was NEVER ONCE seen as background:

    background_views == 0  AND  foreground_views >= --min-foreground-views

Why that and not a fraction threshold. A mask fraction ("flag vertices that are
foreground in >=70% of views") looks fine on the baseline -- it flags ~25% --
but on a control mesh that was built with NO furniture at all it still flags
9.8% of vertices. Those are real wall and floor pixels behind the furniture:
the furniture stands in front, so the wall inherits the furniture's mask. A
fraction threshold cannot tell "this surface IS furniture" from "furniture was
in front of this surface most of the time".

The disambiguator is that a real surface becomes visible eventually: across 71
keyframes of a walkthrough, a wall behind a sofa is seen directly as background
in at least one view, while a sofa body is foreground in every view that sees
it. Measured against the route-A control (geometry built from masked frames, no
furniture): ``bg==0 & fg>=3`` flags 16.34% of the baseline and only 0.40% of the
control -- 41x separation, versus ~10x for the best fraction threshold. The
control number is the honest false-positive rate of the rule.

Coordinate note (a real trap): depth/confidence are MODEL-space (518x294 here)
while the masks are ORIGINAL-space (960x540). Depth is therefore indexed with
the model-space projection, and the mask with ``original_coords``' letterbox
box. Resizing one to the other silently misattributes votes.

Sampling is restricted to keyframes that carry a pose; cameras that never see a
vertex simply contribute nothing to it.

Usage (vbr environment):

    # report only -- no mesh written, prints the vote histogram
    python -m tools.subtract_foreground \
        --run-dir outputs/vggt_slam_baseline \
        --mesh outputs/vggt_slam_baseline/background_mesh_denoised_reg.ply \
        --dry-run

    # carve
    python -m tools.subtract_foreground \
        --run-dir outputs/vggt_slam_baseline \
        --mesh outputs/vggt_slam_baseline/background_mesh_denoised_reg.ply \
        --out outputs/vggt_slam_baseline/background_mesh_nofurniture.ply

Validate any new threshold against a furniture-free control mesh (route A)
before trusting it -- that is what exposed the fraction threshold.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from vbr.cli import PROJECT_ROOT
from vbr.config import load_config


def _as_vector3i(triangles):
    import open3d as o3d

    return o3d.utility.Vector3iVector(np.ascontiguousarray(triangles, dtype=np.int32))


def load_masks(mask_dir, frame_ids, shape, cache, resample=None):
    """Load the 2D foreground masks for the given frame ids (bool, HxW).

    Masks live per FULL-VIDEO frame id (``000000.png``), in ORIGINAL image
    resolution -- *not* the model-space depth resolution. A mask whose stored
    size differs from ``shape`` is resized to it (``resample`` given) or treated
    as unusable, never silently indexed out of bounds.
    """
    import cv2

    height, width = shape
    bad = []
    for frame_id in frame_ids:
        if frame_id in cache:
            continue
        path = Path(mask_dir) / f"{frame_id:06d}.png"
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) if path.exists() else None
        if image is None:
            bad.append({"frame_id": int(frame_id), "reason": "missing"})
            cache[frame_id] = np.zeros((height, width), dtype=bool)
            continue
        if image.shape != (height, width):
            if resample is None:
                bad.append({"frame_id": int(frame_id),
                            "reason": f"shape {image.shape} != {(height, width)}"})
                cache[frame_id] = np.zeros((height, width), dtype=bool)
                continue
            image = cv2.resize(image, (width, height), interpolation=resample)
        cache[frame_id] = image > 127
    return cache, bad


def accumulate_votes(
    vertices,
    data,
    mask_dir,
    min_depth_ratio=0.90,
    max_depth_ratio=1.10,
):
    """Project every vertex into every posed keyframe and tally the votes.

    Returns per-vertex ``foreground``, ``background`` and ``observed`` counts.
    Only samples landing on the vertex's own surface are counted (see module
    docstring for why the band is tight).
    """
    import cv2

    extrinsics = data["extrinsics"]
    intrinsics = data["intrinsics"]
    coords = np.asarray(data["original_coords"], dtype=np.float64).reshape(-1, 6)
    depth = data["depth"][..., 0]
    frame_ids = [int(value) for value in data["frame_ids"]]
    model_height, model_width = int(depth.shape[1]), int(depth.shape[2])
    # Masks are ORIGINAL-space: the letterbox box gives their size per frame.
    original_height = int(round(coords[0, 5]))
    original_width = int(round(coords[0, 4]))

    vertices = np.asarray(vertices, dtype=np.float64)
    count = len(vertices)
    foreground = np.zeros(count, dtype=np.int64)
    background = np.zeros(count, dtype=np.int64)
    observed = np.zeros(count, dtype=np.int64)

    cache, bad_masks = load_masks(
        mask_dir, frame_ids, (original_height, original_width), {}, resample=cv2.INTER_NEAREST
    )

    for index in range(len(frame_ids)):
        frame_id = frame_ids[index]
        extrinsic = np.asarray(extrinsics[index], dtype=np.float64)
        intrinsic = np.asarray(intrinsics[index], dtype=np.float64)
        box = coords[index]
        point_camera = vertices @ extrinsic[:3, :3].T + extrinsic[:3, 3]
        depth_z = point_camera[:, 2]
        projected = point_camera @ intrinsic.T
        safe_z = np.where(np.abs(projected[:, 2]) > 1e-6, projected[:, 2], 1.0)
        u_model = projected[:, 0] / safe_z
        v_model = projected[:, 1] / safe_z
        width_original = max(box[2] - box[0], 1e-6)
        height_original = max(box[3] - box[1], 1e-6)
        ox = (u_model - box[0]) / width_original * box[4]
        oy = (v_model - box[1]) / height_original * box[5]
        in_bounds = (
            (depth_z > 1e-6)
            & (ox >= 0) & (ox < box[4])
            & (oy >= 0) & (oy < box[5])
        )
        if not in_bounds.any():
            continue
        # depth/confidence are MODEL-space -> index with the model projection.
        mx = np.clip(u_model.astype(int), 0, model_width - 1)
        my = np.clip(v_model.astype(int), 0, model_height - 1)
        depth_at = depth[index][my, mx]
        ratio = depth_at / np.where(depth_z > 1e-6, depth_z, 1.0)
        on_own_surface = (
            in_bounds
            & (depth_at > 0)
            & (ratio >= min_depth_ratio)
            & (ratio <= max_depth_ratio)
        )
        if not on_own_surface.any():
            continue
        # masks are ORIGINAL-space -> index with the letterbox projection.
        ox_idx = np.clip(ox.astype(int), 0, original_width - 1)
        oy_idx = np.clip(oy.astype(int), 0, original_height - 1)
        mask = cache[frame_id]
        is_foreground = mask[oy_idx, ox_idx]
        observed += on_own_surface
        foreground += on_own_surface & is_foreground
        background += on_own_surface & ~is_foreground

    stats = {
        "keyframes_used": len(frame_ids),
        "mask_shape": [original_height, original_width],
        "missing_or_bad_masks": bad_masks[:32],
        "bad_mask_count": len(bad_masks),
    }
    return foreground, background, observed, stats


def select_foreground(foreground, background, observed, min_foreground_views=3,
                      max_background_views=0):
    """Vertices that are furniture: never background, foreground often enough."""
    return (
        (observed > 0)
        & (background <= int(max_background_views))
        & (foreground >= int(min_foreground_views))
    )


def carve_mesh(mesh, flagged, min_component_triangles=0):
    """Drop triangles whose every corner is flagged; optionally prune leftovers.

    Requiring ALL THREE corners keeps the boundary ring of a furniture body
    (which touches the floor) in the mesh, so the carve leaves a clean rim
    rather than a ragged one-triangle-wide fringe.
    """
    import open3d as o3d

    triangles = np.asarray(mesh.triangles)
    drop = flagged[triangles].all(axis=1)
    subset = o3d.geometry.TriangleMesh()
    subset.vertices = mesh.vertices
    subset.triangles = _as_vector3i(triangles[~drop])
    if mesh.has_vertex_colors():
        subset.vertex_colors = mesh.vertex_colors
    if mesh.has_vertex_normals():
        subset.vertex_normals = mesh.vertex_normals
    subset.remove_unreferenced_vertices()
    subset.compute_vertex_normals()

    pruned = 0
    if min_component_triangles > 0:
        labels, _, _ = subset.cluster_connected_triangles()
        labels = np.asarray(labels)
        sizes = np.bincount(labels)
        keep_components = np.flatnonzero(sizes >= min_component_triangles)
        keep_faces = np.isin(labels, keep_components)
        pruned = int((~keep_faces).sum())
        kept = np.asarray(subset.triangles)[keep_faces]
        subset2 = o3d.geometry.TriangleMesh()
        subset2.vertices = subset.vertices
        subset2.triangles = _as_vector3i(kept)
        if subset.has_vertex_colors():
            subset2.vertex_colors = subset.vertex_colors
        subset2.remove_unreferenced_vertices()
        subset2.compute_vertex_normals()
        subset = subset2

    return subset, {
        "triangles_before": int(len(triangles)),
        "triangles_carved": int(drop.sum()),
        "triangles_after": int(len(subset.triangles)),
        "components_pruned_triangles": pruned,
        "vertices_before": int(len(np.asarray(mesh.vertices))),
        "vertices_after": int(len(subset.vertices)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path,
                        default=PROJECT_ROOT / "outputs/vggt_slam_baseline",
                        help="dir holding slam/points_background.npz")
    parser.add_argument("--npz", type=Path, default=None)
    parser.add_argument("--mesh", type=Path, required=True,
                        help="mesh to carve (furniture still present)")
    parser.add_argument("--out", type=Path, default=None,
                        help="output PLY; omit with --dry-run")
    parser.add_argument("--masks", type=Path, default=None,
                        help="per-frame foreground masks "
                             "(default: <config output_dir>/masks)")
    parser.add_argument("--config", default="configs/vggt_slam.yaml")
    parser.add_argument("--min-depth-ratio", type=float, default=0.90)
    parser.add_argument("--max-depth-ratio", type=float, default=1.10)
    parser.add_argument("--min-foreground-views", type=int, default=3)
    parser.add_argument("--max-background-views", type=int, default=0,
                        help="a vertex seen as background more often than this "
                             "is structure and is never carved; 0 = the "
                             "validated rule")
    parser.add_argument("--min-component-triangles", type=int, default=0,
                        help="after carving, drop connected components smaller "
                             "than this (0 disables)")
    parser.add_argument("--dry-run", action="store_true",
                        help="report the vote histogram without writing a mesh")
    args = parser.parse_args()

    run_dir = (PROJECT_ROOT / args.run_dir).resolve() if not args.run_dir.is_absolute() else args.run_dir
    npz_path = args.npz or (run_dir / "slam" / "points_background.npz")
    if not Path(npz_path).exists():
        raise FileNotFoundError(npz_path)
    if args.masks is not None:
        mask_dir = (PROJECT_ROOT / args.masks).resolve() if not args.masks.is_absolute() else args.masks
    else:
        cfg = load_config(args.config)
        mask_dir = (PROJECT_ROOT / cfg["output_dir"]).resolve() / "masks"
    if not Path(mask_dir).exists():
        raise FileNotFoundError(f"mask dir not found: {mask_dir}")

    mesh_path = (PROJECT_ROOT / args.mesh).resolve() if not args.mesh.is_absolute() else args.mesh

    import open3d as o3d

    data = np.load(npz_path)
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    vertices = np.asarray(mesh.vertices)
    print(f"mesh={mesh_path} vertices={len(vertices)} "
          f"triangles={len(mesh.triangles)}")
    print(f"npz={npz_path} keyframes={len(data['frame_ids'])} masks={mask_dir}")

    foreground, background, observed, stats = accumulate_votes(
        vertices, data, mask_dir,
        min_depth_ratio=args.min_depth_ratio,
        max_depth_ratio=args.max_depth_ratio,
    )
    if stats["bad_mask_count"]:
        raise RuntimeError(
            f"{stats['bad_mask_count']} unusable masks (first: "
            f"{stats['missing_or_bad_masks'][:3]}); refusing to carve with "
            "partial masks"
        )
    flagged = select_foreground(
        foreground, background, observed,
        min_foreground_views=args.min_foreground_views,
        max_background_views=args.max_background_views,
    )
    covered = observed > 0
    ratio = np.where(covered, foreground / np.maximum(observed, 1), 0.0)
    histogram = {
        f">={threshold:.1f}": int((covered & (ratio >= threshold)).sum())
        for threshold in (0.3, 0.5, 0.7, 0.9, 1.0)
    }
    print(f"observed vertices: {int(covered.sum())}/{len(vertices)}")
    print("foreground fraction histogram:", json.dumps(histogram))
    print(f"flagged (bg<={args.max_background_views} & fg>={args.min_foreground_views}): "
          f"{int(flagged.sum())} ({100 * flagged.sum() / max(len(vertices), 1):.2f}%)")

    report = {
        "mesh": str(mesh_path),
        "reconstruction": str(npz_path),
        "masks": str(mask_dir),
        "rule": {
            "min_depth_ratio": args.min_depth_ratio,
            "max_depth_ratio": args.max_depth_ratio,
            "min_foreground_views": args.min_foreground_views,
            "max_background_views": args.max_background_views,
        },
        "votes": {
            "vertices": int(len(vertices)),
            "observed_vertices": int(covered.sum()),
            "flagged_vertices": int(flagged.sum()),
            "flagged_fraction": float(flagged.sum()) / max(len(vertices), 1),
            "foreground_fraction_histogram": histogram,
            "never_background_and_foreground": int(
                (covered & (background == 0) & (foreground >= 1)).sum()
            ),
        },
        "sampling": stats,
    }

    if args.dry_run:
        report["dry_run"] = True
    else:
        out_path = args.out or (mesh_path.with_name(mesh_path.stem + "_nofurniture.ply"))
        out_path = (PROJECT_ROOT / out_path).resolve() if not out_path.is_absolute() else out_path
        subset, carve_stats = carve_mesh(
            mesh, flagged, min_component_triangles=args.min_component_triangles
        )
        o3d.io.write_triangle_mesh(str(out_path), subset)
        report["carve"] = carve_stats
        report["output"] = str(out_path)
        print("carve:", json.dumps(carve_stats))
        print(f"wrote {out_path}")

    report_path = (args.out if args.out else mesh_path).with_suffix(".foreground_report.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()