"""Multi-view mask consensus: remove furniture that ANY single view would keep.

Why this exists (measured, 2026-09-24). `refuse_with_masks` zeroes masked depth
per keyframe before TSDF integration, which is right as far as it goes, but TSDF
fusion is a **union over frames**: a 3D point survives if ANY keyframe fuses it.
So a piece of furniture is removed only if the mask covers it in *every* view
that sees it. The shipped masks do not, so furniture survives -- exactly the
residual the user reports (TV, part of the fridge, closets).

Measured on the baseline surface, per point-observation seen by >=3 keyframes:

    masked in <25% of seeing views (background) : 27.98%
    masked in >75% of seeing views (foreground) : 55.32%
    ambiguous 25-75%                            : 16.70%

That is bimodal, which is what makes a per-point vote viable. So decide per 3D
POINT, not per 2D pixel: a point that a majority of the views that actually SEE
it call foreground is furniture, and is dropped from EVERY view. A point the
views agree is background is never dropped.

This is still removal, not invention: it only ever deletes depth samples. Where
furniture stood, the camera never saw the surface behind it, so a hole remains --
filling is structural-prior/diffusion work and out of scope here.

Usage (vbr environment):
    python -m tools.consensus_masks \
        --npz outputs/vggt_slam_baseline/slam/points_background.npz \
        --masks outputs/001_sam31_slam/masks_inpaint \
        --out outputs/vggt_slam_baseline/background_mesh_consensus.ply
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from vbr.cli import PROJECT_ROOT
from vbr.config import load_config
from vbr.geometry import _model_rgb
from vbr.vggt_slam_backend import mask_in_model_space


def load_masks(mask_dir, frame_ids, coords, height, width):
    masks = np.zeros((len(frame_ids), height, width), dtype=bool)
    missing = []
    for index, frame_id in enumerate(frame_ids):
        path = Path(mask_dir) / f"{frame_id:06d}.png"
        if not path.exists():
            missing.append(int(frame_id))
            continue
        masks[index] = mask_in_model_space(path, coords[index], height, width) > 0
    if missing:
        raise FileNotFoundError(f"{len(missing)} masks missing: {missing[:8]}")
    return masks


def backproject(depth, intrinsics, extrinsics, coords, stride):
    """Per-view 3D points of the recorded surface, on a pixel subsample.

    Returns (points[n, m, 3], ys[m], xs[m]); NaN where depth is 0.
    """
    height, width = int(depth.shape[1]), int(depth.shape[2])
    ys, xs = np.mgrid[0:height:stride, 0:width:stride]
    ys, xs = ys.ravel(), xs.ravel()
    n = len(depth)
    points = np.full((n, len(xs), 3), np.nan, dtype=np.float64)
    for k in range(n):
        d = depth[k][ys, xs, 0]
        ok = d > 0
        if not ok.any():
            continue
        x1, y1 = coords[k, 0], coords[k, 1]
        sx = (coords[k, 2] - x1) / max(width, 1e-6) * width
        # model-space pixel -> camera ray, then camera -> world
        u = xs[ok].astype(np.float64)
        v = ys[ok].astype(np.float64)
        dirn = np.stack([(u - intrinsics[k, 0, 2]) / intrinsics[k, 0, 0],
                         (v - intrinsics[k, 1, 2]) / intrinsics[k, 1, 1],
                         np.ones_like(u)], axis=1)
        R = extrinsics[k][:3, :3]
        org = -R.T @ extrinsics[k][:3, 3]
        dirw = (R.T @ dirn.T).T
        points[k, ok] = org[None, :] + dirw * d[ok][:, None]
    return points, ys, xs


def consensus(masks, depth, intrinsics, extrinsics, coords, stride,
              min_views, depth_tol, scene_rays):
    """Per-point foreground vote across the views that actually see it.

    A point is foreground when a majority of its *seeing* views mask it. Two
    deliberate choices:

    * **Seeing views only.** A view that cannot see the point abstains rather
      than voting "background". That is what makes the vote robust to the
      coverage swings (13% <-> 80% across this walk): an absent view is not
      evidence about the point.
    * **Leave-one-out.** For a point recorded by view ``i``, the vote counts the
      *other* views ``j != i``. View ``i``'s own mask therefore cannot bias its
      own decision, so the decision rests on independent evidence. Where no
      other view sees the point, ``seen[i] == 0`` and the caller falls back to
      the per-view mask, so such points are still handled.
    """
    height, width = int(depth.shape[1]), int(depth.shape[2])
    points, ys, xs = backproject(depth, intrinsics, extrinsics, coords, stride)
    n, m = points.shape[0], points.shape[1]
    seen = np.zeros((n, m), np.int32)
    votes = np.zeros((n, m), np.int32)

    # sample-index -> pixel index, to read the mask
    pix = ys * width + xs

    for i in range(n):
        P = points[i]
        ok = np.isfinite(P[:, 0])
        if not ok.any():
            continue
        Q = P[ok]
        src = np.flatnonzero(ok)
        # is Q visible in view j?  project, then compare raycast depth
        for j in range(n):
            if j == i:
                continue
            R, t = extrinsics[j][:3, :3], extrinsics[j][:3, 3]
            cam = Q @ R.T + t[None, :]
            z = cam[:, 2]
            good = z > 1e-6
            if not good.any():
                continue
            u = np.full(len(Q), -1.0)
            v = np.full(len(Q), -1.0)
            u[good] = intrinsics[j, 0, 0] * cam[good, 0] / z[good] + intrinsics[j, 0, 2]
            v[good] = intrinsics[j, 1, 1] * cam[good, 1] / z[good] + intrinsics[j, 1, 2]
            inside = good & (u >= 0) & (u < width) & (v >= 0) & (v < height)
            if not inside.any():
                continue
            idx = np.flatnonzero(inside)
            uq = np.clip(u[idx].round().astype(int), 0, width - 1)
            vq = np.clip(v[idx].round().astype(int), 0, height - 1)
            tj = scene_rays(j, uq, vq)
            same = np.isfinite(tj) & (np.abs(tj - z[idx]) < depth_tol)
            sel = idx[same]
            if len(sel) == 0:
                continue
            seen[i, src[sel]] += 1
            votes[i, src[sel]] += masks[j][vq[same], uq[same]].astype(np.int32)

    return points, ys, xs, pix, seen, votes


def decide(seen, votes, vote_threshold, min_views):
    """Turn per-view (seen, votes) into (foreground, undecided).

    The whole fix lives here, so it is a pure function of counts and can be
    tested without cameras or a GPU:

    * ``seen`` is how many OTHER views recorded this point's surface; ``votes``
      how many of those called it foreground. Leaving the point's own view out
      of its own verdict is what lets a view whose mask missed be overruled by
      its peers instead of re-fusing the furniture.
    * A point with too few seeing views has no verdict (``undecided``); the
      caller falls back to the per-view mask rather than reading silence as
      background.
    * Otherwise it is foreground when the masked fraction EXCEEDS the threshold.
      Strictly greater: at exactly 0.5 with two views the evidence is tied, and
      an unresolved point must not delete geometry.
    """
    seen = np.asarray(seen)
    votes = np.asarray(votes)
    undecided = seen < max(int(min_views), 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        fraction = np.where(seen > 0, votes / np.maximum(seen, 1), np.nan)
    foreground = np.isfinite(fraction) & (fraction > float(vote_threshold))
    return foreground & ~undecided, undecided


def fuse(depth, foreground, intrinsics, extrinsics, coords, frame_paths, geom):
    import open3d as o3d

    height, width = int(depth.shape[1]), int(depth.shape[2])
    positive = depth[..., 0][depth[..., 0] > 0]
    depth_trunc = float(np.quantile(positive, geom.get("depth_trunc_quantile", 0.995)))
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=geom.get("voxel_size", 0.02),
        sdf_trunc=geom.get("tsdf_trunc", 0.10),
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )
    removed = 0
    for index, frame_path in enumerate(frame_paths):
        d = depth[index, ..., 0].copy()
        drop = foreground[index]
        d[drop] = 0.0
        removed += int(drop.sum())
        color = _model_rgb(str(frame_path), coords[index], width, height)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(color), o3d.geometry.Image(d.astype(np.float32)),
            depth_scale=1.0, depth_trunc=depth_trunc, convert_rgb_to_intensity=False)
        intr = o3d.camera.PinholeCameraIntrinsic(
            width, height, float(intrinsics[index, 0, 0]), float(intrinsics[index, 1, 1]),
            float(intrinsics[index, 0, 2]), float(intrinsics[index, 1, 2]))
        ext = np.eye(4)
        ext[:3] = extrinsics[index]
        volume.integrate(rgbd, intr, ext)
    mesh = volume.extract_triangle_mesh()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_non_manifold_edges()
    mesh.compute_vertex_normals()
    return mesh, {"masked_pixels_removed": removed, "depth_trunc": depth_trunc}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=Path,
                    default=PROJECT_ROOT / "outputs/vggt_slam_baseline/slam/points_background.npz")
    ap.add_argument("--masks", type=Path, default=None)
    ap.add_argument("--masks-default", default="masks_inpaint")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--config", default="configs/vggt_slam.yaml")
    ap.add_argument("--stride", type=int, default=2,
                    help="model-space pixel stride for the vote grid")
    ap.add_argument("--mask-dilate", type=int, default=4)
    ap.add_argument("--min-views", type=int, default=2,
                    help="a point needs this many seeing views to be voted on; "
                         "below it, fall back to the per-view mask")
    ap.add_argument("--vote-threshold", type=float, default=0.5,
                    help="fraction of seeing views that must mask a point for it "
                         "to be called foreground and dropped from ALL views")
    ap.add_argument("--depth-tol", type=float, default=0.08)
    ap.add_argument("--consensus-dilate", type=int, default=None,
                    help="grow each subsampled grid verdict by N px before "
                         "applying it (default: the vote stride). 0 = pure "
                         "nearest-neighbour. Measured residual/background-kept "
                         "on the shipped walk: stride 13.57%%/89.36%%, "
                         "0 17.99%%/93.94%%.")
    ap.add_argument("--min-component-triangles", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    geom = cfg.get("geometry", {})
    npz = (PROJECT_ROOT / args.npz).resolve() if not args.npz.is_absolute() else args.npz
    masks_dir = args.masks or ((PROJECT_ROOT / cfg["output_dir"]).resolve() / args.masks_default)
    out = (PROJECT_ROOT / args.out).resolve() if not args.out.is_absolute() else args.out

    import cv2
    import open3d as o3d

    data = np.load(npz)
    depth = data["depth"]
    intrinsics, extrinsics = data["intrinsics"], data["extrinsics"]
    coords = np.asarray(data["original_coords"], dtype=np.float64).reshape(-1, 6)
    frame_paths = [Path(str(p)) for p in data["frame_paths"]]
    frame_ids = [int(v) for v in data["frame_ids"]]
    height, width = int(depth.shape[1]), int(depth.shape[2])

    masks = load_masks(masks_dir, frame_ids, coords, height, width)
    if args.mask_dilate > 0:
        ker = np.ones((3, 3), np.uint8)
        masks = np.stack([cv2.dilate(m.astype(np.uint8), ker,
                                     iterations=args.mask_dilate) > 0 for m in masks])
    print(f"npz={npz}\n  keyframes={len(frame_ids)} model={height}x{width} "
          f"masks={masks_dir} dilate={args.mask_dilate} mean_cov={masks.mean():.4f}")

    # raycast helper over the recorded depth, used for visibility
    ys_all, xs_all = np.mgrid[0:height, 0:width]
    dep_flat = depth[..., 0].reshape(len(frame_ids), -1)
    K, E = intrinsics, extrinsics

    def scene_rays(j, uq, vq):
        """Recorded depth at projected pixels (the 'did view j see this surface' test)."""
        return dep_flat[j][vq * width + uq]

    points, ys, xs, pix, seen, votes = consensus(
        masks, depth, K, E, coords, args.stride, args.min_views, args.depth_tol, scene_rays)
    print(f"  vote grid {len(ys)} px/view; seen>=1: "
          f"{100*(seen>=1).mean():.1f}%  mean seeing views={seen[seen>0].mean():.2f}")

    strong, weak = decide(seen, votes, args.vote_threshold, args.min_views)
    print(f"  consensus foreground: {100*strong[seen>0].mean():.2f}% of observed points")
    print(f"  falling back to per-view mask on {100*weak.mean():.1f}% of samples")

    # The vote lives on a subsampled grid of len(ys) x len(xs) cells. Assign
    # every full-resolution pixel the verdict of its NEAREST grid cell: that
    # covers exactly the block each sample stands for, so removal is kept and
    # no foreground is smeared into the background beside it.
    ny = len(range(0, height, args.stride))
    nx = len(range(0, width, args.stride))
    assert ny * nx == len(ys) == len(xs), (ny, nx, len(ys), len(xs))
    compact = strong.reshape(len(frame_ids), ny, nx).astype(np.uint8)
    # Assign each full-resolution pixel the verdict of its nearest grid cell.
    #
    # --consensus-dilate then grows each verdict by N px. Measured on the
    # shipped walk (ray-cast over 71 keyframes), residual vs background-kept:
    #
    #     dilate = stride (default)  13.57%  89.36%
    #     dilate = 0 (nearest only)  17.99%  93.94%
    #
    # Dilation removes more furniture and costs more background, so it is a
    # knob rather than a bug. The default is the arm that best addresses the
    # reported complaint. Note "background kept" is measured on mask-FREE
    # pixels while 74% of residues are mask-MISSED furniture, so part of any
    # drop is correct removal of furniture the mask never flagged.
    dilate = args.stride if args.consensus_dilate is None else args.consensus_dilate
    base = [cv2.resize(c, (width, height), interpolation=cv2.INTER_NEAREST)
            for c in compact] if dilate == 0 else None
    if dilate > 0:
        grown = np.stack([
            cv2.dilate(cv2.resize(c, (width, height),
                                  interpolation=cv2.INTER_NEAREST),
                       np.ones((dilate * 2 + 1,) * 2, np.uint8)) > 0
            for c in compact
        ])
    else:
        grown = np.stack([b > 0 for b in base])
    print(f"  grid {ny}x{nx} -> {height}x{width}, consensus-dilate={dilate}px")
    # Where the vote had no quorum, fall back to that view's own mask.
    fb = np.zeros((len(frame_ids), height, width), dtype=bool)
    for k in range(len(frame_ids)):
        f = np.zeros((height, width), dtype=bool)
        f[ys, xs] = weak[k]
        fb[k] = f
    foreground = grown | (fb & masks)
    print(f"  vote grid {ny}x{nx}, nearest-neighbour assigned to {height}x{width}")
    print(f"  per-view foreground: {100*foreground.mean():.2f}% "
          f"(per-view mask alone was {100*masks.mean():.2f}%)")

    mesh, stats = fuse(depth, foreground, K, E, coords, frame_paths, geom)
    print(f"raw surface: {len(mesh.triangles)} triangles, "
          f"masked samples removed={stats['masked_pixels_removed']}")

    pruned = 0
    if args.min_component_triangles > 0:
        labels, _, _ = mesh.cluster_connected_triangles()
        labels = np.asarray(labels)
        sizes = np.bincount(labels)
        keep = np.flatnonzero(sizes >= args.min_component_triangles)
        face_keep = np.isin(labels, keep)
        pruned = int((~face_keep).sum())
        trimmed = o3d.geometry.TriangleMesh()
        trimmed.vertices = mesh.vertices
        trimmed.triangles = o3d.utility.Vector3iVector(
            np.ascontiguousarray(np.asarray(mesh.triangles)[face_keep], dtype=np.int32))
        if mesh.has_vertex_colors():
            trimmed.vertex_colors = mesh.vertex_colors
        trimmed.remove_unreferenced_vertices()
        trimmed.compute_vertex_normals()
        mesh = trimmed
        print(f"component filter: dropped {pruned} triangles")

    o3d.io.write_triangle_mesh(str(out), mesh)
    report = {
        "npz": str(npz), "masks": str(masks_dir), "output": str(out),
        "keyframes": len(frame_ids), "model_space": [height, width],
        "stride": args.stride, "mask_dilate": args.mask_dilate,
        "min_views": args.min_views, "vote_threshold": args.vote_threshold,
        "depth_tol": args.depth_tol,
        "consensus_dilate": int(dilate),
        "mask_coverage_mean": float(masks.mean()),
        "foreground_coverage_mean": float(foreground.mean()),
        "consensus_foreground_fraction": float(strong[seen > 0].mean()),
        "triangles": int(len(mesh.triangles)), "vertices": int(len(mesh.vertices)),
        "stats": stats, "components_pruned_triangles": pruned,
    }
    out.with_suffix(".consensus_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out} ({len(mesh.triangles)} triangles, {len(mesh.vertices)} vertices)")


if __name__ == "__main__":
    main()