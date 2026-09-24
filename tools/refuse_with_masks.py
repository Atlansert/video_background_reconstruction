"""Re-fuse a reconstruction with the REAL foreground masks.

Why this exists (and why carving was abandoned). `outputs/vggt_slam_baseline/`
was deliberately built with EMPTY masks, so furniture is fused into its geometry.
`tools/subtract_foreground.py` tried to repair that *afterwards*, by deleting
triangles from the finished mesh. Measured with leave-one-frame-out against the
2D masks, that approach leaves **55.8% of furniture triangles in place** and
turns the survivors into 853 disconnected fragments. Three visible symptoms,
one root cause:

  1. **only the middle removed** -- a vertex was carved only if it was NEVER once
     seen as background. Along an object's silhouette, where it meets the floor,
     the mask edge is genuinely ambiguous, so a collar of the object always
     survived and the object came out hollowed rather than removed.
  2. **shredded** -- the decision was per-vertex over noisy depth votes, so what
     did survive was fragments, not objects.
  3. **not removed at all** -- the tight "is this pixel this vertex's own
     surface" band discarded many genuinely-visible samples, and a single
     background sighting disqualified a vertex forever.

The root cause is that carving is a decision about *already-fused* geometry: a
tabletop and the floor it stands on are the same continuous fused surface, and
no post-hoc rule separates them reliably.

So: do not fuse the furniture in the first place. Masked depth is zeroed BEFORE
integration; Open3D skips depth 0, so those pixels contribute no surface and the
object is absent by construction -- no collar, no shredding, no survivors. This
reuses the baseline's OWN poses/depth/confidence and changes only the masking,
so the result stays comparable to the baseline 1:1.

Honest limitation: where furniture stood, VGGT never observed the surface behind
it, so zeroing leaves a hole. Holes are filled by structural priors in route A
(`tools/rebuild_geometry_prior.py`); this tool deliberately does not invent
geometry, it only removes.

Usage (vbr environment):
    python -m tools.refuse_with_masks \
        --npz outputs/vggt_slam_baseline/slam/points_background.npz \
        --out outputs/vggt_slam_baseline/background_mesh_masked.ply
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


def load_masks_model_space(mask_dir, frame_ids, coords, model_height, model_width):
    """Per-keyframe bool masks in MODEL space, via the canonical mapping.

    Reuses `vbr.vggt_slam_backend.mask_in_model_space` so the mask lands on
    exactly the pixels the depth grid uses. A missing mask raises: silently
    treating it as empty would turn a masked run into an unmasked one.
    """
    masks = np.zeros((len(frame_ids), model_height, model_width), dtype=bool)
    missing = []
    for index, frame_id in enumerate(frame_ids):
        path = Path(mask_dir) / f"{frame_id:06d}.png"
        if not path.exists():
            missing.append(int(frame_id))
            continue
        masks[index] = mask_in_model_space(
            path, coords[index], model_height, model_width
        ) > 0
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} masks missing for keyframes {missing[:8]}"
        )
    return masks


def fuse(depth, foreground, intrinsics, extrinsics, coords, frame_paths, geom,
         color_paths=None):
    """TSDF-fuse the depth volume with masked samples removed."""
    import open3d as o3d

    height = int(depth.shape[1])
    width = int(depth.shape[2])
    positive = depth[..., 0][depth[..., 0] > 0]
    depth_trunc = float(
        np.quantile(positive, geom.get("depth_trunc_quantile", 0.995))
    )
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=geom.get("voxel_size", 0.02),
        sdf_trunc=geom.get("tsdf_trunc", 0.10),
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )
    removed = 0
    for index, frame_path in enumerate(frame_paths):
        depth_image = depth[index, ..., 0].copy()
        drop = foreground[index]
        depth_image[drop] = 0.0            # the whole point: skip, do not fuse
        removed += int(drop.sum())
        color_source = frame_paths[index] if color_paths is None else color_paths[index]
        color_image = _model_rgb(str(color_source), coords[index], width, height)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(color_image),
            o3d.geometry.Image(depth_image.astype(np.float32)),
            depth_scale=1.0,
            depth_trunc=depth_trunc,
            convert_rgb_to_intensity=False,
        )
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width, height,
            float(intrinsics[index, 0, 0]), float(intrinsics[index, 1, 1]),
            float(intrinsics[index, 0, 2]), float(intrinsics[index, 1, 2]),
        )
        extrinsic = np.eye(4)
        extrinsic[:3] = extrinsics[index]
        volume.integrate(rgbd, intrinsic, extrinsic)
    mesh = volume.extract_triangle_mesh()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_non_manifold_edges()
    mesh.compute_vertex_normals()
    return mesh, {"masked_pixels_removed": removed, "depth_trunc": depth_trunc}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=Path,
                        default=PROJECT_ROOT / "outputs/vggt_slam_baseline/slam/points_background.npz")
    parser.add_argument("--masks", type=Path, default=None,
                        help="per-frame foreground masks (default: <config output_dir>/masks)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", default="configs/vggt_slam.yaml")
    parser.add_argument("--min-component-triangles", type=int, default=0)
    parser.add_argument("--mask-dilate", type=int, default=4,
                        help="grow the mask by N px before zeroing depth. The "
                             "mask edge is imprecise, so pixels just outside it "
                             "are often furniture too; without this a collar "
                             "survives at every silhouette and the object looks "
                             "'only half removed'. Measured (ray-cast, held-out "
                             "keyframes, furniture still present): dilate 0 -> "
                             "21.9%%, 2 -> 16.8%%, 4 -> 12.2%%.")
    parser.add_argument("--masks-default", default="masks_inpaint",
                        help="mask subdir to use when --masks is omitted")
    args = parser.parse_args()

    cfg = load_config(args.config)
    geom = cfg.get("geometry", {})
    npz = (PROJECT_ROOT / args.npz).resolve() if not args.npz.is_absolute() else args.npz
    masks = args.masks or (
        (PROJECT_ROOT / cfg["output_dir"]).resolve() / args.masks_default
    )
    out = (PROJECT_ROOT / args.out).resolve() if not args.out.is_absolute() else args.out

    import open3d as o3d

    data = np.load(npz)
    depth = data["depth"]
    intrinsics = data["intrinsics"]
    extrinsics = data["extrinsics"]
    coords = np.asarray(data["original_coords"], dtype=np.float64).reshape(-1, 6)
    frame_paths = [Path(str(p)) for p in data["frame_paths"]]
    frame_ids = [int(v) for v in data["frame_ids"]]
    model_height, model_width = int(depth.shape[1]), int(depth.shape[2])

    print(f"npz={npz}\n  keyframes={len(frame_ids)} model={model_height}x{model_width} "
          f"masks={masks}")
    foreground = load_masks_model_space(
        masks, frame_ids, coords, model_height, model_width
    )
    if args.mask_dilate > 0:
        import cv2

        kernel = np.ones((3, 3), np.uint8)
        foreground = np.stack([
            cv2.dilate(layer.astype(np.uint8), kernel,
                       iterations=args.mask_dilate) > 0
            for layer in foreground
        ])
        print(f"  mask dilated by {args.mask_dilate} px -> coverage "
              f"{foreground.mean():.4f}")
    per_frame = foreground.reshape(len(frame_ids), -1).sum(1)
    print(f"  mask coverage: mean={foreground.mean():.4f} "
          f"min={per_frame.min()} max={per_frame.max()} px "
          f"frames_with_fg={int((per_frame > 0).sum())}/{len(frame_ids)}")
    if not foreground.any():
        raise RuntimeError("all masks are empty; refusing to write an unmasked build")

    mesh, stats = fuse(depth, foreground, intrinsics, extrinsics, coords,
                       frame_paths, geom)
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
            np.ascontiguousarray(np.asarray(mesh.triangles)[face_keep], dtype=np.int32)
        )
        if mesh.has_vertex_colors():
            trimmed.vertex_colors = mesh.vertex_colors
        trimmed.remove_unreferenced_vertices()
        trimmed.compute_vertex_normals()
        mesh = trimmed
        print(f"component filter: dropped {pruned} triangles "
              f"({len(sizes)} -> {len(keep)} components)")

    o3d.io.write_triangle_mesh(str(out), mesh)
    report = {
        "npz": str(npz),
        "masks": str(masks),
        "output": str(out),
        "keyframes": len(frame_ids),
        "model_space": [model_height, model_width],
        "mask_coverage_mean": float(foreground.mean()),
        "mask_dilate_px": args.mask_dilate,
        "stats": stats,
        "triangles": int(len(mesh.triangles)),
        "vertices": int(len(mesh.vertices)),
        "components_pruned_triangles": pruned,
    }
    out.with_suffix(".refuse_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out} ({len(mesh.triangles)} triangles, {len(mesh.vertices)} vertices)")


if __name__ == "__main__":
    main()