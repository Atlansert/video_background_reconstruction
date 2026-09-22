"""Flatten structural planes of a reconstructed mesh (wall/floor/ceiling polish).

Why: the raw VGGT-SLAM surface carries ~6 mm of local roughness on the walls.
Under a directional light that roughness turns into visible shading "potholes"
even though the texture underneath is smooth (unlit renders show clean walls).
Two effects stay separate: geometry roughness and texture noise. This tool
fixes the geometry side only.

Approach: fit the dominant planes (reusing ``vbr.geometry.fit_planes``) and
snap vertices *toward* the plane inside a soft band:

- only near-plane vertices move (|distance| < ``--band``; the fade ramp keeps
  furniture bodies and corner creases connected to the wall untouched);
- movement is capped (``--max-shift``) so a genuine protrusion that happens to
  be near the plane cannot be flattened away;
- vertices whose normals deviate from the plane normal are skipped
  (``--normal-tolerance``), preserving decorative elements mounted on walls.

Snapping is a blend: new = old + pull * (projection - old), pull ramping from
1 at the plane core to 0 at the band edge. No vertex is removed, so color and
triangle count stay identical.

Usage (vbr environment):
    python -m tools.regularize_planes \
        --mesh outputs/vggt_slam_baseline/background_mesh_denoised.ply \
        --npz outputs/vggt_slam_baseline/slam/points_background.npz \
        --out outputs/vggt_slam_baseline/background_mesh_denoised_flat.ply
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from vbr.cli import PROJECT_ROOT
from vbr.geometry import estimate_gravity, fit_planes


def regularize(mesh, gravity, cfg):
    import open3d as o3d

    vertices = np.asarray(mesh.vertices, dtype=np.float64).copy()
    normals = (
        np.asarray(mesh.vertex_normals, dtype=np.float64)
        if mesh.has_vertex_normals()
        else None
    )
    if normals is None or len(normals) != len(vertices):
        mesh.compute_vertex_normals()
        normals = np.asarray(mesh.vertex_normals, dtype=np.float64)

    band = float(cfg["band"])
    max_shift = float(cfg["max_shift"])
    normal_tolerance = float(np.cos(np.radians(cfg["normal_tolerance_deg"])))
    normal_pull_weight = float(cfg.get("normal_pull_weight", 0.0))
    planes = fit_planes(
        vertices,
        gravity,
        threshold=float(cfg["plane_threshold"]),
        min_points=int(cfg["min_plane_points"]),
        max_planes=int(cfg["max_planes"]),
    )

    out_normals = normals.copy()
    moved_total = 0
    normalized_total = 0
    plane_report = []
    for index, plane in enumerate(planes):
        normal = np.asarray(plane["normal"], dtype=np.float64)
        offset = float(plane["d"])
        distance = vertices @ normal + offset
        # Two overlapping selections:
        # - snap: vertices close enough that pulling them onto the plane is a
        #   small correction (|distance| < max_shift).
        # - regularize: the whole band. A vertex 3-6 cm off the fitted plane
        #   should not be repositioned, but its normal still belongs to this
        #   wall and blending it toward the plane normal is safe. Limiting the
        #   normal blend to the snap set (an earlier bug) left the very
        #   ripples that shade as potholes untouched.
        in_band = np.abs(distance) < band
        if normal_tolerance > 0:
            in_band &= np.abs(normals @ normal) >= normal_tolerance
        snap = in_band & (np.abs(distance) < max_shift)
        if not in_band.any():
            plane_report.append(
                {"kind": plane["kind"], "points": plane["points"], "moved": 0}
            )
            continue
        # Fade: 1 at the plane core, 0 at the band edge.
        pull = 1.0 - (np.abs(distance[in_band]) / band)

        if snap.any():
            snap_pull = pull[np.abs(distance[in_band]) < max_shift]
            shift = -distance[snap] * snap_pull
            shift = np.clip(shift, -max_shift, max_shift)
            before = float(np.sqrt((distance[snap] ** 2).mean()))
            vertices[snap] += shift[:, None] * normal[None]
            after = float(
                np.sqrt(((vertices[snap] @ normal + offset) ** 2).mean())
            )
            moved_total += int(snap.sum())
        else:
            before = after = 0.0

        # Shading-side regularization: blend band-wide normals toward the
        # plane normal with the same fade, so the wall shades as the flat
        # surface it is.
        if normal_pull_weight > 0:
            signs = np.sign(
                np.sum(out_normals[in_band] * normal[None], axis=1, keepdims=True)
            ).reshape(-1, 1)
            target = normal[None] * signs
            weight = (pull * normal_pull_weight).reshape(-1, 1)
            blended = out_normals[in_band] * (1.0 - weight) + target * weight
            lengths = np.linalg.norm(blended, axis=1, keepdims=True)
            out_normals[in_band] = blended / np.maximum(lengths, 1e-12)
            normalized_total += int(in_band.sum())

        plane_report.append(
            {
                "kind": plane["kind"],
                "points": int(plane["points"]),
                "moved": int(snap.sum()),
                "rms_before_mm": before * 1000.0,
                "rms_after_mm": after * 1000.0,
            }
        )

    result = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices),
        o3d.utility.Vector3iVector(np.asarray(mesh.triangles)),
    )
    if mesh.has_vertex_colors():
        result.vertex_colors = mesh.vertex_colors
    # Assign the regularized normals instead of recomputing from geometry.
    result.vertex_normals = o3d.utility.Vector3dVector(out_normals)
    return result, {
        "planes": plane_report,
        "moved_vertices": moved_total,
        "normals_regularized": normalized_total,
        "vertices": int(len(vertices)),
        "band": band,
        "max_shift": max_shift,
        "normal_pull_weight": normal_pull_weight,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--npz", type=Path, required=True,
                        help="reconstruction NPZ (for the gravity direction)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--band", type=float, default=0.06,
                        help="metres around a plane that are eligible to snap")
    parser.add_argument("--max-shift", type=float, default=0.015,
                        help="cap on how far one vertex may move (m)")
    parser.add_argument("--normal-tolerance-deg", type=float, default=25.0,
                        help="skip vertices whose normal deviates more than this "
                             "from the plane normal (wall-mounted detail)")
    parser.add_argument("--normal-pull-weight", type=float, default=0.85,
                        help="how strongly to blend near-plane vertex normals "
                             "toward the plane normal (0-1). Vertex positions "
                             "alone leave the normals jittery, and a directional "
                             "light turns that into visible potholes.")
    parser.add_argument("--plane-threshold", type=float, default=0.03)
    parser.add_argument("--min-plane-points", type=int, default=2000)
    parser.add_argument("--max-planes", type=int, default=12)
    parser.add_argument("--iterations", type=int, default=3,
                        help="snap passes. One pass leaves vertices a fraction "
                             "of the way to the plane (the fade ramp stops "
                             "hard snapping from creasing boundaries); each "
                             "pass re-fits the planes and shrinks the rest.")
    args = parser.parse_args()

    import open3d as o3d

    mesh = o3d.io.read_triangle_mesh(str(args.mesh))
    if not len(mesh.vertices):
        raise RuntimeError(f"empty mesh: {args.mesh}")
    mesh.compute_vertex_normals()
    values = np.load(args.npz)
    gravity = estimate_gravity(values["extrinsics"])
    print(f"mesh: {len(mesh.vertices)} verts, {len(mesh.triangles)} tris")
    rounds = []
    result = mesh
    for iteration in range(max(1, int(args.iterations))):
        result, report = regularize(
            result,
            gravity,
            {
                "band": args.band,
                "max_shift": args.max_shift,
                "normal_tolerance_deg": args.normal_tolerance_deg,
                "normal_pull_weight": args.normal_pull_weight,
                "plane_threshold": args.plane_threshold,
                "min_plane_points": args.min_plane_points,
                "max_planes": args.max_planes,
            },
        )
        rounds.append(report)
        wall_rms = [
            plane["rms_after_mm"]
            for plane in report["planes"]
            if plane.get("moved", 0) and plane.get("rms_after_mm") is not None
        ]
        print(f"  pass {iteration + 1}: moved {report['moved_vertices']} | "
              f"wall rms after {min(wall_rms) if wall_rms else -1:.1f}-"
              f"{max(wall_rms) if wall_rms else -1:.1f} mm")
    report = {
        "input": str(args.mesh),
        "output": str(args.out),
        "iterations": max(1, int(args.iterations)),
        "passes": rounds,
        "final": {
            "moved_vertices": rounds[-1]["moved_vertices"],
            "normals_regularized": rounds[-1]["normals_regularized"],
            "vertices": rounds[-1]["vertices"],
        },
        "band": args.band,
        "max_shift": args.max_shift,
        "normal_pull_weight": args.normal_pull_weight,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(args.out), result)
    args.out.with_suffix(".json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()