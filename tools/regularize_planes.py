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

Normal side (``--normal-pull-weight``): one weighted target normal is
accumulated per vertex across every plane it belongs to, blended ONCE, oriented
from the true geometric normal, and capped by ``--max-normal-drift-deg``. An
earlier version blended per plane in a loop, so a vertex inside two or three
overlapping bands was blended repeatedly toward different normals and ended up
matching none -- 0.47% came out flipped.

IMPORTANT -- what actually reaches a render: ``tools/render_trajectory_video.py``
calls ``mesh.compute_vertex_normals()`` after loading, which OVERWRITES whatever
normals the PLY stores. Rendering a mesh with deliberately garbage normals gives
a pixel-identical image (verified: mean abs diff 0.000000), so the normal field
written here does NOT affect the walkthrough videos or any other Open3D render.
Those depend on vertex POSITIONS only, which is why the position ramp below
matters for the rendered result and the normal blend does not.

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


def _geometric_normals(vertices, faces):
    """Vertex normals implied by the current triangles (area weighted)."""
    import open3d as o3d

    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(vertices, dtype=np.float64)),
        o3d.utility.Vector3iVector(np.asarray(faces)),
    )
    mesh.compute_vertex_normals()
    return np.asarray(mesh.vertex_normals, dtype=np.float64)


def _mesh_graph(faces, vertex_count):
    """Edge lists for smoothing a per-vertex field over the mesh graph."""
    pairs = set()
    for a, b, c in np.asarray(faces):
        pairs.update({(a, b), (b, a), (b, c), (c, b), (c, a), (a, c)})
    source = np.fromiter((p[0] for p in pairs), dtype=np.int64, count=len(pairs))
    target = np.fromiter((p[1] for p in pairs), dtype=np.int64, count=len(pairs))
    degree = np.bincount(source, minlength=vertex_count).astype(np.float64)
    return source, target, degree


def _smooth_field(field, source, target, degree, rounds, blend, cap):
    """Average a displacement field with its neighbours, then re-cap it.

    Vertices of one smooth patch are often claimed by different planes and so
    are displaced along different directions; the patch is torn apart and the
    render shows fresh speckle exactly where the surface had been clean.
    Blending the field across mesh edges makes neighbours move together, which
    preserves the smoothness of a smooth patch without giving up the flattening
    of a rough one.
    """
    out = field.copy()
    for _ in range(int(rounds)):
        accumulated = np.zeros_like(out)
        np.add.at(accumulated, source, out[target])
        mean = accumulated / np.maximum(degree[:, None], 1.0)
        out = (1.0 - blend) * out + blend * mean
    magnitude = np.linalg.norm(out, axis=1)
    over = magnitude > cap
    if over.any():
        out[over] *= (cap / magnitude[over])[:, None]
    return out


def _field_roughness(field, source, target):
    """Mean neighbour disagreement of a displacement field (mm).

    A field that tears a flat patch apart shows a large value; a rigid
    displacement of the same patch shows ~0.
    """
    delta = np.linalg.norm(field[source] - field[target], axis=1)
    return float(delta.mean() * 1000.0)


def regularize(mesh, gravity, cfg):
    import open3d as o3d

    vertices = np.asarray(mesh.vertices, dtype=np.float64).copy()
    faces = np.asarray(mesh.triangles)

    band = float(cfg["band"])
    max_shift = float(cfg["max_shift"])
    normal_tolerance = float(np.cos(np.radians(cfg["normal_tolerance_deg"])))
    normal_pull_weight = float(cfg.get("normal_pull_weight", 0.0))
    # Shading normals default to the snap radius: rewriting a normal where the
    # geometry did not move makes the shade disagree with the surface, which is
    # what added speckle to walls this polish was meant to smooth.
    normal_band = float(cfg.get("normal_band") or max_shift)
    max_drift_deg = float(cfg.get("max_normal_drift_deg", 30.0))

    planes = fit_planes(
        vertices,
        gravity,
        threshold=float(cfg["plane_threshold"]),
        min_points=int(cfg["min_plane_points"]),
        max_planes=int(cfg["max_planes"]),
    )

    geometric = _geometric_normals(vertices, faces)
    moved_total = 0
    plane_report = []

    # ---- phase 1: positions -------------------------------------------------
    for plane in planes:
        normal = np.asarray(plane["normal"], dtype=np.float64)
        offset = float(plane["d"])
        distance = vertices @ normal + offset
        in_band = np.abs(distance) < band
        if normal_tolerance > 0:
            in_band &= np.abs(geometric @ normal) >= normal_tolerance
        snap = in_band & (np.abs(distance) < max_shift)
        if snap.any():
            # The ramp must reach zero EXACTLY at max_shift, the radius that
            # defines eligibility. Ramping over a wider band instead (the
            # original 1 - |d|/band with band=0.10 > max_shift=0.04) left a
            # vertex at the cutoff moving ~24 mm while its neighbour just
            # outside moved nothing: a cliff in the displacement field that
            # tears smooth patches into jagged ones and shows up in the render
            # as fresh speckle on walls that were already flat.
            pull = 1.0 - (np.abs(distance[snap]) / max_shift)
            shift = np.clip(-distance[snap] * pull, -max_shift, max_shift)
            before = float(np.sqrt((distance[snap] ** 2).mean()))
            vertices[snap] += shift[:, None] * normal[None]
            after = float(np.sqrt(((vertices[snap] @ normal + offset) ** 2).mean()))
            moved_total += int(snap.sum())
        else:
            before = after = 0.0
        plane_report.append(
            {
                "kind": plane["kind"],
                "points": int(plane["points"]),
                "moved": int(snap.sum()),
                "rms_before_mm": before * 1000.0,
                "rms_after_mm": after * 1000.0,
            }
        )

    # ---- phase 1b: make the displacement field neighbour-consistent -------
    # Applied once by default: more passes over-smooth and start to round off
    # genuine edges (measured pass rates 3/5 for one pass, 0/2 for three).
    field_rounds = int(cfg.get("field_smooth_rounds", 1))
    field_blend = float(cfg.get("field_smooth_blend", 0.5))
    field_before = field_after = None
    if field_rounds > 0 and moved_total:
        source, target, degree = _mesh_graph(faces, len(vertices))
        displacement = vertices - np.asarray(mesh.vertices, dtype=np.float64)
        field_before = _field_roughness(displacement, source, target)
        smoothed = _smooth_field(displacement, source, target, degree,
                                 field_rounds, field_blend, max_shift)
        field_after = _field_roughness(smoothed, source, target)
        vertices = np.asarray(mesh.vertices, dtype=np.float64) + smoothed

    # ---- phase 2: normals, measured against the surface as it now is --------
    geometric = _geometric_normals(vertices, faces)
    out_normals = geometric.copy()
    normalized_total = 0

    if normal_pull_weight > 0:
        # One accumulated target per vertex. The previous version wrote the
        # blend inside the plane loop, so a vertex inside two or three
        # overlapping bands (20.7% of them) was blended repeatedly and the last
        # plane won -- a blend-of-blends matching no real surface.
        target_sum = np.zeros_like(vertices)
        weight_sum = np.zeros(len(vertices))
        for plane in planes:
            normal = np.asarray(plane["normal"], dtype=np.float64)
            offset = float(plane["d"])
            distance = vertices @ normal + offset
            mask = np.abs(distance) < normal_band
            if normal_tolerance > 0:
                mask &= np.abs(geometric @ normal) >= normal_tolerance
            if not mask.any():
                continue
            fade = 1.0 - np.abs(distance[mask]) / normal_band
            # Orientation from the TRUE surface normal. Taking it from an
            # already-blended value flipped normals on geometry that sits
            # near-perpendicular to the plane.
            signs = np.sign(geometric[mask] @ normal).reshape(-1, 1)
            signs[signs == 0] = 1.0
            weight = (fade * normal_pull_weight).reshape(-1, 1)
            target_sum[mask] += normal[None] * signs * weight
            weight_sum[mask] += weight[:, 0]

        active = weight_sum > 0
        if active.any():
            target = target_sum[active] / weight_sum[active][:, None]
            target /= np.maximum(np.linalg.norm(target, axis=1, keepdims=True), 1e-12)
            weight = np.clip(weight_sum[active], 0.0, 1.0).reshape(-1, 1)
            blended = geometric[active] * (1.0 - weight) + target * weight
            blended /= np.maximum(np.linalg.norm(blended, axis=1, keepdims=True), 1e-12)
            # Cap against the CURRENT surface normal; beyond the cap the honest
            # geometric normal is kept, so a shading normal can never end up
            # opposing its own triangle.
            limit = float(np.cos(np.radians(max_drift_deg)))
            drift_dot = np.einsum("ij,ij->i", blended, geometric[active])
            beyond = drift_dot < limit
            if beyond.any():
                blended[beyond] = geometric[active][beyond]
            out_normals[active] = blended
            normalized_total = int(active.sum())

    final_dot = np.einsum("ij,ij->i", out_normals, geometric)
    flipped = int((final_dot < 0.0).sum())
    drift_deg = np.degrees(np.arccos(np.clip(np.abs(final_dot), -1.0, 1.0)))

    result = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices),
        o3d.utility.Vector3iVector(faces),
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
        "normal_band": normal_band,
        "max_normal_drift_deg": max_drift_deg,
        "field_smooth_rounds": field_rounds,
        "field_smooth_blend": field_blend,
        "field_roughness_before_mm": field_before,
        "field_roughness_after_mm": field_after,
        "flipped_normals": flipped,
        "normal_drift_mean_deg": float(drift_deg.mean()),
        "normal_drift_p90_deg": float(np.percentile(drift_deg, 90)),
        "normal_drift_max_deg": float(drift_deg.max()),
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
    parser.add_argument("--field-smooth-rounds", type=int, default=1,
                        help="Laplacian passes over the displacement field. "
                             "Vertices in one smooth patch are often claimed by "
                             "different planes and get displaced differently, "
                             "which tears the patch and adds render speckle; "
                             "blending the field keeps neighbours together. "
                             "0 disables it.")
    parser.add_argument("--field-smooth-blend", type=float, default=0.5,
                        help="weight of the neighbour average per pass (0-1).")
    parser.add_argument("--normal-band", type=float, default=None,
                        help="metres around a plane where vertex normals may be "
                             "re-blended. Defaults to --max-shift, so the shading "
                             "only changes where the geometry actually moved.")
    parser.add_argument("--max-normal-drift-deg", type=float, default=30.0,
                        help="cap on how far a shading normal may end up from the "
                             "true surface normal; beyond it the geometric normal "
                             "is kept (guards flips and blend-of-blends).")
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
                "field_smooth_rounds": args.field_smooth_rounds,
                "field_smooth_blend": args.field_smooth_blend,
                "normal_band": args.normal_band,
                "max_normal_drift_deg": args.max_normal_drift_deg,
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
        "field_smooth_rounds": args.field_smooth_rounds,
        "field_smooth_blend": args.field_smooth_blend,
        "normal_band": args.normal_band or args.max_shift,
        "max_normal_drift_deg": args.max_normal_drift_deg,
        "flipped_normals": rounds[-1].get("flipped_normals"),
        "normal_drift_mean_deg": rounds[-1].get("normal_drift_mean_deg"),
        "normal_drift_max_deg": rounds[-1].get("normal_drift_max_deg"),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(args.out), result)
    args.out.with_suffix(".json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()