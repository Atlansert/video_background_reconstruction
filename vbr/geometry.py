"""Foreground-aware surface fitting and Manhattan-room completion."""

from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import numpy as np


def save_pointcloud(points, path, colors=None):
    import open3d as o3d

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(array))
    if colors is not None and len(colors) == len(array):
        color_array = np.asarray(colors, dtype=np.float64).reshape(-1, 3)
        if color_array.max(initial=0) > 1:
            color_array /= 255.0
        cloud.colors = o3d.utility.Vector3dVector(color_array.clip(0, 1))
    if not o3d.io.write_point_cloud(str(path), cloud):
        raise RuntimeError(f"Failed to write point cloud {path}")


def estimate_gravity(extrinsics: np.ndarray | None) -> np.ndarray:
    if extrinsics is None or not len(extrinsics):
        return np.array([0.0, 1.0, 0.0])
    camera_down = []
    for extrinsic in np.asarray(extrinsics):
        direction = extrinsic[:3, :3].T @ np.array([0.0, 1.0, 0.0])
        direction /= np.linalg.norm(direction) + 1e-12
        if camera_down and np.dot(direction, camera_down[0]) < 0:
            direction = -direction
        camera_down.append(direction)
    gravity = np.mean(camera_down, axis=0)
    return gravity / (np.linalg.norm(gravity) + 1e-12)


def fit_planes(points, gravity, threshold=0.05, min_points=500, max_planes=12):
    import open3d as o3d

    points = np.asarray(points, dtype=np.float64)
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    remaining = np.arange(len(points))
    planes = []
    for _ in range(max_planes):
        if len(remaining) < min_points:
            break
        model, local_indices = cloud.segment_plane(
            distance_threshold=threshold, ransac_n=3, num_iterations=1500
        )
        if len(local_indices) < min_points:
            break
        normal = np.asarray(model[:3], dtype=float)
        normal /= np.linalg.norm(normal) + 1e-12
        alignment = abs(float(np.dot(normal, gravity)))
        kind = "horizontal" if alignment >= 0.75 else "vertical" if alignment <= 0.3 else "other"
        global_indices = remaining[np.asarray(local_indices, dtype=int)]
        planes.append(
            {
                "normal": normal,
                "d": float(model[3]),
                "indices": global_indices,
                "points": len(global_indices),
                "gravity_alignment": alignment,
                "kind": kind,
            }
        )
        keep = np.ones(len(remaining), dtype=bool)
        keep[np.asarray(local_indices, dtype=int)] = False
        remaining = remaining[keep]
        cloud = cloud.select_by_index(local_indices, invert=True)
    return planes


def _plan_axes(gravity):
    """Orthonormal (axis_u, axis_v) spanning the horizontal plane."""
    reference = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(reference, gravity)) > 0.85:
        reference = np.array([0.0, 0.0, 1.0])
    axis_u = np.cross(gravity, reference)
    axis_u /= np.linalg.norm(axis_u) + 1e-12
    axis_v = np.cross(gravity, axis_u)
    axis_v /= np.linalg.norm(axis_v) + 1e-12
    return axis_u, axis_v


def fit_wall_lines(points, gravity, floor_level, ceiling_level, cfg):
    """Detect vertical walls as lines in the horizontal footprint plane.

    2D RANSAC on mid-height background points (floor and ceiling slabs
    excluded). Walls show up as straight lines in the plan view even when 3D
    RANSAC prefers the larger horizontal planes, which is why the previous
    pipeline fell back to footprint walls. Room-relative thresholds keep the
    behaviour stable across videos with different monocular scales.

    Returns walls as dicts with a 3D unit normal pointing away from the
    room interior, the plane offset, and the tangent extent [low, high].
    """
    points = np.asarray(points, dtype=np.float64)
    room_height = float(floor_level - ceiling_level)
    if room_height <= 0:
        return []
    heights = points @ gravity
    middle_low = ceiling_level + 0.15 * room_height
    middle_high = floor_level - 0.15 * room_height
    middle = (heights >= middle_low) & (heights <= middle_high)
    middle_points = points[middle]
    min_support = max(
        int(cfg.get("wall_min_support_points", 120)),
        int(len(middle_points) * cfg.get("wall_min_support_fraction", 0.03)),
    )
    if len(middle_points) < min_support:
        return []

    axis_u, axis_v = _plan_axes(gravity)
    uv = np.stack([middle_points @ axis_u, middle_points @ axis_v], axis=1)
    threshold = cfg.get("wall_inlier_room_fraction", 0.12) * room_height
    iterations = int(cfg.get("wall_ransac_iterations", 400))
    rng = np.random.default_rng(cfg.get("wall_ransac_seed", 7))

    candidates = []
    for _ in range(iterations):
        first, second = rng.choice(len(uv), size=2, replace=False)
        direction = uv[second] - uv[first]
        length = np.linalg.norm(direction)
        if length < 0.5 * room_height:
            continue
        direction /= length
        normal_2d = np.array([-direction[1], direction[0]])
        distance = (uv - uv[first]) @ normal_2d
        inliers = np.abs(distance) < threshold
        count = int(np.count_nonzero(inliers))
        if count < min_support:
            continue
        candidates.append((count, normal_2d, float(uv[first] @ normal_2d), inliers))
        if count >= 0.6 * len(uv):
            break

    accepted = []
    for count, normal_2d, offset, inliers in sorted(
        candidates, key=lambda item: item[0], reverse=True
    ):
        duplicate = False
        for kept in accepted:
            angle = abs(float(np.dot(normal_2d, kept["normal_2d"])))
            if angle > np.cos(np.radians(8.0)) and abs(offset - kept["offset"]) < threshold:
                duplicate = True
                break
            # Same line, opposite normal sign.
            if angle < -np.cos(np.radians(8.0)) and abs(offset + kept["offset"]) < threshold:
                duplicate = True
                break
        if duplicate:
            continue
        accepted.append(
            {"normal_2d": normal_2d, "offset": offset, "inliers": inliers, "count": count}
        )

    min_width = max(
        float(cfg.get("min_wall_width", 0.3)), 0.5 * room_height
    )
    max_gap = cfg.get("wall_max_gap_room_fraction", 0.5) * room_height
    walls = []
    for line in accepted[: int(cfg.get("max_walls", 8))]:
        inliers = line["inliers"]
        normal = axis_u * line["normal_2d"][0] + axis_v * line["normal_2d"][1]
        unoriented_offset = line["offset"]
        # Orient the normal away from the room interior (the mid-height
        # cloud's centroid lies on the interior side), so "behind the wall" —
        # where see-through evidence is collected — is the positive side.
        if float(np.mean(middle_points @ normal)) > unoriented_offset:
            normal = -normal
            offset = -unoriented_offset
        else:
            offset = unoriented_offset
        tangent = np.cross(gravity, normal)
        tangent /= np.linalg.norm(tangent) + 1e-12
        along = np.sort(middle_points[inliers] @ tangent)
        # Keep only the longest contiguous run of support: a real wall's
        # inliers are contiguous, while accidental collinear points from
        # different rooms leave long gaps along the line.
        splits = np.flatnonzero(np.diff(along) > max_gap)
        run = max(np.split(along, splits + 1), key=len)
        low, high = np.quantile(run, [0.02, 0.98])
        if high - low < min_width or len(run) < min_support:
            continue
        walls.append(
            {
                "normal": normal,
                "offset": offset,
                "low": float(low),
                "high": float(high),
                "support": int(len(run)),
            }
        )
    return walls


def _quad(vertices, color):
    import open3d as o3d

    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(vertices, dtype=float)),
        o3d.utility.Vector3iVector(np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int32)),
    )
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.tile(color, (4, 1)))
    mesh.compute_vertex_normals()
    return mesh


def _emit_wall(
    points,
    colors,
    gravity,
    axis_u,
    axis_v,
    floor_level,
    ceiling_level,
    wall,
    cfg,
    opening_hints=None,
):
    """Emit one wall as a single quad, or as strips around carved openings.

    Two opening mechanisms combine:
    - See-through evidence: a large hole in the wall's point coverage with
      background points observed *behind* the wall (doorway, stairwell).
    - Preserve-mask hints: per-frame fixed-structure masks (door, window,
      stairs, cabinets, ...) projected onto the wall grid with depth-based
      visibility; cells covered by a majority of observing views are carved.

    Without such evidence an unobserved region stays solid, so occlusion
    gaps never punch holes.
    """
    import open3d as o3d

    normal = wall["normal"]
    offset = wall["offset"]
    tangent = np.cross(gravity, normal)
    tangent /= np.linalg.norm(tangent) + 1e-12
    room_height = float(floor_level - ceiling_level)

    band = max(
        float(cfg.get("voxel_size", 0.03)) * 2.0,
        cfg.get("wall_band_room_fraction", 0.15) * room_height,
    )
    distances = points @ normal - offset
    t_values = points @ tangent
    h_values = points @ gravity
    inside = (
        (np.abs(distances) <= band)
        & (t_values >= wall["low"])
        & (t_values <= wall["high"])
        & (h_values >= ceiling_level)
        & (h_values <= floor_level)
    )
    behind = distances > band

    cell = max(cfg.get("wall_cell_room_fraction", 0.2) * room_height, 1e-6)
    span_t = wall["high"] - wall["low"]
    span_h = room_height
    n_t = int(np.clip(np.ceil(span_t / cell), 1, cfg.get("wall_max_cells_per_axis", 48)))
    n_h = int(np.clip(np.ceil(span_h / cell), 1, 24))

    observed = np.zeros((n_h, n_t), dtype=bool)
    if inside.any():
        columns = np.clip(
            ((t_values[inside] - wall["low"]) / span_t * n_t).astype(int), 0, n_t - 1
        )
        # Rows grow from the ceiling (min height) toward the floor (max height),
        # matching gravity pointing down.
        rows = np.clip(
            ((h_values[inside] - ceiling_level) / span_h * n_h).astype(int), 0, n_h - 1
        )
        observed[rows, columns] = True

    behind_grid = np.zeros((n_h, n_t), dtype=bool)
    if behind.any():
        behind_columns = np.clip(
            ((t_values[behind] - wall["low"]) / span_t * n_t).astype(int),
            0,
            n_t - 1,
        )
        behind_rows = np.clip(
            ((h_values[behind] - ceiling_level) / span_h * n_h).astype(int),
            0,
            n_h - 1,
        )
        behind_grid[behind_rows, behind_columns] = True

    opening = np.zeros((n_h, n_t), dtype=bool)

    # Preserve-mask hint carving: project wall cell centers into every hint
    # view whose camera sits on the wall's interior side, and carve cells
    # that a majority of unoccluded views mark as fixed structure / opening.
    if opening_hints:
        col_centers = wall["low"] + span_t * (np.arange(n_t) + 0.5) / n_t
        row_centers = ceiling_level + span_h * (np.arange(n_h) + 0.5) / n_h
        mesh_t, mesh_h = np.meshgrid(col_centers, row_centers)
        centers = (
            normal[None] * offset
            + tangent[None] * mesh_t.ravel()[:, None]
            + gravity[None] * mesh_h.ravel()[:, None]
        )
        votes = np.zeros(n_h * n_t, dtype=np.int32)
        observed_votes = np.zeros(n_h * n_t, dtype=np.int32)
        associated_votes = np.zeros(n_h * n_t, dtype=np.int32)
        min_vote_fraction = cfg.get("opening_min_vote_fraction", 0.5)
        for hint in opening_hints:
            extrinsic = np.asarray(hint["extrinsic"], dtype=np.float64)
            cam_center = -extrinsic[:3, :3].T @ extrinsic[:3, 3]
            if float(cam_center @ normal - offset) >= 0.0:
                continue
            p_cam = extrinsic[:3, :3] @ centers.T + extrinsic[:3, 3:4]
            depth_z = p_cam[2]
            front = depth_z > 1e-6
            proj = hint["intrinsic"] @ p_cam
            u_model = proj[0] / np.where(front, proj[2], 1.0)
            v_model = proj[1] / np.where(front, proj[2], 1.0)
            coords = hint["original_coords"]
            width_orig = max(coords[2] - coords[0], 1e-6)
            height_orig = max(coords[3] - coords[1], 1e-6)
            ox = (u_model - coords[0]) / width_orig * coords[4]
            oy = (v_model - coords[1]) / height_orig * coords[5]
            in_bounds = (
                front
                & (ox >= 0)
                & (ox < coords[4])
                & (oy >= 0)
                & (oy < coords[5])
            )
            if not in_bounds.any():
                continue
            # ox/oy index the ORIGINAL-resolution preserve mask; the depth
            # map lives in model space, so it gets its own indices.
            ox_idx = np.clip(ox.astype(int), 0, int(coords[4]) - 1)
            oy_idx = np.clip(oy.astype(int), 0, int(coords[5]) - 1)
            depth_map = hint["depth"]
            mx_idx = np.clip(u_model.astype(int), 0, depth_map.shape[1] - 1)
            my_idx = np.clip(v_model.astype(int), 0, depth_map.shape[0] - 1)
            depth_at = depth_map[my_idx, mx_idx]
            # A mask pixel describes this wall cell when its surface sits at
            # the wall plane (flush door/window) or in front of it (stairs,
            # cabinets, fridge occluding the wall). A pixel whose depth is
            # far beyond the wall belongs to a different surface and must
            # not carve this cell.
            associated = (
                in_bounds & (depth_at > 0) & (depth_at <= depth_z * 1.15)
            )
            associated_votes += associated.astype(np.int32)
            votes += (associated & (hint["mask"][oy_idx, ox_idx] > 0)).astype(
                np.int32
            )
        # The majority is taken over the views whose depth associates the
        # projected pixel with this wall, not over all in-bounds views.
        needed = np.maximum(1, np.ceil(min_vote_fraction * associated_votes).astype(int))
        carved = (associated_votes > 0) & (votes >= needed)
        if os.environ.get("VBR_DEBUG_WALL_VOTES"):
            print(
                f"[wall votes] grid=({n_h}x{n_t}) cells_with_assoc="
                f"{int((associated_votes > 0).sum())} cells_with_votes="
                f"{int((votes > 0).sum())} max_assoc={int(associated_votes.max())} "
                f"max_votes={int(votes.max())} carved={int(carved.sum())}"
            )
        if carved.any():
            opening |= carved.reshape(n_h, n_t)

    try:
        from scipy import ndimage

        # Opening candidates are unobserved cells backed by see-through
        # evidence behind the wall; metric size filters then reject speckle
        # and coverage gaps.
        candidates = (~observed) & ndimage.binary_dilation(behind_grid, iterations=1)
        labels, count = ndimage.label(candidates)
        for label_id in range(1, count + 1):
            component = labels == label_id
            rows_idx, cols_idx = np.nonzero(component)
            if cols_idx.min() == 0 or cols_idx.max() == n_t - 1:
                # Touching a vertical wall edge is a coverage gap, not an
                # opening; only the floor edge (doorways) is allowed.
                continue
            width = (cols_idx.max() - cols_idx.min() + 1) * cell
            height = (rows_idx.max() - rows_idx.min() + 1) * cell
            area = len(rows_idx) * cell**2
            if (
                width < cfg.get("opening_min_width_room_fraction", 0.3) * room_height
                or height
                < cfg.get("opening_min_height_room_fraction", 0.35) * room_height
                or area < cfg.get("opening_min_area_room_fraction", 0.12) * room_height**2
            ):
                continue
            opening |= component
    except ImportError:
        opening = np.zeros((n_h, n_t), dtype=bool)

    openings = int(np.count_nonzero(opening))
    if openings == 0:
        return _quad(
            [
                normal * offset + tangent * wall["low"] + gravity * ceiling_level,
                normal * offset + tangent * wall["high"] + gravity * ceiling_level,
                normal * offset + tangent * wall["high"] + gravity * floor_level,
                normal * offset + tangent * wall["low"] + gravity * floor_level,
            ],
            wall["color"],
        ), 0

    # Emit solid cells as horizontal strip quads (run-length merged per row).
    solid_colors = {}
    if inside.any():
        support_colors = colors[inside]
        support_keys = rows * n_t + columns
        for key in np.unique(support_keys):
            solid_colors[int(key)] = np.median(support_colors[support_keys == key], axis=0)
    mesh = o3d.geometry.TriangleMesh()
    for row in range(n_h):
        column = 0
        while column < n_t:
            if opening[row, column]:
                column += 1
                continue
            end = column
            while end < n_t and not opening[row, end]:
                end += 1
            t_lo = wall["low"] + span_t * column / n_t
            t_hi = wall["low"] + span_t * end / n_t
            h_lo = ceiling_level + span_h * row / n_h
            h_hi = ceiling_level + span_h * (row + 1) / n_h
            color = np.median(colors, axis=0)
            sampled = [solid_colors.get(row * n_t + c) for c in range(column, end)]
            sampled = [s for s in sampled if s is not None]
            if sampled:
                color = np.median(np.stack(sampled), axis=0)
            mesh += _quad(
                [
                    normal * offset + tangent * t_lo + gravity * h_hi,
                    normal * offset + tangent * t_hi + gravity * h_hi,
                    normal * offset + tangent * t_hi + gravity * h_lo,
                    normal * offset + tangent * t_lo + gravity * h_lo,
                ],
                color,
            )
            column = end
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.compute_vertex_normals()
    return mesh, openings


def build_structural_mesh(points, colors, planes, gravity, cfg, opening_hints=None):
    import open3d as o3d

    points = np.asarray(points, dtype=float)
    colors = np.asarray(colors, dtype=float)
    if colors.size == 0:
        colors = np.full_like(points, 0.72)
    if colors.max(initial=0) > 1:
        colors /= 255.0

    heights = points @ gravity
    ceiling_level, floor_level = np.quantile(heights, [0.015, 0.985])
    axis_u, axis_v = _plan_axes(gravity)
    u_values, v_values = points @ axis_u, points @ axis_v
    u_min, u_max = np.quantile(u_values, [0.01, 0.99])
    v_min, v_max = np.quantile(v_values, [0.01, 0.99])

    floor_color = np.median(colors[heights >= np.quantile(heights, 0.94)], axis=0)
    ceiling_color = np.median(colors[heights <= np.quantile(heights, 0.06)], axis=0)
    structural = o3d.geometry.TriangleMesh()
    structural += _quad(
        [
            axis_u * u_min + axis_v * v_min + gravity * floor_level,
            axis_u * u_max + axis_v * v_min + gravity * floor_level,
            axis_u * u_max + axis_v * v_max + gravity * floor_level,
            axis_u * u_min + axis_v * v_max + gravity * floor_level,
        ],
        floor_color,
    )
    structural += _quad(
        [
            axis_u * u_min + axis_v * v_max + gravity * ceiling_level,
            axis_u * u_max + axis_v * v_max + gravity * ceiling_level,
            axis_u * u_max + axis_v * v_min + gravity * ceiling_level,
            axis_u * u_min + axis_v * v_min + gravity * ceiling_level,
        ],
        ceiling_color,
    )

    walls = []
    wall_source = "ransac"
    for plane in planes:
        if plane["kind"] != "vertical" or len(walls) >= cfg.get("max_walls", 8):
            continue
        indices = plane["indices"]
        inliers = points[indices]
        normal = np.asarray(plane["normal"], dtype=float)
        normal -= gravity * np.dot(normal, gravity)
        normal /= np.linalg.norm(normal) + 1e-12
        tangent = np.cross(gravity, normal)
        tangent /= np.linalg.norm(tangent) + 1e-12
        offset = float(np.median(inliers @ normal))
        tangent_values = inliers @ tangent
        low, high = np.quantile(tangent_values, [0.01, 0.99])
        if high - low < cfg.get("min_wall_width", 0.3):
            continue
        walls.append(
            {
                "normal": normal,
                "offset": offset,
                "low": float(low),
                "high": float(high),
                "color": np.median(colors[indices], axis=0),
            }
        )

    if not walls and cfg.get("plan_wall_detection", True):
        detected = fit_wall_lines(points, gravity, floor_level, ceiling_level, cfg)
        color_band = max(
            float(cfg.get("voxel_size", 0.03)) * 2,
            cfg.get("wall_band_room_fraction", 0.15) * (floor_level - ceiling_level),
        )
        for wall in detected:
            nearby = np.abs(points @ wall["normal"] - wall["offset"]) <= color_band
            color = np.median(colors[nearby], axis=0) if np.any(nearby) else np.median(colors, axis=0)
            wall["color"] = color
            walls.append(wall)
        if walls:
            wall_source = "plan_ransac"

    if not walls and cfg.get("footprint_wall_fallback", True):
        wall_source = "robust_footprint_fallback"
        height_low = ceiling_level + 0.15 * (floor_level - ceiling_level)
        height_high = floor_level - 0.15 * (floor_level - ceiling_level)
        middle = (heights >= height_low) & (heights <= height_high)
        middle_points = points[middle]
        # Extents are expressed in each wall's tangent coordinate
        # t = cross(gravity, normal); for the axis_v walls that coordinate is
        # -u, so the range is (-u_max, -u_min), not (u_min, u_max).
        fallback_walls = [
            (axis_u, u_min, v_min, v_max, u_values),
            (axis_u, u_max, v_min, v_max, u_values),
            (axis_v, v_min, -u_max, -u_min, v_values),
            (axis_v, v_max, -u_max, -u_min, v_values),
        ]
        for normal, offset, low, high, coordinate_values in fallback_walls:
            band = max(
                float(cfg.get("voxel_size", 0.03)) * 2,
                0.05 * float(np.ptp(coordinate_values)),
            )
            nearby = middle & (np.abs(coordinate_values - offset) <= band)
            # Anchor the fallback plane at the nearby wall-point median so
            # it matches the actually observed surface instead of the raw
            # footprint quantile (which drifts with furniture placement).
            if np.any(nearby):
                offset = float(np.median(coordinate_values[nearby]))
            color = np.median(colors[nearby], axis=0) if np.any(nearby) else np.median(colors, axis=0)
            # Orient the normal away from the room interior so "behind the
            # wall" (used for opening evidence) is the positive side.
            if len(middle_points) and np.median(middle_points @ normal - offset) > 0:
                normal, offset, low, high = -normal, -offset, -high, -low
            walls.append(
                {
                    "normal": normal,
                    "offset": float(offset),
                    "low": float(low),
                    "high": float(high),
                    "color": color,
                }
            )

    wall_count = 0
    openings_total = 0
    for wall in walls:
        wall_mesh, openings = _emit_wall(
            points, colors, gravity, axis_u, axis_v,
            floor_level, ceiling_level, wall, cfg,
            opening_hints=opening_hints,
        )
        structural += wall_mesh
        wall_count += 1
        openings_total += openings

    structural.remove_duplicated_vertices()
    # Quads built from different wall/ceiling expressions can differ by one
    # ulp; merge them so the structural mesh stays a clean closed box.
    structural = structural.merge_close_vertices(
        1e-6 * max(float(floor_level - ceiling_level), 1e-6)
    )
    structural.remove_degenerate_triangles()
    structural.compute_vertex_normals()
    bounds = {
        "ceiling_level": float(ceiling_level),
        "floor_level": float(floor_level),
        "room_height": float(floor_level - ceiling_level),
        "wall_count": wall_count,
        "wall_source": wall_source,
        "wall_openings": openings_total,
        "footprint": [float(u_min), float(u_max), float(v_min), float(v_max)],
    }
    return structural, bounds


def _model_rgb(frame_path: str, coords: np.ndarray, width: int, height: int) -> np.ndarray:
    image = cv2.imread(frame_path)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    x1, y1, x2, y2 = np.rint(coords[:4]).astype(int)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    if image is None or x2 <= x1 or y2 <= y1:
        return canvas
    resized = cv2.resize(image, (x2 - x1, y2 - y1), interpolation=cv2.INTER_CUBIC)
    canvas[y1:y2, x1:x2] = resized[:, :, ::-1]
    return canvas


def build_tsdf_mesh(reconstruction_path: Path, cfg):
    import open3d as o3d

    data = np.load(reconstruction_path)
    depth = data["depth"]
    confidence = data["confidence"]
    cutoff = float(data["confidence_cutoff"])
    foreground = data["foreground_masks"]
    intrinsics = data["intrinsics"]
    extrinsics = data["extrinsics"]
    coords = data["original_coords"]
    frame_paths = data["frame_paths"]
    height = int(depth.shape[1])
    width = int(depth.shape[2])
    positive_depth = depth[..., 0][depth[..., 0] > 0]
    depth_trunc = float(np.quantile(positive_depth, cfg.get("depth_trunc_quantile", 0.995)))
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=cfg.get("voxel_size", 0.03),
        sdf_trunc=cfg.get("tsdf_trunc", cfg.get("voxel_size", 0.03) * 5),
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )
    for index, frame_path in enumerate(frame_paths):
        depth_image = depth[index, ..., 0].copy()
        depth_image[(confidence[index] < cutoff) | foreground[index]] = 0
        color_image = _model_rgb(str(frame_path), coords[index], width, height)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(color_image),
            o3d.geometry.Image(depth_image.astype(np.float32)),
            depth_scale=1.0,
            depth_trunc=depth_trunc,
            convert_rgb_to_intensity=False,
        )
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width,
            height,
            float(intrinsics[index, 0, 0]),
            float(intrinsics[index, 1, 1]),
            float(intrinsics[index, 0, 2]),
            float(intrinsics[index, 1, 2]),
        )
        extrinsic = np.eye(4)
        extrinsic[:3] = extrinsics[index]
        volume.integrate(rgbd, intrinsic, extrinsic)
    mesh = volume.extract_triangle_mesh()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_non_manifold_edges()
    mesh.compute_vertex_normals()
    return mesh


def build_poisson_mesh(points, colors, cfg):
    import open3d as o3d

    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(points)))
    if colors is not None and len(colors) == len(points):
        color_array = np.asarray(colors, dtype=float)
        if color_array.max(initial=0) > 1:
            color_array /= 255.0
        cloud.colors = o3d.utility.Vector3dVector(color_array)
    cloud.estimate_normals()
    cloud.orient_normals_consistent_tangent_plane(
        min(cfg.get("normal_neighbors", 30), max(3, len(cloud.points) - 1))
    )
    max_points = cfg.get("mesh_max_points", 150000)
    if len(cloud.points) > max_points:
        cloud = cloud.random_down_sample(max_points / len(cloud.points))
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        cloud, depth=cfg.get("poisson_depth", 8)
    )
    densities = np.asarray(densities)
    if len(densities):
        mesh.remove_vertices_by_mask(densities < np.quantile(densities, 0.02))
    mesh = mesh.crop(cloud.get_axis_aligned_bounding_box())
    mesh.compute_vertex_normals()
    return mesh


def _polygon_normal_2d(vector):
    normal = np.cross(vector[:-1], vector[1:]).sum()
    return normal


def _project_loop_to_2d(vertices):
    """Project a boundary loop onto its best-fit plane (Newell's method)."""
    points = np.asarray(vertices, dtype=float)
    normal = np.zeros(3)
    for index in range(len(points)):
        current = points[index]
        following = points[(index + 1) % len(points)]
        normal += np.array(
            [
                (current[1] - following[1]) * (current[2] + following[2]),
                (current[2] - following[2]) * (current[0] + following[0]),
                (current[0] - following[0]) * (current[1] + following[1]),
            ]
        )
    norm = np.linalg.norm(normal)
    if norm < 1e-12:
        normal = np.array([0.0, 0.0, 1.0])
    else:
        normal /= norm
    reference = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(reference, normal))) > 0.9:
        reference = np.array([0.0, 0.0, 1.0])
    axis_u = np.cross(normal, reference)
    axis_u /= np.linalg.norm(axis_u) + 1e-12
    axis_v = np.cross(normal, axis_u)
    uv = np.stack([points @ axis_u, points @ axis_v], axis=1)
    return uv, normal


def _ear_clip(uv):
    """Triangulate a simple polygon with ear clipping.

    ``uv`` must be oriented counter-clockwise (positive signed area).
    Returns a list of (a, b, c) local vertex indices.
    """
    points = [tuple(point) for point in uv]
    order = list(range(len(points)))
    area = 0.5 * sum(
        points[order[i]][0] * points[order[(i + 1) % len(order)]][1]
        - points[order[(i + 1) % len(order)]][0] * points[order[i]][1]
        for i in range(len(order))
    )
    if area < 0:
        order.reverse()
        area = -area
    if area < 1e-12:
        return []

    def inside(point, triangle):
        a, b, c = triangle
        def cross(o, p, q):
            return (p[0] - o[0]) * (q[1] - o[1]) - (p[1] - o[1]) * (q[0] - o[0])
        signs = [
            cross(a, b, point),
            cross(b, c, point),
            cross(c, a, point),
        ]
        return all(value >= 0 for value in signs) or all(value <= 0 for value in signs)

    triangles = []
    remaining = order[:]
    guard = 0
    while len(remaining) > 3 and guard < 10 * len(order):
        guard += 1
        clipped = False
        for index in range(len(remaining)):
            previous = remaining[index - 1]
            current = remaining[index]
            following = remaining[(index + 1) % len(remaining)]
            a, b, c = points[previous], points[current], points[following]
            cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
            if cross <= 0:
                continue
            occluded = any(
                order_index not in (previous, current, following)
                and inside(points[order_index], (a, b, c))
                for order_index in remaining
            )
            if occluded:
                continue
            triangles.append((previous, current, following))
            remaining.pop(index)
            clipped = True
            break
        if not clipped:
            break
    # A loop that could not be fully clipped (self-touching boundary chains,
    # duplicate vertices) must not leave a partially triangulated patch: the
    # caller treats [] as "skip this hole" and keeps the original boundary.
    if len(remaining) != 3:
        return []
    triangles.append((remaining[0], remaining[1], remaining[2]))
    return triangles


def _boundary_loops(mesh):
    """Return the loops of boundary edges as vertex-index lists."""
    triangles = np.asarray(mesh.triangles, dtype=np.int64)
    edge_counts = {}
    for triangle in triangles:
        for first, second in (
            (triangle[0], triangle[1]),
            (triangle[1], triangle[2]),
            (triangle[2], triangle[0]),
        ):
            key = (int(min(first, second)), int(max(first, second)))
            edge_counts[key] = edge_counts.get(key, 0) + 1
    boundary = {key for key, count in edge_counts.items() if count == 1}
    adjacency: dict[int, list[int]] = {}
    for first, second in boundary:
        adjacency.setdefault(first, []).append(second)
        adjacency.setdefault(second, []).append(first)
    loops = []
    visited = set()
    for start in adjacency:
        if start in visited:
            continue
        loop = [start]
        previous, current = -1, start
        visited.add(start)
        while True:
            candidates = [node for node in adjacency[current] if node != previous]
            if not candidates:
                break
            following = candidates[0]
            if following in visited:
                break
            visited.add(following)
            loop.append(following)
            previous, current = current, following
            if current == start:
                break
        if len(loop) >= 3:
            loops.append(loop)
    return loops


def fill_small_boundary_holes(mesh, max_loop_edges=60):
    """Fan-fill boundary loops whose perimeter is small enough to be noise.

    Big loops (doorways, stairs openings) are left untouched. Returns the
    updated mesh and the number of filled loops.

    The mesh is rebuilt once at the end: appending via ``mesh += ...`` while
    the addend's vertex buffer aliases ``mesh.vertices`` (a numpy view)
    corrupts the buffer and duplicates vertices exponentially.
    """
    import open3d as o3d

    loops = _boundary_loops(mesh)
    filled = 0
    closed_loops = [loop for loop in loops if len(loop) <= max_loop_edges]
    original_vertices = np.asarray(mesh.vertices).copy()
    original_faces = np.asarray(mesh.triangles)
    patch_faces = []
    for loop in closed_loops:
        vertices = original_vertices[loop]
        uv, normal = _project_loop_to_2d(vertices)
        orientation = 0.5 * sum(
            uv[i][0] * uv[(i + 1) % len(uv)][1] - uv[(i + 1) % len(uv)][0] * uv[i][1]
            for i in range(len(uv))
        )
        if orientation < 0:
            uv = uv[::-1]
            loop = loop[::-1]
        triangles = _ear_clip(uv)
        if not triangles:
            continue
        local_to_global = np.asarray(loop, dtype=np.int64)
        patch_faces.append(
            np.asarray(
                [[local_to_global[a], local_to_global[b], local_to_global[c]]
                 for a, b, c in triangles],
                dtype=np.int64,
            )
        )
        filled += 1
    if filled:
        all_faces = np.vstack([original_faces] + patch_faces)
        mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(original_vertices),
            o3d.utility.Vector3iVector(all_faces),
        )
        mesh.remove_duplicated_vertices()
        mesh.remove_degenerate_triangles()
        mesh.compute_vertex_normals()
    return mesh, filled


def clean_mesh(mesh, cfg):
    """Drop floating fragments and close small artifact holes.

    Components with fewer than ``mesh_min_component_triangles`` triangles are
    removed (TSDF slivers and Poisson blobs), then boundary loops shorter than
    ``mesh_fill_hole_max_edges`` are ear-clipped shut. Returns the cleaned
    mesh and a statistics dict for the geometry report.
    """
    import open3d as o3d

    stats = {"components_before": 0, "components_after": 0,
             "largest_fraction": 1.0, "boundary_loops": 0, "holes_filled": 0}
    triangle_count = len(np.asarray(mesh.triangles))
    if triangle_count == 0:
        return mesh, stats
    clusters, triangle_counts, _ = mesh.cluster_connected_triangles()
    cluster_sizes = np.bincount(np.asarray(clusters))
    stats["components_before"] = int(len(cluster_sizes))
    if len(cluster_sizes) > 1:
        largest = int(cluster_sizes.max())
        min_triangles = int(cfg.get("mesh_min_component_triangles", 100))
        keep = set(
            int(cluster)
            for cluster, size in enumerate(cluster_sizes)
            if size >= min_triangles
        )
        triangles = np.asarray(mesh.triangles)
        mask = np.asarray([cluster in keep for cluster in clusters])
        if mask.any():
            mesh = o3d.geometry.TriangleMesh(
                o3d.utility.Vector3dVector(np.asarray(mesh.vertices)),
                o3d.utility.Vector3iVector(triangles[mask]),
            )
            mesh.remove_duplicated_vertices()
            mesh.remove_degenerate_triangles()
        stats["largest_fraction"] = largest / max(1, int(triangle_count))
    stats["components_after"] = len(
        np.unique(np.asarray(mesh.cluster_connected_triangles()[0]))
    )
    loops = _boundary_loops(mesh)
    stats["boundary_loops"] = int(len(loops))
    mesh, filled = fill_small_boundary_holes(
        mesh, max_loop_edges=int(cfg.get("mesh_fill_hole_max_edges", 60))
    )
    stats["holes_filled"] = filled
    return mesh, stats


def build_mesh(points, colors, output_dir, cfg, reconstruction_path=None, opening_hints=None):
    import open3d as o3d

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    colors = np.asarray(colors, dtype=float).reshape(-1, 3) if colors is not None else None
    if not len(points):
        raise RuntimeError("Cannot fit geometry without reconstructed background points")

    extrinsics = None
    if reconstruction_path and Path(reconstruction_path).exists():
        with np.load(reconstruction_path) as data:
            extrinsics = data["extrinsics"]
    gravity = estimate_gravity(extrinsics)
    planes = fit_planes(
        points,
        gravity,
        threshold=cfg.get("plane_distance_threshold", 0.04),
        min_points=cfg.get("min_plane_points", 1000),
        max_planes=cfg.get("max_planes", 12),
    )
    structural, bounds = build_structural_mesh(
        points, colors, planes, gravity, cfg, opening_hints=opening_hints
    )
    structural_path = output_dir / "structural_planes.ply"
    o3d.io.write_triangle_mesh(str(structural_path), structural)

    surface_method = "poisson"
    surface = None
    tsdf_error = None
    if cfg.get("use_tsdf", True) and reconstruction_path:
        try:
            surface = build_tsdf_mesh(Path(reconstruction_path), cfg)
            if len(surface.vertices) == 0:
                raise RuntimeError("TSDF returned an empty mesh")
            surface_method = "tsdf"
        except Exception as error:
            tsdf_error = str(error)
    if surface is None:
        surface = build_poisson_mesh(points, colors, cfg)

    combined = surface + structural
    combined.remove_degenerate_triangles()
    combined.remove_duplicated_triangles()
    combined.remove_non_manifold_edges()
    combined.compute_vertex_normals()
    combined, mesh_clean = clean_mesh(combined, cfg)
    target_triangles = cfg.get("mesh_target_triangles", 250000)
    if len(combined.triangles) > target_triangles:
        combined = combined.simplify_quadric_decimation(target_triangles)
    mesh_path = output_dir / "background_mesh.ply"
    if not o3d.io.write_triangle_mesh(str(mesh_path), combined, write_ascii=False):
        raise RuntimeError(f"Failed to write {mesh_path}")

    glb_error = None
    try:
        import trimesh

        trimesh.load(str(mesh_path), process=False).export(output_dir / "background_scene.glb")
    except Exception as error:
        glb_error = str(error)

    serializable_planes = []
    for plane in planes:
        serializable_planes.append(
            {
                "normal": np.asarray(plane["normal"]).tolist(),
                "d": plane["d"],
                "points": plane["points"],
                "gravity_alignment": plane["gravity_alignment"],
                "kind": plane["kind"],
            }
        )
    report = {
        "gravity_down": gravity.tolist(),
        "planes": serializable_planes,
        "room": bounds,
        "surface_method": surface_method,
        "surface_vertices": len(surface.vertices),
        "structural_vertices": len(structural.vertices),
        "combined_vertices": len(combined.vertices),
        "combined_triangles": len(combined.triangles),
        "mesh_clean": mesh_clean,
        "tsdf_error": tsdf_error,
        "glb_error": glb_error,
    }
    (output_dir / "geometry_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report
