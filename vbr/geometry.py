"""Foreground-aware surface fitting and Manhattan-room completion."""

from __future__ import annotations

import json
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


def _quad(vertices, color):
    import open3d as o3d

    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(vertices, dtype=float)),
        o3d.utility.Vector3iVector(np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int32)),
    )
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.tile(color, (4, 1)))
    mesh.compute_vertex_normals()
    return mesh


def build_structural_mesh(points, colors, planes, gravity, cfg):
    import open3d as o3d

    points = np.asarray(points, dtype=float)
    colors = np.asarray(colors, dtype=float)
    if colors.size == 0:
        colors = np.full_like(points, 0.72)
    if colors.max(initial=0) > 1:
        colors /= 255.0

    heights = points @ gravity
    ceiling_level, floor_level = np.quantile(heights, [0.015, 0.985])
    reference = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(reference, gravity)) > 0.85:
        reference = np.array([0.0, 0.0, 1.0])
    axis_u = np.cross(gravity, reference)
    axis_u /= np.linalg.norm(axis_u) + 1e-12
    axis_v = np.cross(gravity, axis_u)
    axis_v /= np.linalg.norm(axis_v) + 1e-12
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

    wall_count = 0
    wall_source = "ransac"
    for plane in planes:
        if plane["kind"] != "vertical" or wall_count >= cfg.get("max_walls", 8):
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
        color = np.median(colors[indices], axis=0)
        structural += _quad(
            [
                normal * offset + tangent * low + gravity * ceiling_level,
                normal * offset + tangent * high + gravity * ceiling_level,
                normal * offset + tangent * high + gravity * floor_level,
                normal * offset + tangent * low + gravity * floor_level,
            ],
            color,
        )
        wall_count += 1

    if wall_count == 0 and cfg.get("footprint_wall_fallback", True):
        wall_source = "robust_footprint_fallback"
        height_low = ceiling_level + 0.15 * (floor_level - ceiling_level)
        height_high = floor_level - 0.15 * (floor_level - ceiling_level)
        middle = (heights >= height_low) & (heights <= height_high)
        fallback_walls = [
            (axis_u, u_min, axis_v, v_min, v_max, u_values),
            (axis_u, u_max, axis_v, v_max, v_min, u_values),
            (axis_v, v_min, axis_u, u_max, u_min, v_values),
            (axis_v, v_max, axis_u, u_min, u_max, v_values),
        ]
        for normal, offset, tangent, low, high, coordinate_values in fallback_walls:
            band = max(
                float(cfg.get("voxel_size", 0.03)) * 2,
                0.05 * float(np.ptp(coordinate_values)),
            )
            nearby = middle & (np.abs(coordinate_values - offset) <= band)
            color = np.median(colors[nearby], axis=0) if np.any(nearby) else np.median(colors, axis=0)
            structural += _quad(
                [
                    normal * offset + tangent * low + gravity * ceiling_level,
                    normal * offset + tangent * high + gravity * ceiling_level,
                    normal * offset + tangent * high + gravity * floor_level,
                    normal * offset + tangent * low + gravity * floor_level,
                ],
                color,
            )
            wall_count += 1

    structural.remove_duplicated_vertices()
    structural.remove_degenerate_triangles()
    structural.compute_vertex_normals()
    bounds = {
        "ceiling_level": float(ceiling_level),
        "floor_level": float(floor_level),
        "room_height": float(floor_level - ceiling_level),
        "wall_count": wall_count,
        "wall_source": wall_source,
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


def build_mesh(points, colors, output_dir, cfg, reconstruction_path=None):
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
    structural, bounds = build_structural_mesh(points, colors, planes, gravity, cfg)
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
        "tsdf_error": tsdf_error,
        "glb_error": glb_error,
    }
    (output_dir / "geometry_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report
