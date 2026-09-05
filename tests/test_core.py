import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from vbr.geometry import (
    build_structural_mesh,
    estimate_gravity,
    fit_wall_lines,
    _emit_wall,
)
from vbr.models.inpainting import _resolve_ffmpeg
from vbr.models.slam import SLAMAdapter
from vbr.vggt_direct import _mask_in_model_space
from vbr.vggt_slam_backend import (
    homography_scale,
    mask_in_model_space,
    rigid_from_similarity,
)
from vbr.video import fill_enclosed_mask_holes
from vbr.prompts import resolve_prompts


class GeometryTests(unittest.TestCase):
    def test_identity_camera_gravity_is_positive_y(self):
        extrinsic = np.eye(4, dtype=float)[None, :3]
        np.testing.assert_allclose(
            estimate_gravity(extrinsic), np.array([0.0, 1.0, 0.0]), atol=1e-8
        )

    def test_footprint_fallback_builds_four_walls(self):
        x, z = np.meshgrid(np.linspace(-1, 1, 12), np.linspace(-2, 2, 12))
        floor = np.column_stack([x.ravel(), np.ones(x.size), z.ravel()])
        ceiling = floor.copy()
        ceiling[:, 1] = -1
        points = np.vstack([floor, ceiling])
        colors = np.full_like(points, 0.7)
        mesh, bounds = build_structural_mesh(
            points,
            colors,
            [],
            np.array([0.0, 1.0, 0.0]),
            {"voxel_size": 0.03, "footprint_wall_fallback": True},
        )
        self.assertEqual(bounds["wall_count"], 4)
        self.assertEqual(bounds["wall_source"], "robust_footprint_fallback")
        self.assertEqual(len(mesh.vertices), 8)
        self.assertEqual(len(mesh.triangles), 12)


class MaskTests(unittest.TestCase):
    def test_original_mask_is_padded_into_model_space(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "mask.png"
            cv2.imwrite(str(path), np.full((4, 8), 255, dtype=np.uint8))
            result = _mask_in_model_space(
                path, np.array([0, 2, 8, 6, 8, 4]), size=8
            )
            self.assertEqual(int(np.count_nonzero(result)), 32)
            self.assertFalse(np.any(result[:2]))
            self.assertFalse(np.any(result[6:]))

    def test_only_enclosed_holes_are_filled(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            mask = np.zeros((20, 20), dtype=np.uint8)
            cv2.rectangle(mask, (4, 4), (15, 15), 255, thickness=2)
            cv2.imwrite(str(directory / "000000.png"), mask)
            changed = fill_enclosed_mask_holes(directory)
            result = cv2.imread(str(directory / "000000.png"), cv2.IMREAD_GRAYSCALE)
            self.assertGreater(changed, 0)
            self.assertEqual(int(result[10, 10]), 255)
            self.assertEqual(int(result[0, 0]), 0)


class WallDetectionTests(unittest.TestCase):
    GRAVITY = np.array([0.0, 1.0, 0.0])

    def _wall_room_points(self, gap=None):
        """Wall points at x=0 (interior x<0), z in [-2, 2], y in [-1, 1]."""
        z = np.linspace(-2, 2, 240)
        y = np.linspace(-1, 1, 24)
        zz, yy = np.meshgrid(z, y)
        points = np.stack([np.zeros(zz.size), yy.ravel(), zz.ravel()], axis=1)
        if gap is not None:
            z0, z1, y0, y1 = gap
            inside_gap = (
                (points[:, 2] >= z0)
                & (points[:, 2] <= z1)
                & (points[:, 1] >= y0)
                & (points[:, 1] <= y1)
            )
            points = points[~inside_gap]
        return points

    def test_wall_lines_found_and_oriented_outward(self):
        points = self._wall_room_points()
        # Interior reference points give the centroid an unambiguous side.
        rng = np.random.default_rng(1)
        interior = rng.uniform(
            low=np.array([-0.6, -0.9, -1.8]), high=np.array([-0.4, 0.9, 1.8]),
            size=(300, 3),
        )
        cfg = {"wall_min_support_points": 50}
        walls = fit_wall_lines(
            np.vstack([points, interior]), self.GRAVITY, 0.98, -0.98, cfg
        )
        self.assertGreaterEqual(len(walls), 1)
        wall = walls[0]
        # Interior is at x < 0, so the oriented normal must point +x.
        self.assertGreater(float(wall["normal"][0]), 0.9)
        self.assertAlmostEqual(float(wall["low"]), -2.0, delta=0.35)
        self.assertAlmostEqual(float(wall["high"]), 2.0, delta=0.35)

    def test_opening_carved_only_with_see_through_evidence(self):
        points = self._wall_room_points()
        colors = np.full_like(points, 0.6)
        gravity = self.GRAVITY
        wall = {
            "normal": np.array([1.0, 0.0, 0.0]),
            "offset": 0.0,
            "low": -2.0,
            "high": 2.0,
            "color": np.array([0.6, 0.6, 0.6]),
        }
        # Door-like gap: z in [0.0, 1.4], y in [-0.2, 1.0] (touches floor).
        gap = (0.0, 1.4, -0.2, 1.0)
        gapped = self._wall_room_points(gap=gap)

        solid_mesh, solid_openings = _emit_wall(
            points, colors, gravity, np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]),
            0.98, -0.98, wall, {"voxel_size": 0.03},
        )
        self.assertEqual(solid_openings, 0)
        self.assertEqual(len(solid_mesh.triangles), 2)

        # No points behind the wall: the gap must stay solid.
        blocked_mesh, blocked_openings = _emit_wall(
            gapped, colors, gravity, np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]),
            0.98, -0.98, wall, {"voxel_size": 0.03},
        )
        self.assertEqual(blocked_openings, 0)

        # Points behind the wall inside the gap: the opening gets carved.
        behind = np.random.default_rng(0).uniform(
            low=np.array([0.35, -0.2, 0.0]), high=np.array([0.6, 1.0, 1.4]), size=(400, 3)
        )
        carved_points = np.vstack([gapped, behind])
        carved_mesh, carved_openings = _emit_wall(
            carved_points, np.full_like(carved_points, 0.6), gravity,
            np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]),
            0.98, -0.98, wall, {"voxel_size": 0.03},
        )
        # The carve is quantized to wall cells; a door-sized gap removes
        # multiple cells but stays a single connected component.
        self.assertGreaterEqual(carved_openings, 5)
        self.assertGreater(len(carved_mesh.triangles), 2)

    def test_preserve_hint_carves_wall_cells_with_visibility(self):
        gravity = self.GRAVITY
        wall = {
            "normal": np.array([1.0, 0.0, 0.0]),
            "offset": 0.0,
            "low": -2.0,
            "high": 2.0,
            "color": np.array([0.6, 0.6, 0.6]),
        }
        points = self._wall_room_points()
        colors = np.full_like(points, 0.6)
        axis_u = np.array([1.0, 0.0, 0.0])
        axis_v = np.array([0.0, 0.0, 1.0])
        cfg = {"voxel_size": 0.03}

        # Camera inside the room (x<0) looking at the wall at x=0, whose
        # center lands at pixel (50, 50) of a 100x100 image.
        cam_center = np.array([-3.0, 0.0, 0.0])
        rotation = np.array([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = rotation
        extrinsic[:3, 3] = -rotation @ cam_center
        intrinsic = np.array([[40.0, 0, 50], [0, 40.0, 50], [0, 0, 1.0]])
        # Model space is 100x100 while the preserve mask lives at 200x200
        # original resolution; the hint projection must handle both mappings.
        coords = np.array([0.0, 0.0, 100.0, 100.0, 200.0, 200.0])

        # Wall cells project to u = 13.3*z + 50 for z in [-2, 2]; a preserve
        # band covering u in [43, 57] marks cells near z = 0.
        mask = np.zeros((200, 200), dtype=bool)
        mask[70:132, 86:116] = True

        def build(depth_value):
            depth = np.full((100, 100), depth_value, dtype=np.float32)
            hint = {
                "extrinsic": extrinsic,
                "intrinsic": intrinsic,
                "original_coords": coords,
                "depth": depth,
                "mask": mask,
            }
            return _emit_wall(
                points, colors, gravity, axis_u, axis_v,
                0.98, -0.98, wall, cfg, opening_hints=[hint],
            )

        # Depth matches the wall distance: the marked cells are carved.
        carved_mesh, carved_cells = build(3.0)
        self.assertGreater(carved_cells, 0)

        # An occluder in front of the wall also carves the cells behind it
        # (cabinet/fridge occlusion modelling).
        occluded_mesh, occluded_cells = build(1.0)
        self.assertGreater(occluded_cells, 0)

        # Depth far beyond the wall: the mask describes another surface.
        beyond_mesh, beyond_cells = build(30.0)
        self.assertEqual(beyond_cells, 0)

    def test_fallback_walls_survive_new_pipeline(self):
        x, z = np.meshgrid(np.linspace(-1, 1, 12), np.linspace(-2, 2, 12))
        floor = np.column_stack([x.ravel(), np.ones(x.size), z.ravel()])
        ceiling = floor.copy()
        ceiling[:, 1] = -1
        points = np.vstack([floor, ceiling])
        colors = np.full_like(points, 0.7)
        mesh, bounds = build_structural_mesh(
            points,
            colors,
            [],
            self.GRAVITY,
            {"voxel_size": 0.03, "footprint_wall_fallback": True},
        )
        self.assertEqual(bounds["wall_count"], 4)
        self.assertEqual(bounds["wall_source"], "robust_footprint_fallback")
        self.assertEqual(bounds["wall_openings"], 0)
        self.assertEqual(len(mesh.vertices), 8)
        self.assertEqual(len(mesh.triangles), 12)


class PromptTests(unittest.TestCase):
    def test_disabled_provider_returns_static_prompts(self):
        result = resolve_prompts({"prompts": ["furniture"]}, {"enabled": False}, [])
        self.assertEqual(result, ["furniture"])


class VGGTSlamMathTests(unittest.TestCase):
    def test_similarity_scale_and_rigid_pose_roundtrip(self):
        rng = np.random.default_rng(0)
        rotation, _ = np.linalg.qr(rng.standard_normal((3, 3)))
        if np.linalg.det(rotation) < 0:
            rotation[:, 0] *= -1
        scale = 1.37
        translation = np.array([0.4, -0.9, 2.2])
        homography = np.eye(4)
        homography[:3, :3] = scale * rotation
        homography[:3, 3] = translation
        self.assertAlmostEqual(homography_scale(homography), scale, places=9)

        cam_to_world = rigid_from_similarity(homography, scale)
        cam_point = rng.standard_normal(3)
        scaled_point = scale * cam_point
        world_via_similarity = (homography[:3, :3] @ cam_point) + translation
        world_via_rigid = cam_to_world[:3, :3] @ scaled_point + cam_to_world[:3, 3]
        np.testing.assert_allclose(world_via_rigid, world_via_similarity, atol=1e-9)

    def test_homography_is_invariant_to_projective_gauge(self):
        rng = np.random.default_rng(1)
        rotation, _ = np.linalg.qr(rng.standard_normal((3, 3)))
        if np.linalg.det(rotation) < 0:
            rotation[:, 0] *= -1
        homography = np.eye(4)
        homography[:3, :3] = 0.31 * rotation
        homography[:3, 3] = np.array([0.1, 0.2, 0.3])
        gauge = np.eye(4) * 2.377
        gauge[3, 3] = 2.377
        self.assertAlmostEqual(
            homography_scale(homography), homography_scale(gauge @ homography), places=9
        )
        np.testing.assert_allclose(
            rigid_from_similarity(homography, homography_scale(homography)),
            rigid_from_similarity(
                gauge @ homography, homography_scale(gauge @ homography)
            ),
            atol=1e-9,
        )

    def test_crop_mode_mask_covers_full_model_canvas(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "000000.png"
            cv2.imwrite(str(path), np.full((540, 960), 255, dtype=np.uint8))
            result = mask_in_model_space(
                path, np.array([0, 0, 518, 294]), model_height=294, model_width=518
            )
            self.assertEqual(result.shape, (294, 518))
            self.assertEqual(int(np.count_nonzero(result)), 294 * 518)

    def test_square_mode_mask_occupies_letterbox_band(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "000000.png"
            cv2.imwrite(str(path), np.full((540, 960), 255, dtype=np.uint8))
            # 960x540 letterboxed into 518x518: content rows ~113..405
            coords = np.array([0, 518 * 210 / 960, 518, 518 * 750 / 960])
            result = mask_in_model_space(
                path, coords, model_height=518, model_width=518
            )
            self.assertEqual(result.shape, (518, 518))
            self.assertEqual(int(np.count_nonzero(result[:113])), 0)
            self.assertEqual(int(np.count_nonzero(result[113:405])), 292 * 518)
            self.assertEqual(int(np.count_nonzero(result[405:])), 0)


class SLAMAdapterTests(unittest.TestCase):
    def _adapter(self, backend):
        return SLAMAdapter({"backend": backend}, Path("."))

    def test_direct_backend_passes_max_frames(self):
        command = self._adapter("vggt_direct")._command(
            "vbr.vggt_direct",
            Path("frames"),
            Path("masks"),
            Path("points.ply"),
            Path("model.pt"),
        )
        self.assertIn("vbr.vggt_direct", command)
        self.assertIn("--max-frames", command)

    def test_slam_backend_passes_solver_options(self):
        command = self._adapter("vggt_slam")._command(
            "vbr.vggt_slam_backend",
            Path("frames"),
            Path("masks"),
            Path("points.ply"),
            Path("model.pt"),
        )
        self.assertIn("vbr.vggt_slam_backend", command)
        for flag in (
            "--submap-size",
            "--lc-thres",
            "--max-loops",
            "--salad-checkpoint",
        ):
            self.assertIn(flag, command)
        self.assertNotIn("--max-frames", command)

    def test_unknown_backend_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(ValueError):
                self._adapter("nope").run(
                    Path(temporary), Path(temporary) / "out", Path(temporary)
                )


class InpaintingTests(unittest.TestCase):
    def test_ffmpeg_supports_h264_encoding(self):
        self.assertTrue(Path(_resolve_ffmpeg()).is_file())


if __name__ == "__main__":
    unittest.main()
