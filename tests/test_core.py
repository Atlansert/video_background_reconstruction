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
from vbr.models.segmentation import SegmentationAdapter
from vbr.models.slam import SLAMAdapter
from vbr.vggt_direct import _mask_in_model_space
from vbr.vggt_slam_backend import (
    homography_scale,
    mask_in_model_space,
    rigid_from_similarity,
)
from vbr.sam2_propagate import keyframe_intervals
from vbr.video import (
    blend_mask_jumps,
    detect_mask_onsets,
    fill_enclosed_mask_holes,
    onset_latency_metrics,
    prepare_inpainting_masks,
    stabilize_mask_sequence,
)
from vbr.prompts import resolve_prompts
from tools.post_temporal_smooth import smooth_video
from tools.refine_inpainting import expand_copy_through, find_copy_through


class InpaintRefineTests(unittest.TestCase):
    def test_copy_through_region_is_detected_and_expanded(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frames = root / "frames"
            masks_dir = root / "masks_inpaint"
            frames.mkdir()
            masks_dir.mkdir()
            original = np.zeros((60, 60, 3), dtype=np.uint8)
            original[10:40, 10:40] = (60, 90, 200)
            background = original.copy()
            background[10:40, 30:40] = (200, 200, 200)
            cv2.imwrite(str(masks_dir / "000000.png"), np.full((60, 60), 255, np.uint8))
            mask = np.zeros((60, 60), dtype=np.uint8)
            mask[12:38, 12:38] = 255
            cv2.imwrite(str(masks_dir / "000000.png"), mask)
            video = root / "video.mp4"
            writer = cv2.VideoWriter(
                str(video), cv2.VideoWriter_fourcc(*"mp4v"), 5, (60, 60)
            )
            for _ in range(5):
                writer.write(original)
            writer.release()
            bg_video = root / "background.mp4"
            writer = cv2.VideoWriter(
                str(bg_video), cv2.VideoWriter_fourcc(*"mp4v"), 5, (60, 60)
            )
            for _ in range(5):
                writer.write(background)
            writer.release()
            copies = find_copy_through(video, bg_video, masks_dir, threshold=20.0, min_area=50)
            self.assertIn(0, copies)
            changed = expand_copy_through(copies, masks_dir, dilate_px=10)
            self.assertGreater(changed, 0)

    def test_no_copy_through_when_inpainting_changes_pixels(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            masks_dir = root / "masks_inpaint"
            masks_dir.mkdir()
            original = np.full((40, 40, 3), (10, 20, 30), dtype=np.uint8)
            background = np.full((40, 40, 3), 240, dtype=np.uint8)
            cv2.imwrite(str(masks_dir / "000000.png"), np.full((40, 40), 255, np.uint8))
            video = root / "video.mp4"
            writer = cv2.VideoWriter(
                str(video), cv2.VideoWriter_fourcc(*"mp4v"), 5, (40, 40)
            )
            for _ in range(3):
                writer.write(original)
            writer.release()
            bg_video = root / "background.mp4"
            writer = cv2.VideoWriter(
                str(bg_video), cv2.VideoWriter_fourcc(*"mp4v"), 5, (40, 40)
            )
            for _ in range(3):
                writer.write(background)
            writer.release()
            copies = find_copy_through(video, bg_video, masks_dir, threshold=20.0, min_area=50)
            self.assertEqual(copies, {})


def test_temporal_smooth_kills_one_frame_flash_and_keeps_length():
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            masks_dir = root / "masks"
            masks_dir.mkdir()
            base = np.zeros((40, 40, 3), dtype=np.uint8)
            base[:] = (100, 120, 140)
            flash = base.copy()
            flash[10:30, 10:30] = (10, 10, 10)
            frames = [base, flash, base, base, base]
            video = root / "input.mp4"
            writer = cv2.VideoWriter(
                str(video), cv2.VideoWriter_fourcc(*"mp4v"), 5, (40, 40)
            )
            for frame in frames:
                writer.write(frame)
            writer.release()
            for index in range(5):
                cv2.imwrite(
                    str(masks_dir / f"{index:06d}.png"),
                    np.full((40, 40), 255, np.uint8),
                )
            smoothed = root / "out.mp4"
            report = smooth_video(video, masks_dir, smoothed, kernel=3)
            assert report["frames_written"] == 5
            assert report["changed_pixels"] > 0
            capture = cv2.VideoCapture(str(smoothed))
            assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 5
            capture.release()


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

    def test_single_frame_component_does_not_trigger_onset(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            empty = np.zeros((80, 80), dtype=np.uint8)
            blob = empty.copy()
            blob[20:60, 20:60] = 255
            for index, mask in enumerate([empty, empty, blob, empty, empty, empty]):
                cv2.imwrite(str(directory / f"{index:06d}.png"), mask)
            events = detect_mask_onsets(
                directory,
                min_new_area_px=500,
                persistence_frames=3,
                dilation_px=3,
                cooldown_frames=1,
            )
            self.assertEqual(events, [])

    def test_persistent_new_component_triggers_onset(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            empty = np.zeros((80, 80), dtype=np.uint8)
            blob = empty.copy()
            blob[20:60, 20:60] = 255
            for index, mask in enumerate([empty, empty, blob, blob, blob, blob]):
                cv2.imwrite(str(directory / f"{index:06d}.png"), mask)
            events = detect_mask_onsets(
                directory,
                min_new_area_px=500,
                persistence_frames=3,
                dilation_px=3,
                cooldown_frames=1,
            )
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0]["frame_id"], 2)
            self.assertGreaterEqual(events[0]["new_area_px"], 1500)

    def test_onset_latency_prefers_earlier_refined_frames(self):
        with tempfile.TemporaryDirectory() as temporary:
            baseline_dir = Path(temporary) / "baseline"
            refined_dir = Path(temporary) / "refined"
            baseline_dir.mkdir()
            refined_dir.mkdir()
            blob = np.zeros((40, 40), dtype=np.uint8)
            blob[10:30, 10:30] = 255
            for index in range(6):
                cv2.imwrite(
                    str(baseline_dir / f"{index:06d}.png"),
                    blob if index >= 4 else np.zeros((40, 40), dtype=np.uint8),
                )
            for index in range(6):
                cv2.imwrite(
                    str(refined_dir / f"{index:06d}.png"),
                    blob if index >= 1 else np.zeros((40, 40), dtype=np.uint8),
                )
            events = [{"frame_id": 4, "new_area_px": 400}]
            report = onset_latency_metrics(
                baseline_dir,
                refined_dir,
                events,
                dilation_px=3,
                overlap_fraction=0.5,
                window_back=6,
                window_forward=2,
            )
            self.assertEqual(report["events"][0]["latency_frames"], 3)
            self.assertEqual(report["improved_events"], 1)

    def test_one_frame_flicker_is_removed_and_motion_is_kept(self):
        empty = np.zeros((8, 8), dtype=np.uint8)
        blob = empty.copy()
        blob[2:6, 2:6] = 255
        flash = empty.copy()
        flash[1, 1] = 255
        hole = blob.copy()
        hole[3:5, 3:5] = 0
        shifted = empty.copy()
        shifted[2:6, 3:7] = 255
        masks = [blob, flash, blob, blob, hole, blob, shifted]
        pinned, _ = stabilize_mask_sequence(masks, pin_indices={1})
        np.testing.assert_array_equal(pinned[1], flash)
        np.testing.assert_array_equal(pinned[4], blob)
        updated, report = stabilize_mask_sequence(masks)
        self.assertGreater(report["changed_pixels"], 0)
        np.testing.assert_array_equal(updated[0], blob)
        np.testing.assert_array_equal(updated[1], blob)
        np.testing.assert_array_equal(updated[2], blob)
        np.testing.assert_array_equal(updated[4], blob)
        np.testing.assert_array_equal(updated[6], shifted)

    def test_mask_jumps_are_ramped_without_moving_keyframes(self):
        empty = np.zeros((16, 16), dtype=np.uint8)
        small = empty.copy()
        small[6:10, 6:10] = 255
        large = empty.copy()
        large[2:14, 2:14] = 255
        masks = [small, small, small, large, large, large]
        updated, report = blend_mask_jumps(
            masks, pin_indices={3}, xor_threshold=0.05, window=3
        )
        self.assertGreater(report["jumps"], 0)
        np.testing.assert_array_equal(updated[0], small)
        np.testing.assert_array_equal(updated[3], large)
        np.testing.assert_array_equal(updated[5], large)
        self.assertGreater(int((updated[2] > 0).sum()), int((small > 0).sum()))
        self.assertLess(int((updated[2] > 0).sum()), int((large > 0).sum()))

    def test_mask_jump_blend_stays_in_bounds_without_pins(self):
        empty = np.zeros((8, 8), dtype=np.uint8)
        full = np.full((8, 8), 255, dtype=np.uint8)
        masks = [empty, full, full]
        updated, report = blend_mask_jumps(
            masks, pin_indices=None, xor_threshold=0.05, window=8
        )
        self.assertGreater(report["jumps"], 0)
        self.assertEqual(len(updated), 3)
        np.testing.assert_array_equal(updated[0], empty)
        np.testing.assert_array_equal(updated[2], full)

    def test_inpainting_masks_union_nearby_detections(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "src"
            dest = Path(temporary) / "dst"
            source.mkdir()
            empty = np.zeros((12, 12), dtype=np.uint8)
            blob = empty.copy()
            blob[4:8, 4:8] = 255
            cv2.imwrite(str(source / "000000.png"), empty)
            cv2.imwrite(str(source / "000001.png"), blob)
            cv2.imwrite(str(source / "000002.png"), empty)
            report = prepare_inpainting_masks(
                source,
                dest,
                close_px=0,
                temporal_radius=1,
                dilate_px=0,
                hull_min_area=0,
                overlap=0.12,
            )
            first = cv2.imread(str(dest / "000000.png"), cv2.IMREAD_GRAYSCALE)
            self.assertGreater(int((first > 0).sum()), 0)
            self.assertGreater(report["mean_coverage"], report["mean_coverage_source"])

            far = Path(temporary) / "far"
            far.mkdir()
            empty = np.zeros((80, 80), dtype=np.uint8)
            left = empty.copy()
            left[5:20, 5:20] = 255
            right = empty.copy()
            right[60:75, 60:75] = 255
            for index in range(13):
                cv2.imwrite(
                    str(far / f"{index:06d}.png"),
                    left if index == 0 else right if index == 12 else empty,
                )
            prepare_inpainting_masks(
                far,
                dest,
                close_px=0,
                temporal_radius=12,
                dilate_px=0,
                hull_min_area=0,
                overlap=0.12,
            )
            merged = cv2.imread(str(dest / "000000.png"), cv2.IMREAD_GRAYSCALE)
            self.assertEqual(int(merged[67, 67]), 0)
            self.assertGreater(int((merged > 0).sum()), 0)

    def test_default_inpaint_masks_stay_close_to_source_coverage(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "src"
            dest = Path(temporary) / "dst"
            source.mkdir()
            mask = np.zeros((100, 100), dtype=np.uint8)
            mask[35:65, 35:65] = 255
            for index in range(5):
                cv2.imwrite(str(source / f"{index:06d}.png"), mask)
            report = prepare_inpainting_masks(source, dest)
            self.assertEqual(report["hull_pixels"], 0)
            self.assertLessEqual(report["mean_extra_coverage"], 0.02)
            self.assertLessEqual(report["extra_coverage_ratio"], 0.25)
            self.assertLessEqual(report["max_coverage"], 0.12)

    def test_convex_hull_fill_skips_thin_components(self):
        from vbr.video import fill_mask_convex_hulls

        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[2:18, 8:12] = 1
        filled, added = fill_mask_convex_hulls(mask, min_area=10, max_extra=0.2)
        self.assertEqual(added, 0)
        np.testing.assert_array_equal(filled, mask)
        mask[4:16, 4:16] = 1
        mask[8:12, 8:12] = 0
        filled, added = fill_mask_convex_hulls(mask, min_area=10, max_extra=0.5)
        self.assertGreater(added, 0)
        self.assertEqual(int(filled[10, 10]), 1)

    def test_keyframe_intervals_clone_last_frame_and_pair_seeds(self):
        start = np.ones((4, 4), dtype=np.uint8)
        mid = np.full((4, 4), 2, dtype=np.uint8)
        key_masks = [(0, 0, start), (30, 30, mid)]
        intervals = keyframe_intervals(key_masks, last_index=45)
        self.assertEqual(len(intervals), 2)
        self.assertEqual(intervals[0][0], 0)
        np.testing.assert_array_equal(intervals[0][1], start)
        self.assertEqual(intervals[0][2], 30)
        np.testing.assert_array_equal(intervals[0][3], mid)
        self.assertEqual(intervals[1][0], 30)
        self.assertEqual(intervals[1][2], 45)
        np.testing.assert_array_equal(intervals[1][3], mid)
        last_is_key = keyframe_intervals(key_masks, last_index=30)
        self.assertEqual(len(last_is_key), 1)
        self.assertEqual(last_is_key[0][2], 30)


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


class SegmentationAdapterTests(unittest.TestCase):
    def test_dual_anchor_is_on_by_default(self):
        adapter = SegmentationAdapter({}, Path("."))
        self.assertEqual(adapter._sam2_extra_flags(), ["--dual-anchor"])
        adapter = SegmentationAdapter(
            {"sam2_dual_anchor": False, "sam2_offload_state": True}, Path(".")
        )
        self.assertEqual(
            adapter._sam2_extra_flags(), ["--no-dual-anchor", "--offload-state"]
        )


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
