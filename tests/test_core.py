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

    def test_ear_clip_triangulates_l_shaped_polygon(self):
        import open3d as o3d

        from vbr.geometry import _ear_clip

        uv = np.asarray(
            [[0.0, 0.0], [3.0, 0.0], [3.0, 2.0], [2.0, 2.0], [2.0, 1.0], [0.0, 1.0]]
        )
        triangles = _ear_clip(uv)
        self.assertEqual(len(triangles), 4)
        area = 0.5 * abs(
            sum(
                uv[a][0] * uv[b][1] - uv[b][0] * uv[a][1]
                for a, b in ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0))
            )
        )
        triangle_area = sum(
            0.5
            * abs(
                uv[a][0] * (uv[b][1] - uv[c][1])
                + uv[b][0] * (uv[c][1] - uv[a][1])
                + uv[c][0] * (uv[a][1] - uv[b][1])
            )
            for a, b, c in triangles
        )
        self.assertAlmostEqual(area, triangle_area, places=5)

    def test_small_boundary_loop_is_filled_and_large_kept(self):
        import open3d as o3d

        from vbr.geometry import fill_small_boundary_holes, _boundary_loops

        # A quad of two triangles, one removed: a triangular 3-edge hole.
        vertices = o3d.utility.Vector3dVector(
            np.asarray(
                [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 2.0, 0.0], [0.0, 2.0, 0.0]]
            )
        )
        complete = o3d.geometry.TriangleMesh(
            vertices, o3d.utility.Vector3iVector(np.asarray([[0, 1, 2], [0, 2, 3]]))
        )
        punctured = o3d.geometry.TriangleMesh(
            vertices, o3d.utility.Vector3iVector(np.asarray([[0, 1, 2]]))
        )
        loops = _boundary_loops(punctured)
        self.assertEqual(len(loops), 1)
        self.assertEqual(len(loops[0]), 3)
        filled, count = fill_small_boundary_holes(punctured, max_loop_edges=10)
        self.assertEqual(count, 1)
        self.assertEqual(len(np.asarray(filled.triangles)), 2)
        self.assertEqual(len(_boundary_loops(filled)), 0)
        # A loop longer than the limit stays open.
        big = o3d.geometry.TriangleMesh(
            vertices, o3d.utility.Vector3iVector(np.asarray([[0, 1, 2], [0, 2, 3]]))
        )
        filled2, count2 = fill_small_boundary_holes(big, max_loop_edges=0)
        self.assertEqual(count2, 0)

    def test_clean_mesh_drops_floating_fragment(self):
        import open3d as o3d

        from vbr.geometry import clean_mesh

        box = o3d.geometry.TriangleMesh.create_box(2.0, 2.0, 2.0)
        fragment = o3d.geometry.TriangleMesh.create_tetrahedron().translate(
            [5.0, 5.0, 5.0]
        )
        combined = box + fragment
        cleaned, stats = clean_mesh(combined, {"mesh_min_component_triangles": 10})
        self.assertEqual(stats["components_before"], 2)
        self.assertEqual(stats["components_after"], 1)
        self.assertAlmostEqual(stats["largest_fraction"], 12 / 16, places=5)

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

    def test_sustained_low_coverage_window_is_detected(self):
        from vbr.video import detect_persistent_misses

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            full = np.full((40, 40), 255, dtype=np.uint8)
            empty = np.zeros((40, 40), dtype=np.uint8)
            for index in range(100):
                if 30 <= index <= 60:
                    cv2.imwrite(str(directory / f"{index:06d}.png"), empty)
                else:
                    cv2.imwrite(str(directory / f"{index:06d}.png"), full)
            windows = detect_persistent_misses(
                directory, 100, min_coverage=0.5, min_window_frames=20
            )
            self.assertEqual(len(windows), 1)
            self.assertEqual(windows[0]["start"], 30)
            self.assertEqual(windows[0]["end"], 60)
            self.assertEqual(windows[0]["frames"], 31)

    def test_short_dip_and_far_windows_do_not_merge(self):
        from vbr.video import detect_persistent_misses

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            full = np.full((40, 40), 255, dtype=np.uint8)
            empty = np.zeros((40, 40), dtype=np.uint8)
            for index in range(100):
                low = (10 <= index <= 9 + 3) or (50 <= index <= 59) or (80 <= index <= 89)
                cv2.imwrite(
                    str(directory / f"{index:06d}.png"), empty if low else full
                )
            windows = detect_persistent_misses(
                directory, 100, min_coverage=0.5, min_window_frames=5,
                merge_gap_frames=3,
            )
            starts = [window["start"] for window in windows]
            self.assertNotIn(10, starts)  # 3-frame dip stays below the threshold
            self.assertIn(50, starts)
            self.assertIn(80, starts)

    def test_missing_mask_counted_as_zero_coverage(self):
        from vbr.video import detect_persistent_misses

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            full = np.full((40, 40), 255, dtype=np.uint8)
            for index in range(40):
                if index in (20, 21, 22):
                    continue
                cv2.imwrite(str(directory / f"{index:06d}.png"), full)
            windows = detect_persistent_misses(
                directory, 40, min_coverage=0.5, min_window_frames=3
            )
            self.assertEqual(len(windows), 1)
            self.assertEqual(windows[0]["start"], 20)
            self.assertEqual(windows[0]["end"], 22)

    def test_miss_seed_requires_substantial_new_area(self):
        from vbr.video import select_miss_seeds

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            current = np.zeros((40, 40), dtype=np.uint8)
            current[10:20, 10:20] = 255
            for frame_id in (20, 21, 22):
                cv2.imwrite(str(directory / f"{frame_id:06d}.png"), current)
            duplicates = np.zeros((40, 40), dtype=np.uint8)
            duplicates[11:19, 11:19] = 255  # inside the current mask: no new area
            expanding = np.zeros((40, 40), dtype=np.uint8)
            expanding[10:20, 10:20] = 255
            expanding[10:20, 30:35] = 255  # 5x10 = 50 new pixels
            refinement = Path(temporary) / "refined"
            refinement.mkdir()
            cv2.imwrite(str(refinement / "000021.png"), duplicates)
            cv2.imwrite(str(refinement / "000022.png"), expanding)
            accepted, summary = select_miss_seeds(
                refinement, directory, min_added_area_px=40
            )
            self.assertEqual(accepted, [22])
            self.assertEqual(summary["rejected_frames"], 1)

    def test_miss_seed_rejects_collapsing_candidate(self):
        from vbr.video import select_miss_seeds

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            current = np.zeros((40, 40), dtype=np.uint8)
            current[10:30, 10:30] = 255
            cv2.imwrite(str(directory / "000010.png"), current)
            shrunk = np.zeros((40, 40), dtype=np.uint8)
            shrunk[13:27, 13:27] = 255  # 196 px < 0.5 * 400, far smaller
            refinement = Path(temporary) / "refined"
            refinement.mkdir()
            cv2.imwrite(str(refinement / "000010.png"), shrunk)
            accepted, summary = select_miss_seeds(
                refinement, directory, min_added_area_px=1, min_keep_fraction=0.5
            )
            self.assertEqual(accepted, [])
            self.assertEqual(summary["rejected_frames"], 1)

    def test_miss_seed_rejects_explosive_coverage_growth(self):
        from vbr.video import select_miss_seeds

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            current = np.zeros((40, 40), dtype=np.uint8)
            current[10:20, 10:20] = 255  # 100 px = 6.25% coverage
            cv2.imwrite(str(directory / "000030.png"), current)
            greedy = np.full((40, 40), 255, dtype=np.uint8)  # 100% coverage
            refinement = Path(temporary) / "refined"
            refinement.mkdir()
            cv2.imwrite(str(refinement / "000030.png"), greedy)
            accepted, summary = select_miss_seeds(
                refinement,
                directory,
                min_added_area_px=1,
                max_coverage_increase=0.15,
            )
            self.assertEqual(accepted, [])
            self.assertEqual(summary["rejected_frames"], 1)
            moderate = np.zeros((40, 40), dtype=np.uint8)
            moderate[10:20, 10:26] = 255  # 160 px = +3.75% coverage, within cap
            cv2.imwrite(str(refinement / "000031.png"), moderate)
            cv2.imwrite(str(directory / "000031.png"), current)
            accepted2, _ = select_miss_seeds(
                refinement,
                directory,
                min_added_area_px=1,
                max_coverage_increase=0.15,
            )
            self.assertEqual(accepted2, [31])

    def test_miss_seed_evidence_waives_coverage_cap(self):
        from vbr.video import select_miss_seeds

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            current = np.zeros((40, 40), dtype=np.uint8)
            current[10:20, 10:20] = 255
            cv2.imwrite(str(directory / "000040.png"), current)
            big = np.zeros((40, 40), dtype=np.uint8)
            big[10:20, 10:20] = 255
            big[10:36, 8:38] = 255  # far above the +15% coverage cap
            refinement = Path(temporary) / "refined"
            refinement.mkdir()
            cv2.imwrite(str(refinement / "000040.png"), big)
            residual = np.zeros((40, 40), dtype=np.uint8)
            residual[12:36, 8:38] = 255  # most of the new area is copy-through
            evidence = {40: residual > 0}
            accepted, summary = select_miss_seeds(
                refinement,
                directory,
                min_added_area_px=1,
                max_coverage_increase=0.15,
                evidence=evidence,
                evidence_min_overlap=0.5,
            )
            self.assertEqual(accepted, [40])
            self.assertEqual(summary["coverage_cap_waived"], 1)
            wrong_evidence = {40: np.zeros((40, 40), dtype=bool)}
            accepted2, _ = select_miss_seeds(
                refinement,
                directory,
                min_added_area_px=1,
                max_coverage_increase=0.15,
                evidence=wrong_evidence,
                evidence_min_overlap=0.5,
            )
            self.assertEqual(accepted2, [])

    def test_derive_box_seeds_tracks_mask_blob_per_subwindow(self):
        from vbr.video import derive_box_seeds

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for frame_id in range(60):
                mask = np.zeros((100, 100), dtype=np.uint8)
                mask[30:70, 40:80] = 255
                cv2.imwrite(str(directory / f"{frame_id:06d}.png"), mask)
            windows = [{"start": 0, "end": 59}]
            seeds = derive_box_seeds(
                directory, windows, subwindow_frames=30, expansion=0.4
            )
            self.assertEqual(len(seeds), 2)
            first = seeds[0]
            self.assertEqual(first["start"], 0)
            self.assertEqual(first["end"], 29)
            box = first["box"]
            # Blob x 40..80 on width 100 -> expanded [0.24, 0.14, 0.96, 0.86].
            self.assertAlmostEqual(box[0], 0.24, places=2)
            self.assertAlmostEqual(box[2], 0.96, places=2)
            self.assertAlmostEqual(box[1], 0.14, places=2)


class TemporalSmoothTests(unittest.TestCase):
    def test_warp_flow_applies_known_displacement(self):
        from vbr.temporal_smooth import warp_flow

        rng = np.random.default_rng(0)
        texture = rng.integers(0, 255, (30, 40, 3), dtype=np.uint8)
        flow = np.zeros((30, 40, 2), dtype=np.float32)
        flow[..., 0] = 4.0
        warped = warp_flow(texture, flow)
        # Sampling at q + (4,0) keeps columns 4..end, replicate-filling the rim.
        np.testing.assert_array_equal(warped[:, :-4], texture[:, 4:])
        np.testing.assert_array_equal(
            warped[:, -4:], np.tile(texture[:, -1:], (1, 4, 1))
        )

    def test_aligned_median_removes_camera_motion_ghost(self):
        from vbr.temporal_smooth import aligned_median

        rng = np.random.default_rng(1)
        texture = rng.integers(0, 255, (40, 60, 3), dtype=np.uint8).astype(np.uint8)
        frames = [np.roll(texture, step * 3, axis=1) for step in range(3)]
        target = frames[1].copy()
        target[20:30, 24:34] = rng.integers(0, 255, (10, 10, 3), dtype=np.uint8)
        mask = np.zeros((40, 60), dtype=bool)
        mask[20:30, 24:34] = True
        back = np.zeros((40, 60, 2), dtype=np.float32)
        back[..., 0] = -3.0
        forward = np.zeros((40, 60, 2), dtype=np.float32)
        forward[..., 0] = 3.0
        smoothed, used = aligned_median(target, [frames[0], frames[2]], [back, forward], mask)
        self.assertEqual(used, 3)
        # The corrupted blob is replaced by the true background of frame 1.
        np.testing.assert_array_equal(smoothed[mask], frames[1][mask])
        # The naive median (no alignment) ghosts instead of restoring.
        naive_stack = np.stack([frames[0], target, frames[2]], axis=0)
        naive = np.median(naive_stack, axis=0).astype(np.uint8)
        self.assertFalse(np.array_equal(naive[mask], frames[1][mask]))

    def test_aligned_median_keeps_target_outside_mask(self):
        from vbr.temporal_smooth import aligned_median

        rng = np.random.default_rng(2)
        target = rng.integers(0, 255, (20, 20, 3), dtype=np.uint8)
        neighbor = target.copy()
        zero_flow = np.zeros((20, 20, 2), dtype=np.float32)
        mask = np.zeros((20, 20), dtype=bool)
        mask[:5, :] = True
        smoothed, used = aligned_median(target, [neighbor], [zero_flow], mask)
        self.assertEqual(used, 2)
        np.testing.assert_array_equal(smoothed[5:, :], target[5:, :])


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

    def test_chunk_ranges_cover_full_video_with_overlap(self):
        from vbr.models.inpainting import chunk_ranges

        ranges = list(chunk_ranges(1000, 240, 60))
        self.assertEqual(ranges[0][0], 0)
        self.assertEqual(ranges[-1][1], 1000)
        for index in range(1, len(ranges) - 1):
            self.assertEqual(ranges[index][0], ranges[index - 1][1] - 60)
            self.assertEqual(ranges[index][1] - ranges[index][0], 240)
        self.assertEqual(ranges[-1][0], ranges[-2][1] - 60)
        self.assertEqual(ranges[-1][1] - ranges[-1][0], 100)
        covered = set()
        for start, end in ranges:
            covered.update(range(start, end))
        self.assertEqual(covered, set(range(1000)))

    def test_chunk_ranges_reject_invalid_params(self):
        from vbr.models.inpainting import chunk_ranges

        with self.assertRaises(ValueError):
            list(chunk_ranges(1000, 100, 100))

    def test_stitch_prefers_later_chunk_in_overlap(self):
        import cv2 as cv2_module

        from vbr.models.inpainting import _stitch_chunks

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            def write_chunk(name, values):
                path = root / name
                writer = cv2_module.VideoWriter(
                    str(path),
                    cv2_module.VideoWriter_fourcc(*"mp4v"),
                    5,
                    (16, 16),
                )
                for value in values:
                    writer.write(np.full((16, 16, 3), value, dtype=np.uint8))
                writer.release()
                return path

            first = write_chunk("first.mp4", [25] * 8)   # globals 0..7
            second = write_chunk("second.mp4", [240] * 8)  # globals 4..11
            stitched_path = root / "stitched.mp4"
            total = _stitch_chunks(
                [(0, 8, first), (4, 12, second)], stitched_path, 5.0
            )
            self.assertEqual(total, 12)
            capture = cv2_module.VideoCapture(str(stitched_path))
            frames = []
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                frames.append(int(frame[0, 0, 0]))
            capture.release()
            # mp4v is lossy, so classify each decoded frame by proximity:
            # globals 0..3 come from chunk 1 (dark), 4..11 from chunk 2 (bright).
            for index, value in enumerate(frames):
                expected_dark = index < 4
                if expected_dark:
                    self.assertLess(value, 128)
                else:
                    self.assertGreater(value, 128)

    def test_reindexed_copy_wipes_stale_target_files(self):
        from vbr.models.inpainting import _reindexed_copy

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            cv2.imwrite(str(source / "000000.png"), np.full((8, 8), 0, np.uint8))
            cv2.imwrite(str(source / "000001.png"), np.full((8, 8), 255, np.uint8))
            target = root / "target"
            target.mkdir()
            cv2.imwrite(str(target / "000000.png"), np.full((8, 8), 128, np.uint8))
            cv2.imwrite(str(target / "000002.png"), np.full((8, 8), 128, np.uint8))
            _reindexed_copy(source, target, 0, 2, ".png")
            remaining = sorted(p.stem for p in target.glob("*.png"))
            self.assertEqual(remaining, ["000000", "000001"])  # stale 000002 gone
            copied = cv2.imread(str(target / "000000.png"), cv2.IMREAD_GRAYSCALE)
            self.assertEqual(int(copied[0, 0]), 0)  # fresh content, not stale 128

    def test_frame_cache_invalidates_on_source_marker_mismatch(self):
        import cv2 as cv2_module

        from vbr.cli import _ensure_frames
        from vbr.video import video_info

        def write_video(path, value, frames=5):
            writer = cv2_module.VideoWriter(
                str(path), cv2_module.VideoWriter_fourcc(*"mp4v"), 5, (40, 40)
            )
            for _ in range(frames):
                writer.write(np.full((40, 40, 3), value, dtype=np.uint8))
            writer.release()

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = root / "video.mp4"
            write_video(video, 30)
            info = video_info(video)
            all_frames = root / "frames_all"
            keyframes = root / "frames_keyframes"
            keyframe_ids = _ensure_frames(video, info, all_frames, keyframes, 2)
            self.assertEqual(keyframe_ids, [0, 2, 4])
            first = cv2_module.imread(str(all_frames / "000000.jpg"))
            self.assertLess(int(first[0, 0, 0]), 128)
            # Same frame count, different content: the marker must force a
            # re-extraction instead of reusing the cached frames.
            write_video(video, 220)
            fresh_info = video_info(video)
            _ensure_frames(video, fresh_info, all_frames, keyframes, 2)
            reextracted = cv2_module.imread(str(all_frames / "000000.jpg"))
            self.assertGreater(int(reextracted[0, 0, 0]), 128)

    def test_frame_dirs_contain_only_images(self):
        import cv2 as cv2_module

        from vbr.cli import _ensure_frames
        from vbr.video import video_info

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = root / "video.mp4"
            writer = cv2_module.VideoWriter(
                str(video), cv2_module.VideoWriter_fourcc(*"mp4v"), 5, (40, 40)
            )
            for _ in range(5):
                writer.write(np.full((40, 40, 3), 60, dtype=np.uint8))
            writer.release()
            info = video_info(video)
            all_frames = root / "frames_all"
            keyframes = root / "frames_keyframes"
            _ensure_frames(video, info, all_frames, keyframes, 2)
            # ProPainter's reader imreads every file in the frames dir, so
            # anything that is not a JPEG (JSON markers, reports) crashes it.
            leftovers = [p.name for p in all_frames.iterdir() if p.suffix != ".jpg"]
            self.assertEqual(leftovers, [])


class SVORTrialTests(unittest.TestCase):
    def test_svor_chunk_ranges_keep_4k_plus_1_lengths(self):
        from vbr.models.svor import svor_chunk_ranges

        ranges = list(svor_chunk_ranges(200, 77, 20))
        for start, end in ranges:
            self.assertEqual((end - start) % 4, 1)
        covered = set()
        for index, (start, end) in enumerate(ranges):
            covered.update(range(start, end))
            if index:
                self.assertEqual(start, ranges[index - 1][1] - 20)
        self.assertEqual(covered, set(range(200)))

    def test_svor_chunk_ranges_reject_invalid(self):
        from vbr.models.svor import svor_chunk_ranges

        with self.assertRaises(ValueError):
            list(svor_chunk_ranges(100, 50, 50))

    def test_svor_chunk_ranges_tail_stays_aligned_without_looping(self):
        from vbr.models.svor import svor_chunk_ranges

        # total=191 stepped backwards forever in the previous implementation
        for total in (191, 1801):  # 1801 = real run padded up from 1799
            ranges = list(svor_chunk_ranges(total, 77, 20))
            covered = set()
            for start, end in ranges:
                self.assertEqual((end - start) % 4, 1)
                covered.update(range(start, end))
            self.assertEqual(covered, set(range(total)))
        # the padded tail keeps at least `overlap` frames shared with the
        # previous chunk so the cross-fade still has material to blend
        ranges = list(svor_chunk_ranges(1801, 77, 20))
        self.assertGreaterEqual(ranges[-2][1] - ranges[-1][0], 20)

    def test_blend_stitch_crossfades_overlap(self):
        import cv2 as cv2_module

        from vbr.models.svor import blend_stitch

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            def write_chunk(name, start_value):
                path = root / name
                writer = cv2_module.VideoWriter(
                    str(path), cv2_module.VideoWriter_fourcc(*"mp4v"), 5, (16, 16)
                )
                for _ in range(10):
                    writer.write(np.full((16, 16, 3), start_value, dtype=np.uint8))
                writer.release()
                return path

            first = write_chunk("first.mp4", 20)   # globals 0..9
            second = write_chunk("second.mp4", 240)  # globals 6..15, overlap 6..9
            stitched_path = root / "stitched.mp4"
            total = blend_stitch(
                [(0, 10, first), (6, 16, second)], stitched_path, 5.0, fade_frames=4
            )
            self.assertEqual(total, 16)
            capture = cv2_module.VideoCapture(str(stitched_path))
            values = []
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                values.append(int(frame[0, 0, 0]))
            capture.release()
            # Pure first-chunk frames stay dark, pure later-chunk stay bright,
            # and the fade zone rises monotonically between them.
            self.assertLess(values[0], 60)
            self.assertGreater(values[-1], 200)
            fade_zone = values[6:10]
            self.assertTrue(
                all(
                    later >= earlier
                    for earlier, later in zip(fade_zone, fade_zone[1:])
                ),
                f"fade zone not monotonic: {fade_zone}",
            )


if __name__ == "__main__":
    unittest.main()
