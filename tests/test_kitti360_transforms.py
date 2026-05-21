import tempfile
import unittest
from pathlib import Path

import torch

from utils.geometry import (
    compose_camera_to_satellite_transform,
    get_world_to_satellite_transform,
    invert_se3,
    load_cam0_to_world_pose,
    load_indexed_pose_txt,
    load_imu_to_world_pose,
    load_kitti360_cam_to_pose_calib,
    nearest_pose,
)


class Kitti360TransformsTest(unittest.TestCase):
    def test_load_kitti360_cam_to_pose_calib_parses_4x4_matrices(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            calib_path = Path(tmpdir) / "calib_cam_to_pose.txt"
            calib_path.write_text(
                "image_00: 1 0 0 1 0 1 0 2 0 0 1 3\n"
                "image_02: 0 -1 0 4 1 0 0 5 0 0 1 6\n"
            )

            data = load_kitti360_cam_to_pose_calib(calib_path)

        self.assertEqual(set(data.keys()), {"image_00", "image_02"})
        self.assertTrue(torch.allclose(data["image_00"], torch.tensor([
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 2.0],
            [0.0, 0.0, 1.0, 3.0],
            [0.0, 0.0, 0.0, 1.0],
        ])))

    def test_indexed_pose_loading_and_nearest_previous(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            poses_path = Path(tmpdir) / "poses.txt"
            poses_path.write_text(
                "0 1 0 0 0 0 1 0 0 0 0 1 0\n"
                "5 1 0 0 5 0 1 0 0 0 0 1 0\n"
            )

            poses = load_indexed_pose_txt(poses_path, mat_size=12)
            nearest = nearest_pose(7, poses)
            direct = load_imu_to_world_pose(poses_path, frame_id=5)
            fallback = load_imu_to_world_pose(poses_path, frame_id=7)

        self.assertEqual(set(poses.keys()), {0, 5})
        self.assertIsNotNone(nearest)
        self.assertTrue(torch.allclose(nearest, poses[5]))
        self.assertTrue(torch.allclose(direct, poses[5]))
        self.assertTrue(torch.allclose(fallback, poses[5]))

    def test_cam0_to_world_loader_parses_4x4_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cam0_to_world.txt"
            path.write_text(
                "3 1 0 0 7 0 1 0 8 0 0 1 9 0 0 0 1\n"
            )

            poses = load_cam0_to_world_pose(path)

        self.assertEqual(set(poses.keys()), {3})
        self.assertTrue(torch.allclose(poses[3], torch.tensor([
            [1.0, 0.0, 0.0, 7.0],
            [0.0, 1.0, 0.0, 8.0],
            [0.0, 0.0, 1.0, 9.0],
            [0.0, 0.0, 0.0, 1.0],
        ])))

    def test_world_to_satellite_transform_is_north_up(self) -> None:
        T = get_world_to_satellite_transform(sat_size=512, resolution_m_per_px=0.2)

        east = T @ torch.tensor([1.0, 0.0, 0.0, 1.0])
        north = T @ torch.tensor([0.0, 1.0, 0.0, 1.0])

        self.assertAlmostEqual(east[0].item(), 261.0, places=5)
        self.assertAlmostEqual(east[1].item(), 256.0, places=5)
        self.assertAlmostEqual(north[0].item(), 256.0, places=5)
        self.assertAlmostEqual(north[1].item(), 251.0, places=5)

    def test_camera_to_satellite_chain_matches_matrix_product(self) -> None:
        T_cam_to_pose = torch.eye(4)
        T_cam_to_pose[:3, 3] = torch.tensor([2.0, 3.0, 4.0])
        T_imu_to_world = torch.eye(4)
        T_imu_to_world[:3, 3] = torch.tensor([10.0, 20.0, 30.0])

        T_cam_to_sat, T_world_to_sat = compose_camera_to_satellite_transform(
            T_cam_to_pose,
            T_imu_to_world,
            sat_size=512,
            resolution_m_per_px=0.2,
        )

        expected = T_world_to_sat @ (T_imu_to_world @ T_cam_to_pose)
        self.assertTrue(torch.allclose(T_cam_to_sat, expected))

    def test_invert_se3_round_trips(self) -> None:
        T = torch.eye(4)
        T[:3, :3] = torch.tensor([
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        T[:3, 3] = torch.tensor([1.0, 2.0, 3.0])

        T_inv = invert_se3(T)
        self.assertTrue(torch.allclose(T @ T_inv, torch.eye(4), atol=1e-6))
        self.assertTrue(torch.allclose(T_inv @ T, torch.eye(4), atol=1e-6))


if __name__ == "__main__":
    unittest.main()
