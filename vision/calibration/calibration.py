"""
Camera Calibration & BEV Homography Module for YUTA.

Derived from TrackStudio (Apache 2.0) camera calibration system.
Provides 4-point perspective transformation between camera pixel coordinates
and Bird's Eye View (BEV) ground plane coordinates.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

logger = logging.getLogger(__name__)


class CameraCalibration:
    """
    Handles camera perspective to Bird's Eye View (BEV) calibration and projections.
    """

    def __init__(self, calibration_file: Optional[str] = None):
        self.calibration_file = Path(calibration_file).resolve() if calibration_file else None
        self.homography_matrices: Dict[str, np.ndarray] = {}
        self.inv_homography_matrices: Dict[str, np.ndarray] = {}

        if self.calibration_file and self.calibration_file.exists():
            self.load_calibration()
        else:
            self._initialize_default_homographies()

    def _initialize_default_homographies(self):
        """Initializes default homography transformations for standard street cameras."""
        # Scale image (e.g. 1920x1080) to normalized BEV coordinate grid (0 to 1000)
        default_matrix = np.array([
            [0.5, 0.0, 50.0],
            [0.0, 0.8, 50.0],
            [0.0, 0.0008, 1.0]
        ], dtype=np.float32)
        self.homography_matrices["default"] = default_matrix
        self.inv_homography_matrices["default"] = np.linalg.inv(default_matrix)

    def compute_homography(
        self,
        image_points: List[Tuple[float, float]],
        bev_points: List[Tuple[float, float]],
    ) -> np.ndarray:
        """
        Compute 3x3 homography matrix from 4 point correspondences.
        Points format: [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
        """
        if len(image_points) != 4 or len(bev_points) != 4:
            raise ValueError("Exactly 4 pairs of points are required to compute homography.")

        src = np.array(image_points, dtype=np.float32)
        dst = np.array(bev_points, dtype=np.float32)

        # Standard direct linear transformation (DLT) for 4 points
        # Build matrix A (8x9) such that A * h = 0
        A = []
        for i in range(4):
            x, y = src[i, 0], src[i, 1]
            u, v = dst[i, 0], dst[i, 1]
            A.append([-x, -y, -1, 0, 0, 0, x * u, y * u, u])
            A.append([0, 0, 0, -x, -y, -1, x * v, y * v, v])
        A = np.array(A, dtype=np.float32)

        # Solve via SVD
        _, _, Vh = np.linalg.svd(A)
        H = Vh[-1].reshape(3, 3)
        if abs(H[2, 2]) > 1e-8:
            H = H / H[2, 2]

        return H

    def register_camera(
        self,
        camera_id: str,
        image_points: List[Tuple[float, float]],
        bev_points: List[Tuple[float, float]],
    ) -> np.ndarray:
        """
        Calibrates and registers a camera with given 4-point ground correspondences.
        """
        H = self.compute_homography(image_points, bev_points)
        self.homography_matrices[camera_id] = H
        self.inv_homography_matrices[camera_id] = np.linalg.inv(H)
        logger.info(f"Registered BEV homography for camera {camera_id}")
        return H

    def project_to_bev(self, camera_id: str, x: float, y: float) -> Tuple[float, float]:
        """
        Projects a point (x, y) in camera image plane to BEV ground coordinates.
        For vehicles, (x, y) should typically be the bottom-center of the bounding box.
        """
        H = self.homography_matrices.get(camera_id, self.homography_matrices.get("default"))
        if H is None:
            return float(x), float(y)

        pt = np.array([x, y, 1.0], dtype=np.float32)
        projected = H @ pt
        if abs(projected[2]) > 1e-8:
            return float(projected[0] / projected[2]), float(projected[1] / projected[2])
        return float(projected[0]), float(projected[1])

    def project_from_bev(self, camera_id: str, bev_x: float, bev_y: float) -> Tuple[float, float]:
        """
        Projects a point from BEV ground coordinates back to camera pixel coordinates.
        """
        H_inv = self.inv_homography_matrices.get(camera_id, self.inv_homography_matrices.get("default"))
        if H_inv is None:
            return float(bev_x), float(bev_y)

        pt = np.array([bev_x, bev_y, 1.0], dtype=np.float32)
        projected = H_inv @ pt
        if abs(projected[2]) > 1e-8:
            return float(projected[0] / projected[2]), float(projected[1] / projected[2])
        return float(projected[0]), float(projected[1])

    def save_calibration(self, filepath: Optional[str] = None):
        """Saves calibration matrices to JSON file."""
        target = Path(filepath) if filepath else self.calibration_file
        if not target:
            raise ValueError("No calibration file path provided.")

        data = {
            cam_id: matrix.tolist()
            for cam_id, matrix in self.homography_matrices.items()
        }
        with open(target, "w") as f:
            json.dump(data, f, indent=2)

    def load_calibration(self, filepath: Optional[str] = None):
        """Loads calibration matrices from JSON file."""
        target = Path(filepath) if filepath else self.calibration_file
        if not target or not target.exists():
            return

        with open(target, "r") as f:
            data = json.load(f)

        for cam_id, mat_list in data.items():
            H = np.array(mat_list, dtype=np.float32)
            self.homography_matrices[cam_id] = H
            try:
                self.inv_homography_matrices[cam_id] = np.linalg.inv(H)
            except np.linalg.LinAlgError:
                pass
