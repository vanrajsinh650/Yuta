"""
ByteTrack-style Local Object Tracker with 8D Kalman Filter for YUTA.

Derived from GMT (FoxCanned/GMT - Apache 2.0) and FairMOT.
Features:
- Two-stage association: high-confidence detections + low-confidence recovery
- 8-dimensional Kalman filter state: [center_x, center_y, aspect_ratio, height, vx, vy, va, vh]
- Pure NumPy + SciPy linear_sum_assignment (no fragile Cython/C++ build dependencies)
"""

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment

from vision.tracking.tracker_interface import Detection, LocalTrack

logger = logging.getLogger(__name__)


class KalmanBoxTracker:
    """
    8D Kalman filter for tracking 2D bounding boxes in image plane:
    State: [x, y, a, h, vx, vy, va, vh]^T
    where (x, y) is center, a is aspect ratio (w/h), h is height.
    """
    _count = 0

    def __init__(self, bbox: Tuple[float, float, float, float], class_name: str = "car", confidence: float = 1.0):
        # bbox format: [x1, y1, x2, y2]
        KalmanBoxTracker._count += 1
        self.id = KalmanBoxTracker._count
        self.class_name = class_name
        self.confidence = confidence

        self.time_since_update = 0
        self.history: List[Tuple[float, float, float, float]] = []
        self.hits = 1
        self.age = 1
        self.state = "tracked"  # new, tracked, lost, removed

        # 8x8 state transition matrix (constant velocity model)
        self.F = np.eye(8, dtype=np.float32)
        for i in range(4):
            self.F[i, i + 4] = 1.0

        # 4x8 measurement matrix
        self.H = np.zeros((4, 8), dtype=np.float32)
        for i in range(4):
            self.H[i, i] = 1.0

        # Initial state vector x
        self.x = np.zeros((8, 1), dtype=np.float32)
        self.x[:4] = self._bbox_to_z(bbox)

        # Covariance matrices
        self.P = np.eye(8, dtype=np.float32) * 10.0
        self.P[4:, 4:] *= 100.0  # High initial uncertainty in velocities

        self.Q = np.eye(8, dtype=np.float32)
        self.Q[:4, :4] *= 1.0
        self.Q[4:, 4:] *= 0.01

        self.R = np.eye(4, dtype=np.float32) * 1.0

    @staticmethod
    def _bbox_to_z(bbox: Tuple[float, float, float, float]) -> np.ndarray:
        """Convert [x1, y1, x2, y2] to [center_x, center_y, aspect_ratio, height]^T."""
        w = max(1.0, bbox[2] - bbox[0])
        h = max(1.0, bbox[3] - bbox[1])
        x = bbox[0] + w / 2.0
        y = bbox[1] + h / 2.0
        a = w / h
        return np.array([[x], [y], [a], [h]], dtype=np.float32)

    def _z_to_bbox(self) -> Tuple[float, float, float, float]:
        """Convert [center_x, center_y, aspect_ratio, height] back to [x1, y1, x2, y2]."""
        x, y, a, h = float(self.x[0, 0]), float(self.x[1, 0]), float(self.x[2, 0]), float(self.x[3, 0])
        w = max(1.0, a * h)
        h = max(1.0, h)
        return (x - w / 2.0, y - h / 2.0, x + w / 2.0, y + h / 2.0)

    def predict(self) -> Tuple[float, float, float, float]:
        """Advances state vector by one time step."""
        if self.x[3, 0] + self.x[7, 0] <= 0:
            self.x[7, 0] = 0.0
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1
        self.time_since_update += 1
        pred_box = self._z_to_bbox()
        self.history.append(pred_box)
        return pred_box

    def update(self, bbox: Tuple[float, float, float, float], confidence: float):
        """Updates state with observed bounding box measurement."""
        self.time_since_update = 0
        self.hits += 1
        self.confidence = confidence

        z = self._bbox_to_z(bbox)
        y = z - self.H @ self.x  # Innovation
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain
        self.x = self.x + K @ y
        I = np.eye(8, dtype=np.float32)
        self.P = (I - K @ self.H) @ self.P

        self.state = "tracked"

    def get_bbox(self) -> Tuple[float, float, float, float]:
        return self._z_to_bbox()


def calculate_ious(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """
    Vectorized IoU calculation between boxes1 (N, 4) and boxes2 (M, 4).
    boxes format: [x1, y1, x2, y2]
    """
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)

    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = np.maximum(boxes1[:, None, :2], boxes2[None, :, :2])  # [N, M, 2]
    rb = np.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])  # [N, M, 2]

    wh = np.clip(rb - lt, 0, None)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

    union = area1[:, None] + area2[None, :] - inter
    union = np.maximum(union, 1e-6)

    return inter / union


class ByteTracker:
    """
    Two-stage ByteTrack algorithm for single-camera vehicle tracking.
    """

    def __init__(
        self,
        camera_id: str,
        high_threshold: float = 0.5,
        low_threshold: float = 0.1,
        match_threshold: float = 0.8,  # Max distance (1 - IoU) for stage 1
        match_threshold_low: float = 0.5,  # Max distance for stage 2
        max_time_lost: int = 30,  # Keep track active up to 30 missed frames
    ):
        self.camera_id = camera_id
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.match_threshold = match_threshold
        self.match_threshold_low = match_threshold_low
        self.max_time_lost = max_time_lost

        self.tracked_stracks: List[KalmanBoxTracker] = []
        self.lost_stracks: List[KalmanBoxTracker] = []
        self.frame_id = 0

    def update(self, detections: List[Detection]) -> List[LocalTrack]:
        """
        Updates tracks with new detections from current video frame.
        """
        self.frame_id += 1

        # Predict current locations for all existing tracks
        for track in self.tracked_stracks:
            track.predict()
        for track in self.lost_stracks:
            track.predict()

        # Partition detections into high and low confidence
        dets_high: List[Detection] = []
        dets_low: List[Detection] = []
        for det in detections:
            if det.confidence >= self.high_threshold:
                dets_high.append(det)
            elif det.confidence >= self.low_threshold:
                dets_low.append(det)

        # STAGE 1: Match high-confidence detections with active tracks
        active_pool = self.tracked_stracks + self.lost_stracks
        matched_tracks_1, unmatched_tracks_1, unmatched_dets_1 = self._associate(
            active_pool, dets_high, max_dist=self.match_threshold
        )

        for track, det in matched_tracks_1:
            track.update(det.bbox, det.confidence)
            if track in self.lost_stracks:
                self.lost_stracks.remove(track)
                self.tracked_stracks.append(track)

        # STAGE 2: Match remaining tracks with low-confidence detections
        # Only try to match tracks that were already active (not new ones)
        remain_tracks = [t for t in unmatched_tracks_1 if t.state == "tracked"]
        matched_tracks_2, unmatched_tracks_2, _ = self._associate(
            remain_tracks, dets_low, max_dist=self.match_threshold_low
        )

        for track, det in matched_tracks_2:
            track.update(det.bbox, det.confidence)
            if track in self.lost_stracks:
                self.lost_stracks.remove(track)
                self.tracked_stracks.append(track)

        # Handle unmatched tracks -> mark lost
        for track in unmatched_tracks_2:
            if track in self.tracked_stracks:
                self.tracked_stracks.remove(track)
                track.state = "lost"
                self.lost_stracks.append(track)

        # Clean expired lost tracks
        self.lost_stracks = [
            t for t in self.lost_stracks if t.time_since_update <= self.max_time_lost
        ]

        # STAGE 3: Initialize new tracks from unmatched high-confidence detections
        for det_idx in unmatched_dets_1:
            det = dets_high[det_idx]
            new_track = KalmanBoxTracker(det.bbox, class_name=det.class_name, confidence=det.confidence)
            self.tracked_stracks.append(new_track)

        # Output active tracks
        output_tracks: List[LocalTrack] = []
        for track in self.tracked_stracks:
            bbox = track.get_bbox()
            output_tracks.append(
                LocalTrack(
                    track_id=f"{self.camera_id}_trk_{track.id}",
                    camera_id=self.camera_id,
                    bbox=bbox,
                    confidence=track.confidence,
                    age=track.age,
                    hits=track.hits,
                    time_since_update=track.time_since_update,
                    state="active",
                )
            )

        return output_tracks

    def _associate(
        self, tracks: List[KalmanBoxTracker], detections: List[Detection], max_dist: float
    ) -> Tuple[List[Tuple[KalmanBoxTracker, Detection]], List[KalmanBoxTracker], List[int]]:
        """Associates tracks to detections via Hungarian algorithm on (1 - IoU) cost."""
        if len(tracks) == 0 or len(detections) == 0:
            return [], tracks.copy(), list(range(len(detections)))

        track_boxes = np.array([t.get_bbox() for t in tracks], dtype=np.float32)
        det_boxes = np.array([d.bbox for d in detections], dtype=np.float32)

        ious = calculate_ious(track_boxes, det_boxes)
        cost_matrix = 1.0 - ious

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched_tracks: List[Tuple[KalmanBoxTracker, Detection]] = []
        unmatched_track_indices = set(range(len(tracks)))
        unmatched_det_indices = set(range(len(detections)))

        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] <= max_dist:
                matched_tracks.append((tracks[r], detections[c]))
                unmatched_track_indices.discard(r)
                unmatched_det_indices.discard(c)

        unmatched_tracks = [tracks[i] for i in unmatched_track_indices]
        return matched_tracks, unmatched_tracks, list(unmatched_det_indices)
