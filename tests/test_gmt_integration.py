"""
Test Suite for GMT Integration in YUTA:
- ByteTracker with 8D Kalman filter & two-stage matching
- AttentionAssociationHead cross-affinity computation
"""

import pytest
import numpy as np

from vision.tracking.byte_tracker import ByteTracker
from vision.tracking.tracker_interface import Detection
from vision.association.attention_association import AttentionAssociationHead


def test_byte_tracker_continuity_and_low_confidence_recovery():
    tracker = ByteTracker(camera_id="cam_01", high_threshold=0.5, low_threshold=0.1)

    # Frame 1: Vehicle detected with high confidence at (100, 100, 200, 200)
    det1 = [Detection(bbox=(100.0, 100.0, 200.0, 200.0), confidence=0.92, class_name="car", class_id=2, camera_id="cam_01", timestamp=1.0)]
    tracks1 = tracker.update(det1)
    assert len(tracks1) == 1
    tid = tracks1[0].track_id

    # Frame 2: Vehicle moves slightly to (105, 105, 205, 205) with high confidence
    det2 = [Detection(bbox=(105.0, 105.0, 205.0, 205.0), confidence=0.88, class_name="car", class_id=2, camera_id="cam_01", timestamp=2.0)]
    tracks2 = tracker.update(det2)
    assert len(tracks2) == 1
    assert tracks2[0].track_id == tid

    # Frame 3: Heavy occlusion or blur causes confidence to drop to 0.2 (low-confidence detection)
    det3 = [Detection(bbox=(110.0, 110.0, 210.0, 210.0), confidence=0.20, class_name="car", class_id=2, camera_id="cam_01", timestamp=3.0)]
    tracks3 = tracker.update(det3)
    # ByteTrack stage 2 must recover this detection and keep the identical track ID!
    assert len(tracks3) == 1
    assert tracks3[0].track_id == tid


def test_byte_tracker_multiple_objects_distinct_ids():
    tracker = ByteTracker(camera_id="cam_01")

    dets = [
        Detection(bbox=(50.0, 50.0, 100.0, 100.0), confidence=0.9, class_name="car", class_id=2, camera_id="cam_01", timestamp=1.0),
        Detection(bbox=(400.0, 400.0, 500.0, 500.0), confidence=0.95, class_name="truck", class_id=7, camera_id="cam_01", timestamp=1.0),
    ]
    tracks = tracker.update(dets)
    assert len(tracks) == 2
    assert tracks[0].track_id != tracks[1].track_id


def test_attention_association_head():
    head = AttentionAssociationHead(feature_dim=64)

    # Vector A and identical/very similar vector A'
    v_a = np.ones((1, 64), dtype=np.float32) / np.sqrt(64)
    v_a_cand = v_a.copy()

    # Dissimilar vector B (orthogonal/negative direction)
    v_b_cand = -v_a.copy()

    keys = np.vstack([v_a_cand, v_b_cand])  # (2, 64)
    affinity = head.compute_affinity(v_a, keys)  # (1, 2)

    assert affinity.shape == (1, 2)
    # Affinity for candidate A must be significantly higher than candidate B
    assert affinity[0, 0] > affinity[0, 1]
    assert pytest.approx(affinity.sum(axis=-1)[0], abs=1e-5) == 1.0
