"""
Test Suite for TrackStudio Integration in YUTA:
- Calibration & BEV ground projections
- BEV cluster association & cross-camera global IDs
- Stream Reader buffer and fallback mechanisms
"""

import time
import pytest
import numpy as np

from vision.calibration.calibration import CameraCalibration
from vision.association.bev_merger import BEVClusterMerger
from vision.tracking.tracker_interface import BEVTrack
from vision.stream.rtsp_stream import RTSPStreamReader, StreamFrame


def test_camera_calibration_homography():
    calib = CameraCalibration()

    # Image points: 4 corners of a quadrilateral
    image_pts = [(100.0, 200.0), (500.0, 200.0), (600.0, 600.0), (50.0, 600.0)]
    # BEV target points: rectified rectangle
    bev_pts = [(100.0, 100.0), (400.0, 100.0), (400.0, 500.0), (100.0, 500.0)]

    H = calib.register_camera("cam_01", image_pts, bev_pts)
    assert H.shape == (3, 3)

    # Test projecting one of the known source points
    bx, by = calib.project_to_bev("cam_01", 100.0, 200.0)
    assert pytest.approx(bx, abs=1.0) == 100.0
    assert pytest.approx(by, abs=1.0) == 100.0

    # Test inverse projection from BEV back to camera image
    ix, iy = calib.project_from_bev("cam_01", 100.0, 100.0)
    assert pytest.approx(ix, abs=1.0) == 100.0
    assert pytest.approx(iy, abs=1.0) == 200.0


def test_bev_cluster_merger_same_object_across_cameras():
    merger = BEVClusterMerger(spatial_threshold=50.0, appearance_threshold=0.3)
    now = time.time()

    # Two observations from Camera 1 and Camera 2 in close proximity in BEV space
    emb1 = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    emb2 = np.array([0.52, 0.49, 0.51, 0.48], dtype=np.float32)  # High cosine similarity

    track_cam1 = BEVTrack(
        local_track_id="trk_101",
        camera_id="cam_01",
        bev_x=200.0,
        bev_y=300.0,
        confidence=0.92,
        feature_vector=emb1,
        timestamp=now,
    )

    track_cam2 = BEVTrack(
        local_track_id="trk_505",
        camera_id="cam_02",
        bev_x=210.0,  # 10 units away (within 50.0 threshold)
        bev_y=305.0,
        confidence=0.88,
        feature_vector=emb2,
        timestamp=now,
    )

    merged = merger.merge_tracks([track_cam1, track_cam2], timestamp=now)
    assert len(merged) == 2
    # Both must share the identical Global ID!
    assert merged[0].global_id is not None
    assert merged[0].global_id == merged[1].global_id
    assert "cam_01" in merger.global_tracks[merged[0].global_id].camera_tracks
    assert "cam_02" in merger.global_tracks[merged[0].global_id].camera_tracks


def test_bev_cluster_merger_distinct_objects_separate_ids():
    merger = BEVClusterMerger(spatial_threshold=50.0)
    now = time.time()

    track_cam1 = BEVTrack(
        local_track_id="trk_1",
        camera_id="cam_01",
        bev_x=100.0,
        bev_y=100.0,
        confidence=0.95,
        timestamp=now,
    )

    track_cam2 = BEVTrack(
        local_track_id="trk_2",
        camera_id="cam_02",
        bev_x=500.0,  # Far away: 400+ units
        bev_y=600.0,
        confidence=0.90,
        timestamp=now,
    )

    merged = merger.merge_tracks([track_cam1, track_cam2], timestamp=now)
    assert len(merged) == 2
    assert merged[0].global_id != merged[1].global_id


def test_rtsp_stream_reader_fallback_generation():
    # Test fallback frame generation when disconnected
    reader = RTSPStreamReader(camera_id="cam_test", rtsp_url="rtsp://invalid-host:8554/live")
    fallback = reader._synthesize_status_frame(width=640, height=480, text="Disconnected Test")
    assert fallback.shape == (480, 640, 3)
    assert fallback.dtype == np.uint8
