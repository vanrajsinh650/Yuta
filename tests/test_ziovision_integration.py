"""
Test Suite for ZIOVISION AIC2025 Trajectory Linker Integration in YUTA.
"""

import time
import pytest
import numpy as np

from vision.association.trajectory_linker import GlobalTrajectoryLinker, Tracklet


def test_trajectory_linker_cross_camera_success():
    linker = GlobalTrajectoryLinker(
        max_time_gap_sec=30.0,
        max_spatial_distance=100.0,
        appearance_similarity_threshold=0.7,
        max_speed_units_per_sec=25.0,
    )
    t0 = 1000.0

    feat_vehicle_a = np.array([0.4, 0.4, 0.4, 0.4], dtype=np.float32)
    feat_vehicle_a_cam2 = np.array([0.42, 0.38, 0.41, 0.39], dtype=np.float32)

    # Tracklet 1 in Camera 01: exits at (50, 50) at t=1010
    trk1 = Tracklet(
        tracklet_id="trk_cam1_01",
        camera_id="cam_01",
        dominant_class="car",
        first_timestamp=t0,
        last_timestamp=t0 + 10.0,
        first_pos=(10.0, 10.0),
        last_pos=(50.0, 50.0),
        mean_feature=feat_vehicle_a,
        plate_text="GJ01AB1234",
        plate_confidence=0.92,
        detection_count=15,
    )

    # Tracklet 2 in Camera 02: enters at (70, 60) at t=1015 (5 seconds later, ~28 units away -> speed ~5.6 units/s)
    trk2 = Tracklet(
        tracklet_id="trk_cam2_05",
        camera_id="cam_02",
        dominant_class="car",
        first_timestamp=t0 + 15.0,
        last_timestamp=t0 + 25.0,
        first_pos=(70.0, 60.0),
        last_pos=(120.0, 100.0),
        mean_feature=feat_vehicle_a_cam2,
        plate_text="GJ01AB1234",
        plate_confidence=0.90,
        detection_count=15,
    )

    trajectories, links = linker.link_tracklets([trk1, trk2])
    # Both tracklets should be merged into a single global vehicle trajectory
    assert len(trajectories) == 1
    gid = list(trajectories.keys())[0]
    assert len(trajectories[gid]) == 2
    assert len(links) == 1
    assert links[0].parent_id == "trk_cam1_01"
    assert links[0].child_id == "trk_cam2_05"
    assert links[0].plate_match is True


def test_trajectory_linker_rejects_impossible_speed():
    linker = GlobalTrajectoryLinker(
        max_time_gap_sec=30.0,
        max_spatial_distance=500.0,
        max_speed_units_per_sec=20.0,  # Max speed: 20 units/sec
    )
    t0 = 1000.0

    # Tracklet 1 exits at (0, 0) at t=1010
    trk1 = Tracklet(
        tracklet_id="t1",
        camera_id="cam_01",
        dominant_class="car",
        first_timestamp=t0,
        last_timestamp=t0 + 10.0,
        first_pos=(0.0, 0.0),
        last_pos=(0.0, 0.0),
        detection_count=10,
    )

    # Tracklet 2 appears 400 units away only 1 second later (speed = 400 units/s > 20)
    trk2 = Tracklet(
        tracklet_id="t2",
        camera_id="cam_02",
        dominant_class="car",
        first_timestamp=t0 + 11.0,
        last_timestamp=t0 + 20.0,
        first_pos=(400.0, 0.0),
        last_pos=(450.0, 0.0),
        detection_count=10,
    )

    trajectories, links = linker.link_tracklets([trk1, trk2])
    # Must NOT be merged due to physically impossible speed!
    assert len(trajectories) == 2
    assert len(links) == 0


def test_trajectory_linker_rejects_conflicting_plates():
    linker = GlobalTrajectoryLinker()
    t0 = 1000.0

    trk1 = Tracklet(
        tracklet_id="t1",
        camera_id="cam_01",
        dominant_class="car",
        first_timestamp=t0,
        last_timestamp=t0 + 10.0,
        first_pos=(0.0, 0.0),
        last_pos=(50.0, 50.0),
        plate_text="GJ01AB1234",
        plate_confidence=0.95,
        detection_count=10,
    )

    trk2 = Tracklet(
        tracklet_id="t2",
        camera_id="cam_02",
        dominant_class="car",
        first_timestamp=t0 + 15.0,
        last_timestamp=t0 + 25.0,
        first_pos=(55.0, 55.0),
        last_pos=(100.0, 100.0),
        plate_text="MH12DE9876",  # Different plate!
        plate_confidence=0.96,
        detection_count=10,
    )

    trajectories, links = linker.link_tracklets([trk1, trk2])
    # Must NOT merge two different vehicles with conflicting verified plates
    assert len(trajectories) == 2
    assert len(links) == 0


def test_trajectory_linker_multihop_three_cameras_with_gmt_attention():
    """
    Tests full 3-camera sequential trajectory linking (Cam 1 -> Cam 2 -> Cam 3)
    coupled with FoxCanned/GMT cross-attention affinity module.
    """
    from vision.association.attention_association import AttentionAssociationHead

    feat_dim = 64
    attention_head = AttentionAssociationHead(feature_dim=feat_dim)

    linker = GlobalTrajectoryLinker(
        max_time_gap_sec=40.0,
        max_spatial_distance=150.0,
        appearance_similarity_threshold=0.6,
        max_speed_units_per_sec=25.0,
        attention_head=attention_head,
    )

    t0 = 2000.0
    # Vehicle feature vector
    base_feat = np.ones((feat_dim,), dtype=np.float32) / np.sqrt(feat_dim)

    # Tracklet 1 in Cam 01: t=2000 to 2010, pos (10, 10) -> (50, 50)
    t1 = Tracklet(
        tracklet_id="trk_c1",
        camera_id="cam_01",
        dominant_class="car",
        first_timestamp=t0,
        last_timestamp=t0 + 10.0,
        first_pos=(10.0, 10.0),
        last_pos=(50.0, 50.0),
        mean_feature=base_feat.copy(),
        plate_text="GJ01AB1234",
        plate_confidence=0.92,
        detection_count=10,
    )

    # Tracklet 2 in Cam 02: t=2015 to 2025, pos (60, 60) -> (100, 100)
    t2 = Tracklet(
        tracklet_id="trk_c2",
        camera_id="cam_02",
        dominant_class="car",
        first_timestamp=t0 + 15.0,
        last_timestamp=t0 + 25.0,
        first_pos=(60.0, 60.0),
        last_pos=(100.0, 100.0),
        mean_feature=base_feat.copy(),
        plate_text="GJ01AB1234",
        plate_confidence=0.90,
        detection_count=10,
    )

    # Tracklet 3 in Cam 03: t=2030 to 2040, pos (110, 110) -> (150, 150)
    t3 = Tracklet(
        tracklet_id="trk_c3",
        camera_id="cam_03",
        dominant_class="car",
        first_timestamp=t0 + 30.0,
        last_timestamp=t0 + 40.0,
        first_pos=(110.0, 110.0),
        last_pos=(150.0, 150.0),
        mean_feature=base_feat.copy(),
        plate_text="GJ01AB1234",
        plate_confidence=0.91,
        detection_count=10,
    )

    trajectories, links = linker.link_tracklets([t1, t2, t3])

    # Must resolve all 3 tracklets across the 3 cameras into a single global vehicle ID!
    assert len(trajectories) == 1
    gid = list(trajectories.keys())[0]
    merged_tracklets = trajectories[gid]
    assert len(merged_tracklets) == 3
    assert [t.camera_id for t in merged_tracklets] == ["cam_01", "cam_02", "cam_03"]
    assert len(links) == 2
    assert links[0].parent_id == "trk_c1" and links[0].child_id == "trk_c2"
    assert links[1].parent_id == "trk_c1" and links[1].child_id == "trk_c3"
