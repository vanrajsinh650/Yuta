"""
Test Suite for Cam-Traj-Rec Route Reconstruction and Camera Topology in YUTA.
"""

import pytest
from vision.trajectory.route_reconstructor import CameraTopologyGraph


def test_camera_topology_and_haversine():
    topo = CameraTopologyGraph(default_speed_kmh=40.0)

    # Ahmedabad City Locations (e.g., Ashram Road corridor)
    topo.add_camera("CAM_01", "Income Tax Circle", 23.0425, 72.5714)
    topo.add_camera("CAM_02", "Vadaj Circle", 23.0588, 72.5728)

    # Check distance
    dist = topo.haversine_distance(23.0425, 72.5714, 23.0588, 72.5728)
    assert 1700 < dist < 1900  # Approximately 1.8 km

    topo.add_road_connection("CAM_01", "CAM_02", distance_meters=dist, speed_limit_kmh=50.0)
    assert topo.graph.has_edge("CAM_01", "CAM_02")


def test_travel_likelihood_realistic_vs_impossible():
    topo = CameraTopologyGraph(default_speed_kmh=40.0)
    dist = 2000.0  # 2 km
    topo.add_camera("CAM_A", "Cam A", 23.0, 72.0)
    topo.add_camera("CAM_B", "Cam B", 23.018, 72.0)
    topo.add_road_connection("CAM_A", "CAM_B", distance_meters=dist, speed_limit_kmh=40.0)

    # Case 1: 3 minutes (180s) to cover 2km = 40 km/h (Optimal speed)
    likelihood_optimal, _, speed_optimal = topo.compute_travel_likelihood("CAM_A", "CAM_B", actual_time_sec=180.0)
    assert likelihood_optimal > 0.8
    assert pytest.approx(speed_optimal, abs=1.0) == 40.0

    # Case 2: 2 seconds to cover 2km = 3600 km/h (Physically impossible teleportation)
    likelihood_impossible, _, _ = topo.compute_travel_likelihood("CAM_A", "CAM_B", actual_time_sec=2.0)
    assert likelihood_impossible < 1e-10


def test_route_reconstruction_and_geojson_generation():
    topo = CameraTopologyGraph()
    topo.add_camera("C1", "Junction 1", 23.01, 72.51)
    topo.add_camera("C2", "Junction 2", 23.02, 72.52)
    topo.add_camera("C3", "Junction 3", 23.03, 72.53)

    topo.add_road_connection("C1", "C2", distance_meters=1500.0)
    topo.add_road_connection("C2", "C3", distance_meters=1800.0)

    # Sightings: C1 at t=0, C2 at t=150s, C3 at t=320s
    sightings = [("C1", 1000.0), ("C2", 1150.0), ("C3", 1320.0)]

    route = topo.reconstruct_route(
        sightings,
        global_vehicle_id="VEH-GUJ-0042",
        plate_number="GJ01AB1234",
    )

    assert route is not None
    assert route.global_vehicle_id == "VEH-GUJ-0042"
    assert route.plate_number == "GJ01AB1234"
    assert len(route.segments) == 2
    assert route.total_distance_meters == 3300.0
    assert route.total_duration_sec == 320.0
    assert route.overall_confidence > 0.0

    # GeoJSON validation
    gj = route.geojson_feature
    assert gj["type"] == "Feature"
    assert gj["properties"]["vehicle_id"] == "VEH-GUJ-0042"
    assert gj["geometry"]["type"] == "LineString"
    assert len(gj["geometry"]["coordinates"]) >= 3
