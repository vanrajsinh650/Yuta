"""
Integration Test Suite for YUTA FastAPI Backend and Investigation Endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_health_endpoint():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["system"] == "YUTA — Unified Video Intelligence Platform"
    assert data["cameras_active"] >= 4


def test_catalogue_and_cameras():
    resp = client.get("/api/cameras")
    assert resp.status_code == 200
    cams = resp.json()
    assert len(cams) >= 4
    cam_ids = [c["camera_id"] for c in cams]
    assert "CAM_01" in cam_ids


def test_vehicle_search_and_route():
    resp = client.get("/api/vehicles/search?q=GJ01AB1234")
    assert resp.status_code == 200
    vehicles = resp.json()
    assert len(vehicles) >= 1
    vid = vehicles[0]["global_id"]

    # Test route reconstruction GeoJSON
    route_resp = client.get(f"/api/vehicles/{vid}/route")
    assert route_resp.status_code == 200
    route = route_resp.json()
    assert route["global_id"] == vid
    assert "geojson" in route
    assert route["geojson"]["type"] == "Feature"
    assert len(route["segments"]) >= 2


def test_vehicle_evidence_graph():
    resp = client.get("/api/vehicles/search?q=GJ01AB1234")
    vid = resp.json()[0]["global_id"]

    ev_resp = client.get(f"/api/vehicles/{vid}/evidence")
    assert ev_resp.status_code == 200
    data = ev_resp.json()
    assert len(data["nodes"]) >= 3
    assert len(data["edges"]) >= 2
    assert data["edges"][0]["plate_match"] is True


def test_watchlist_and_alerts():
    # List alerts
    resp = client.get("/api/alerts")
    assert resp.status_code == 200
    alerts = resp.json()
    assert len(alerts) >= 1
    assert any("GJ01AB1234" in a["plate_number"] for a in alerts)

    # Add new plate to watchlist
    create_resp = client.post("/api/watchlist", json={
        "plate_number": "GJ05CD5678",
        "reason": "Vehicle under investigation in Surat",
        "severity": "HIGH",
    })
    assert create_resp.status_code == 200

    # Verify it exists
    list_resp = client.get("/api/watchlist")
    assert any(w["plate_number"] == "GJ05CD5678" for w in list_resp.json())


def test_nl_investigation_endpoint():
    resp = client.post("/api/investigate/nlq", json={
        "query": "Show all sightings of GJ01AB1234"
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["intent"] == "plate_search"
    assert len(data["evidence"]) >= 3
    assert "GLOBAL-VEH-0001" in data["matched_vehicles"]
