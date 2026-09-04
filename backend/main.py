"""
YUTA — Unified Video Intelligence Platform
FastAPI Server & Investigation API
Gujarat Police Innovation Challenge 2026
"""

import time
import asyncio
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# Vision & Tracking Engines
from vision.stream.sentinel_client import SentinelCatalogueClient, SentinelCamera
from vision.calibration.calibration import CameraCalibration
from vision.association.bev_merger import BEVClusterMerger
from vision.association.trajectory_linker import GlobalTrajectoryLinker, Tracklet
from vision.trajectory.route_reconstructor import CameraTopologyGraph, ReconstructedRoute
from vision.anpr.indian_anpr_engine import IndianPlateValidator, TemporalTrackANPRVoter
from vision.tracking.byte_tracker import ByteTracker
from vision.tracking.tracker_interface import Detection

# Investigation & Alert Engines
from backend.investigation.evidence_graph import EvidenceGraph, EvidenceNode, EvidenceEdge
from backend.investigation.nl_query_engine import InvestigationQueryEngine, InvestigationAnswer
from backend.alerts.watchlist_engine import WatchlistAlertEngine, WatchlistEntry, AlertEvent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("YUTA")

app = FastAPI(
    title="YUTA — Unified Video Intelligence Platform",
    description="Advanced Multi-Camera Video Intelligence, Indian ANPR, and Spatio-Temporal Route Reconstruction for Gujarat Police",
    version="1.0.0",
)

# Enable CORS for Frontend UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/")
    def serve_frontend_index():
        return FileResponse(FRONTEND_DIR / "index.html")


# === Global In-Memory Intelligence Core ===
catalogue_client = SentinelCatalogueClient()
calibration = CameraCalibration()
bev_merger = BEVClusterMerger(spatial_threshold=80.0, appearance_threshold=0.35)
trajectory_linker = GlobalTrajectoryLinker(max_time_gap_sec=120.0, max_speed_units_per_sec=40.0)
topology_graph = CameraTopologyGraph(default_speed_kmh=45.0)
anpr_voter = TemporalTrackANPRVoter()
evidence_graph = EvidenceGraph()
query_engine = InvestigationQueryEngine(evidence_graph)
watchlist_engine = WatchlistAlertEngine()

# Active camera trackers: camera_id -> ByteTracker
camera_trackers: Dict[str, ByteTracker] = {}
# Active WebSocket subscribers
active_websockets: List[WebSocket] = []


def initialize_demo_dataset():
    """Initializes realistic camera topology and tracked vehicles for demonstration."""
    cameras = catalogue_client.fetch_catalogue()

    # Build road connectivity in topology graph along Ahmedabad corridor
    for cam in cameras:
        topology_graph.add_camera(
            camera_id=cam.camera_id,
            name=cam.name,
            lat=cam.lat,
            lon=cam.lon,
        )
        if cam.camera_id not in camera_trackers:
            camera_trackers[cam.camera_id] = ByteTracker(camera_id=cam.camera_id)

    # Road edges between consecutive cameras
    cam_ids = [c.camera_id for c in cameras]
    for i in range(len(cam_ids) - 1):
        c1, c2 = cam_ids[i], cam_ids[i + 1]
        topology_graph.add_road_connection(c1, c2, speed_limit_kmh=50.0)

    # Seed initial realistic vehicle trajectory: White SUV GJ01AB1234
    t_start = time.time() - 300.0  # 5 minutes ago
    v1_id = "GLOBAL-VEH-0001"
    v1_plate = "GJ01AB1234"

    sighting_cams = cam_ids[:4]  # Traveled through first 4 cameras
    for idx, cid in enumerate(sighting_cams):
        cam_obj = catalogue_client.cameras[cid]
        s_time = t_start + (idx * 65.0)  # 65s between cameras

        node = EvidenceNode(
            node_id=f"node_{v1_id}_{cid}",
            global_vehicle_id=v1_id,
            camera_id=cid,
            camera_name=cam_obj.name,
            timestamp=s_time,
            bbox=(200.0 + idx * 10, 150.0, 450.0 + idx * 10, 380.0),
            plate_number=v1_plate,
            plate_confidence=0.94,
            vehicle_class="suv",
            vehicle_color="white",
            appearance_embedding=np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32),
        )
        evidence_graph.add_node(node)

        # Trigger watchlist alert for GJ01AB1234 (configured as wanted vehicle)
        watchlist_engine.check_observation(
            global_vehicle_id=v1_id,
            plate_number=v1_plate,
            camera_id=cid,
            camera_name=cam_obj.name,
            lat=cam_obj.lat,
            lon=cam_obj.lon,
            confidence=0.94,
            timestamp=s_time,
        )

        # Add transition edge
        if idx > 0:
            prev_cid = sighting_cams[idx - 1]
            prev_node_id = f"node_{v1_id}_{prev_cid}"
            edge = EvidenceEdge(
                source_node_id=prev_node_id,
                target_node_id=node.node_id,
                from_camera=prev_cid,
                to_camera=cid,
                time_gap_sec=65.0,
                spatial_dist_meters=850.0,
                implied_speed_kmh=47.1,
                appearance_similarity=0.98,
                plate_match=True,
                road_likelihood=0.96,
                confidence=0.98,
                reason="consecutive_plate_match_and_reid",
            )
            evidence_graph.add_edge(edge)


# Run startup initialization
initialize_demo_dataset()


# === Request / Response Schemas ===
class WatchlistCreateRequest(BaseModel):
    plate_number: str
    reason: str
    severity: str = "HIGH"
    case_id: Optional[str] = None
    vehicle_model: Optional[str] = None
    vehicle_color: Optional[str] = None


class NaturalLanguageQueryRequest(BaseModel):
    query: str


# === REST Endpoints ===

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "system": "YUTA — Unified Video Intelligence Platform",
        "version": "1.0.0",
        "cameras_active": len(catalogue_client.cameras),
        "tracked_vehicles": len(evidence_graph.vehicle_index),
        "active_alerts": len(watchlist_engine.alerts),
    }


@app.get("/api/ingest")
def get_sentinel_catalogue():
    """Returns the Sentinel catalogue cameras with stream endpoints."""
    return [
        {
            "camera_id": c.camera_id,
            "name": c.name,
            "lat": c.lat,
            "lon": c.lon,
            "rtsp_url": c.rtsp_url,
            "whep_url": c.whep_url,
            "hls_url": c.hls_url,
            "status": c.status,
        }
        for c in catalogue_client.cameras.values()
    ]


@app.get("/api/cameras")
def list_cameras():
    """Returns all camera nodes in the active network."""
    return get_sentinel_catalogue()


@app.get("/api/vehicles")
def list_vehicles():
    """Lists all tracked vehicles with latest state and sightings."""
    vehicles = []
    for vid, node_ids in evidence_graph.vehicle_index.items():
        nodes = [evidence_graph.nodes[nid] for nid in node_ids]
        nodes.sort(key=lambda n: n.timestamp)
        first_node = nodes[0]
        latest_node = nodes[-1]

        vehicles.append({
            "global_id": vid,
            "plate_number": latest_node.plate_number,
            "plate_confidence": latest_node.plate_confidence,
            "vehicle_class": latest_node.vehicle_class,
            "vehicle_color": latest_node.vehicle_color,
            "first_seen_time": first_node.timestamp,
            "last_seen_time": latest_node.timestamp,
            "last_camera_id": latest_node.camera_id,
            "last_camera_name": latest_node.camera_name,
            "total_sightings": len(nodes),
            "cameras_traversed": list({n.camera_id for n in nodes}),
        })
    return vehicles


@app.get("/api/vehicles/search")
def search_vehicles(
    q: Optional[str] = Query(None, description="Plate search (exact or partial)"),
    vehicle_class: Optional[str] = Query(None),
    vehicle_color: Optional[str] = Query(None),
    camera_id: Optional[str] = Query(None),
):
    """Multi-attribute vehicle search across plates, colors, classes, and cameras."""
    matched_vids = set()

    if q:
        plate_vids = evidence_graph.search_by_plate(q)
        matched_vids.update(plate_vids)

    results = []
    candidates = list(matched_vids) if q else list(evidence_graph.vehicle_index.keys())

    for vid in candidates:
        nodes = evidence_graph.get_vehicle_nodes(vid)
        if not nodes:
            continue
        latest = nodes[-1]

        if vehicle_class and vehicle_class.lower() not in latest.vehicle_class.lower():
            continue
        if vehicle_color and vehicle_color.lower() not in latest.vehicle_color.lower():
            continue
        if camera_id and not any(n.camera_id == camera_id for n in nodes):
            continue

        results.append({
            "global_id": vid,
            "plate_number": latest.plate_number,
            "confidence": latest.plate_confidence,
            "vehicle_class": latest.vehicle_class,
            "vehicle_color": latest.vehicle_color,
            "sightings_count": len(nodes),
            "last_camera": latest.camera_name,
            "last_seen": latest.timestamp,
        })

    return results


@app.get("/api/vehicles/{global_id}")
def get_vehicle_details(global_id: str):
    nodes = evidence_graph.get_vehicle_nodes(global_id)
    if not nodes:
        raise HTTPException(status_code=404, detail="Vehicle identity not found.")

    latest = nodes[-1]
    return {
        "global_id": global_id,
        "plate_number": latest.plate_number,
        "plate_confidence": latest.plate_confidence,
        "vehicle_class": latest.vehicle_class,
        "vehicle_color": latest.vehicle_color,
        "first_seen": nodes[0].timestamp,
        "last_seen": latest.timestamp,
        "nodes": [
            {
                "node_id": n.node_id,
                "camera_id": n.camera_id,
                "camera_name": n.camera_name,
                "timestamp": n.timestamp,
                "bbox": n.bbox,
                "confidence": n.plate_confidence,
            }
            for n in nodes
        ],
    }


@app.get("/api/vehicles/{global_id}/route")
def get_vehicle_route(global_id: str):
    """Reconstructs spatio-temporal route and returns GeoJSON for GIS map display."""
    nodes = evidence_graph.get_vehicle_nodes(global_id)
    if not nodes or len(nodes) < 2:
        # Return point feature if only 1 sighting
        if nodes:
            cam = catalogue_client.cameras.get(nodes[0].camera_id)
            lat = cam.lat if cam else 23.0
            lon = cam.lon if cam else 72.0
            return {
                "type": "Feature",
                "properties": {"vehicle_id": global_id, "status": "single_sighting"},
                "geometry": {"type": "Point", "coordinates": [lon, lat]},
            }
        raise HTTPException(status_code=404, detail="Vehicle not found.")

    sightings = [(n.camera_id, n.timestamp) for n in nodes]
    route = topology_graph.reconstruct_route(
        sightings=sightings,
        global_vehicle_id=global_id,
        plate_number=nodes[-1].plate_number,
    )
    if not route:
        raise HTTPException(status_code=500, detail="Failed to reconstruct route.")

    return {
        "global_id": route.global_vehicle_id,
        "plate_number": route.plate_number,
        "total_distance_m": round(route.total_distance_meters, 1),
        "duration_sec": round(route.total_duration_sec, 1),
        "average_speed_kmh": round(route.average_speed_kmh, 1),
        "overall_confidence": round(route.overall_confidence, 3),
        "geojson": route.geojson_feature,
        "segments": [
            {
                "from_camera": s.from_camera,
                "to_camera": s.to_camera,
                "distance_m": round(s.distance_meters, 1),
                "time_sec": round(s.travel_time_sec, 1),
                "speed_kmh": round(s.implied_speed_kmh, 1),
                "likelihood": round(s.likelihood, 3),
            }
            for s in route.segments
        ],
    }


@app.get("/api/vehicles/{global_id}/evidence")
def get_vehicle_evidence_graph(global_id: str):
    """Returns the Grand-Finale Evidence Graph for a vehicle."""
    nodes = evidence_graph.get_vehicle_nodes(global_id)
    if not nodes:
        raise HTTPException(status_code=404, detail="Vehicle not found.")

    edges = evidence_graph.get_vehicle_edges(global_id)

    return {
        "global_vehicle_id": global_id,
        "plate_number": nodes[-1].plate_number,
        "nodes": [
            {
                "node_id": n.node_id,
                "camera_id": n.camera_id,
                "camera_name": n.camera_name,
                "timestamp": n.timestamp,
                "plate_number": n.plate_number,
                "plate_confidence": n.plate_confidence,
                "vehicle_class": n.vehicle_class,
                "vehicle_color": n.vehicle_color,
            }
            for n in nodes
        ],
        "edges": [
            {
                "source": e.source_node_id,
                "target": e.target_node_id,
                "from_camera": e.from_camera,
                "to_camera": e.to_camera,
                "time_gap_sec": round(e.time_gap_sec, 1),
                "distance_m": round(e.spatial_dist_meters, 1),
                "implied_speed_kmh": round(e.implied_speed_kmh, 1),
                "appearance_similarity": round(e.appearance_similarity, 3),
                "plate_match": e.plate_match,
                "road_likelihood": round(e.road_likelihood, 3),
                "confidence": round(e.confidence, 3),
                "reason": e.reason,
            }
            for e in edges
        ],
    }


@app.get("/api/watchlist")
def get_watchlist():
    return list(watchlist_engine.watchlist.values())


@app.post("/api/watchlist")
def add_to_watchlist(req: WatchlistCreateRequest):
    entry = watchlist_engine.add_entry(
        plate_number=req.plate_number,
        reason=req.reason,
        severity=req.severity,
        case_id=req.case_id,
        vehicle_model=req.vehicle_model,
        vehicle_color=req.vehicle_color,
    )
    return {"status": "success", "entry": entry}


@app.delete("/api/watchlist/{plate}")
def remove_from_watchlist(plate: str):
    removed = watchlist_engine.remove_entry(plate)
    if not removed:
        raise HTTPException(status_code=404, detail="Plate not found in watchlist.")
    return {"status": "removed", "plate": plate}


@app.get("/api/alerts")
def get_alerts(limit: int = 50):
    return watchlist_engine.get_recent_alerts(limit=limit)


@app.post("/api/investigate/nlq")
def natural_language_investigate(req: NaturalLanguageQueryRequest):
    """Grounded AI investigation query engine."""
    ans = query_engine.execute_query(req.query)
    return {
        "query": ans.query,
        "intent": ans.intent,
        "summary": ans.summary_answer,
        "matched_vehicles": ans.matched_vehicles,
        "evidence": ans.evidence_sightings,
        "confidence": round(ans.confidence, 3),
    }


@app.websocket("/ws/live-events")
async def websocket_live_events(websocket: WebSocket):
    await websocket.accept()
    active_websockets.append(websocket)
    logger.info("Client connected to /ws/live-events")
    try:
        # Send initial status
        await websocket.send_json({
            "type": "INITIAL_STATE",
            "cameras": [c.camera_id for c in catalogue_client.cameras.values()],
            "active_alerts_count": len(watchlist_engine.alerts),
        })
        while True:
            # Keepalive / ping
            await asyncio.sleep(10)
            await websocket.send_json({"type": "HEARTBEAT", "timestamp": time.time()})
    except WebSocketDisconnect:
        active_websockets.remove(websocket)
        logger.info("Client disconnected from /ws/live-events")
