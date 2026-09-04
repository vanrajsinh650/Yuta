"""
Multi-Camera Evidence Graph Engine for YUTA.

Derived from TAU-Agent (AI City 2026 - MIT License) and adapted for Gujarat Police investigations.
Represents multi-modal vehicle sightings, cross-camera transitions, and explainable graph reasoning:
- Every cross-camera link has concrete evidence (plate match, appearance cosine, road travel-time likelihood).
- Traces vehicle origins and destinations: "Where was this vehicle before Camera X?"
- Vector similarity search for finding visually similar suspect vehicles.
- Anomaly detection for speed violations and impossible transitions.
"""

import math
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EvidenceNode:
    """A verified sighting of a vehicle in a specific camera feed."""
    node_id: str
    global_vehicle_id: str
    camera_id: str
    camera_name: str
    timestamp: float
    bbox: Tuple[float, float, float, float]  # [x1, y1, x2, y2]
    plate_number: Optional[str] = None
    plate_confidence: float = 0.0
    vehicle_class: str = "car"
    vehicle_color: str = "unknown"
    appearance_embedding: Optional[np.ndarray] = None
    crop_image_uri: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvidenceEdge:
    """An explainable transition link between two camera sightings."""
    source_node_id: str
    target_node_id: str
    from_camera: str
    to_camera: str
    time_gap_sec: float
    spatial_dist_meters: float
    implied_speed_kmh: float
    appearance_similarity: float
    plate_match: Optional[bool]
    road_likelihood: float
    confidence: float
    reason: str


class EvidenceGraph:
    """
    Directed Evidence Graph connecting all multi-camera observations into a unified intelligence network.
    """

    def __init__(self):
        self.nodes: Dict[str, EvidenceNode] = {}
        self.edges: List[EvidenceEdge] = []
        # Index: global_vehicle_id -> list of node_ids
        self.vehicle_index: Dict[str, List[str]] = {}
        # Index: plate_number -> list of global_vehicle_ids
        self.plate_index: Dict[str, set] = {}

    def add_node(self, node: EvidenceNode):
        self.nodes[node.node_id] = node
        if node.global_vehicle_id not in self.vehicle_index:
            self.vehicle_index[node.global_vehicle_id] = []
        self.vehicle_index[node.global_vehicle_id].append(node.node_id)

        if node.plate_number:
            clean_plate = node.plate_number.replace(" ", "").replace("-", "").upper()
            if clean_plate not in self.plate_index:
                self.plate_index[clean_plate] = set()
            self.plate_index[clean_plate].add(node.global_vehicle_id)

    def add_edge(self, edge: EvidenceEdge):
        self.edges.append(edge)

    def get_vehicle_nodes(self, global_vehicle_id: str) -> List[EvidenceNode]:
        node_ids = self.vehicle_index.get(global_vehicle_id, [])
        nodes = [self.nodes[nid] for nid in node_ids]
        nodes.sort(key=lambda n: n.timestamp)
        return nodes

    def get_vehicle_edges(self, global_vehicle_id: str) -> List[EvidenceEdge]:
        node_ids = set(self.vehicle_index.get(global_vehicle_id, []))
        return [e for e in self.edges if e.source_node_id in node_ids and e.target_node_id in node_ids]

    def search_by_plate(self, plate_query: str) -> List[str]:
        """Finds all global vehicle IDs associated with a plate (exact or partial)."""
        clean_q = plate_query.replace(" ", "").replace("-", "").upper()
        matches = set()
        for indexed_plate, gids in self.plate_index.items():
            if clean_q in indexed_plate:
                matches.update(gids)
        return list(matches)

    def get_predecessor_sightings(self, global_vehicle_id: str, camera_id: str) -> List[EvidenceNode]:
        """
        Answers: 'Where was this vehicle before Camera X?'
        Returns all chronological sightings of the vehicle prior to reaching camera_id.
        """
        all_sightings = self.get_vehicle_nodes(global_vehicle_id)
        # Find first sighting in target camera
        target_time = None
        for s in all_sightings:
            if s.camera_id == camera_id:
                target_time = s.timestamp
                break

        if target_time is None:
            return []

        return [s for s in all_sightings if s.timestamp < target_time]

    def find_similar_vehicles(self, target_global_id: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Finds vehicles with matching visual appearance embeddings.
        Returns: list of (global_vehicle_id, cosine_similarity)
        """
        target_nodes = self.get_vehicle_nodes(target_global_id)
        target_embs = [n.appearance_embedding for n in target_nodes if n.appearance_embedding is not None]
        if not target_embs:
            return []

        avg_target_emb = np.mean(target_embs, axis=0)
        norm_target = np.linalg.norm(avg_target_emb)
        if norm_target < 1e-6:
            return []
        avg_target_emb = avg_target_emb / norm_target

        scores = []
        for gid, node_ids in self.vehicle_index.items():
            if gid == target_global_id:
                continue
            cand_nodes = [self.nodes[nid] for nid in node_ids]
            cand_embs = [n.appearance_embedding for n in cand_nodes if n.appearance_embedding is not None]
            if not cand_embs:
                continue

            avg_cand_emb = np.mean(cand_embs, axis=0)
            norm_cand = np.linalg.norm(avg_cand_emb)
            if norm_cand < 1e-6:
                continue
            avg_cand_emb = avg_cand_emb / norm_cand

            cos_sim = float(np.dot(avg_target_emb, avg_cand_emb))
            scores.append((gid, cos_sim))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def detect_anomalies(self, max_speed_kmh: float = 120.0) -> List[Dict[str, Any]]:
        """
        Detects suspicious movements, speed violations, or impossible transitions across cameras.
        """
        anomalies = []
        for edge in self.edges:
            if edge.implied_speed_kmh > max_speed_kmh:
                anomalies.append({
                    "type": "excessive_speed",
                    "severity": "high",
                    "from_camera": edge.from_camera,
                    "to_camera": edge.to_camera,
                    "implied_speed_kmh": round(edge.implied_speed_kmh, 1),
                    "time_gap_sec": round(edge.time_gap_sec, 1),
                    "distance_m": round(edge.spatial_dist_meters, 1),
                    "source_node": edge.source_node_id,
                    "target_node": edge.target_node_id,
                })
        return anomalies
