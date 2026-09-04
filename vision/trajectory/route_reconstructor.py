"""
Spatio-Temporal Vehicle Route Reconstruction and Camera Topology Engine for YUTA.

Derived and adapted from Cam-Traj-Rec (Tsinghua University - KDD 2022 - MIT License).
Provides:
1. Camera topology graph and road network modeling (distances, speed limits, connectivity).
2. Gaussian travel-time likelihood estimation.
3. K-shortest paths route recovery between sparse camera observations.
4. GeoJSON route synthesis for GIS investigation mapping.
"""

import math
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import networkx as nx

logger = logging.getLogger(__name__)


@dataclass
class CameraNode:
    """Represents a CCTV camera location in the urban road network."""
    camera_id: str
    name: str
    lat: float
    lon: float
    road_id: Optional[str] = None


@dataclass
class RouteSegment:
    """A segment of a reconstructed vehicle route between two cameras."""
    from_camera: str
    to_camera: str
    distance_meters: float
    travel_time_sec: float
    implied_speed_kmh: float
    likelihood: float
    path_nodes: List[str] = field(default_factory=list)
    path_coordinates: List[Tuple[float, float]] = field(default_factory=list)  # (lat, lon)


@dataclass
class ReconstructedRoute:
    """Full end-to-end spatio-temporal route reconstructed from camera sightings."""
    global_vehicle_id: str
    plate_number: Optional[str]
    start_time: float
    end_time: float
    total_distance_meters: float
    total_duration_sec: float
    average_speed_kmh: float
    overall_confidence: float
    segments: List[RouteSegment] = field(default_factory=list)
    full_path_coordinates: List[Tuple[float, float]] = field(default_factory=list)
    geojson_feature: Dict[str, Any] = field(default_factory=dict)


class CameraTopologyGraph:
    """
    Graph representing camera network topology, road connections, and travel-time priors.
    """

    def __init__(self, default_speed_kmh: float = 40.0, sigma_ratio: float = 0.4):
        self.default_speed_kmh = default_speed_kmh
        self.sigma_ratio = sigma_ratio
        self.cameras: Dict[str, CameraNode] = {}
        self.graph = nx.DiGraph()

    def add_camera(self, camera_id: str, name: str, lat: float, lon: float, road_id: Optional[str] = None):
        """Adds a camera node to the topology."""
        node = CameraNode(camera_id=camera_id, name=name, lat=lat, lon=lon, road_id=road_id)
        self.cameras[camera_id] = node
        self.graph.add_node(camera_id, lat=lat, lon=lon, name=name)

    def add_road_connection(
        self,
        from_cam: str,
        to_cam: str,
        distance_meters: Optional[float] = None,
        speed_limit_kmh: Optional[float] = None,
        intermediate_coords: Optional[List[Tuple[float, float]]] = None,
    ):
        """Adds a directed road segment connecting two cameras."""
        if from_cam not in self.cameras or to_cam not in self.cameras:
            raise ValueError(f"Both cameras ({from_cam}, {to_cam}) must exist in topology.")

        c1 = self.cameras[from_cam]
        c2 = self.cameras[to_cam]

        if distance_meters is None:
            # Haversine distance estimate
            distance_meters = self.haversine_distance(c1.lat, c1.lon, c2.lat, c2.lon)

        expected_speed = speed_limit_kmh if speed_limit_kmh is not None else self.default_speed_kmh
        expected_time_sec = distance_meters / (max(5.0, expected_speed) * 1000.0 / 3600.0)

        coords = [ (c1.lat, c1.lon) ]
        if intermediate_coords:
            coords.extend(intermediate_coords)
        coords.append((c2.lat, c2.lon))

        self.graph.add_edge(
            from_cam,
            to_cam,
            weight=distance_meters,
            distance_meters=distance_meters,
            expected_speed_kmh=expected_speed,
            expected_time_sec=expected_time_sec,
            coordinates=coords,
        )

    @staticmethod
    def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculates great-circle distance between two GPS coordinates in meters."""
        R = 6371000.0  # Earth radius in meters
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)

        a = math.sin(delta_phi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
        c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
        return R * c

    def compute_travel_likelihood(self, from_cam: str, to_cam: str, actual_time_sec: float) -> Tuple[float, float, float]:
        """
        Computes Gaussian travel-time likelihood P(actual_time | road_prior):
        Returns: (likelihood, distance_meters, implied_speed_kmh)
        """
        if not self.graph.has_edge(from_cam, to_cam):
            # Compute shortest path if direct edge doesn't exist
            try:
                path = nx.shortest_path(self.graph, from_cam, to_cam, weight="weight")
                total_dist = 0.0
                total_expected_time = 0.0
                for u, v in zip(path, path[1:]):
                    edge = self.graph[u][v]
                    total_dist += edge["distance_meters"]
                    total_expected_time += edge["expected_time_sec"]
            except nx.NetworkXNoPath:
                return 0.0, 0.0, 0.0
        else:
            edge = self.graph[from_cam][to_cam]
            total_dist = edge["distance_meters"]
            total_expected_time = edge["expected_time_sec"]

        if actual_time_sec <= 0.1:
            return 0.0, total_dist, 999.0

        # Implied speed
        implied_speed_mps = total_dist / actual_time_sec
        implied_speed_kmh = implied_speed_mps * 3.6

        expected_speed_mps = total_dist / max(0.1, total_expected_time)
        sigma = max(2.0, expected_speed_mps * self.sigma_ratio)

        # Gaussian likelihood
        exponent = -((implied_speed_mps - expected_speed_mps) ** 2) / (2.0 * (sigma ** 2))
        likelihood = math.exp(max(-50.0, exponent))

        return float(likelihood), float(total_dist), float(implied_speed_kmh)

    def reconstruct_route(
        self,
        sightings: List[Tuple[str, float]],  # List of (camera_id, timestamp) in chronological order
        global_vehicle_id: str = "GLOBAL-VEH",
        plate_number: Optional[str] = None,
    ) -> Optional[ReconstructedRoute]:
        """
        Reconstructs the full spatio-temporal route from sparse camera sightings.
        """
        if len(sightings) < 2:
            return None

        # Ensure sorted by timestamp
        sorted_sightings = sorted(sightings, key=lambda s: s[1])
        segments: List[RouteSegment] = []
        all_coords: List[Tuple[float, float]] = []

        total_distance = 0.0
        segment_confidences = []

        for i in range(len(sorted_sightings) - 1):
            cam_a, t_a = sorted_sightings[i]
            cam_b, t_b = sorted_sightings[i + 1]

            if cam_a == cam_b:
                continue

            dt = max(0.1, t_b - t_a)
            likelihood, dist, speed = self.compute_travel_likelihood(cam_a, cam_b, dt)

            # Retrieve coordinates
            try:
                path = nx.shortest_path(self.graph, cam_a, cam_b, weight="weight")
                seg_coords: List[Tuple[float, float]] = []
                for u, v in zip(path, path[1:]):
                    edge = self.graph[u][v]
                    seg_coords.extend(edge.get("coordinates", []))
            except nx.NetworkXNoPath:
                c1 = self.cameras.get(cam_a)
                c2 = self.cameras.get(cam_b)
                seg_coords = [(c1.lat, c1.lon), (c2.lat, c2.lon)] if (c1 and c2) else []
                path = [cam_a, cam_b]

            segments.append(
                RouteSegment(
                    from_camera=cam_a,
                    to_camera=cam_b,
                    distance_meters=dist,
                    travel_time_sec=dt,
                    implied_speed_kmh=speed,
                    likelihood=likelihood,
                    path_nodes=path,
                    path_coordinates=seg_coords,
                )
            )
            total_distance += dist
            segment_confidences.append(likelihood)
            if seg_coords:
                all_coords.extend(seg_coords)

        if not segments:
            return None

        total_duration = sorted_sightings[-1][1] - sorted_sightings[0][1]
        avg_speed = (total_distance / max(0.1, total_duration)) * 3.6
        overall_conf = float(sum(segment_confidences) / len(segment_confidences)) if segment_confidences else 0.5

        # Format GeoJSON Feature
        geojson_feature = {
            "type": "Feature",
            "properties": {
                "vehicle_id": global_vehicle_id,
                "plate_number": plate_number,
                "total_distance_m": round(total_distance, 1),
                "duration_sec": round(total_duration, 1),
                "avg_speed_kmh": round(avg_speed, 1),
                "confidence": round(overall_conf, 3),
            },
            "geometry": {
                "type": "LineString",
                "coordinates": [[lon, lat] for lat, lon in all_coords],
            },
        }

        return ReconstructedRoute(
            global_vehicle_id=global_vehicle_id,
            plate_number=plate_number,
            start_time=sorted_sightings[0][1],
            end_time=sorted_sightings[-1][1],
            total_distance_meters=total_distance,
            total_duration_sec=total_duration,
            average_speed_kmh=avg_speed,
            overall_confidence=overall_conf,
            segments=segments,
            full_path_coordinates=all_coords,
            geojson_feature=geojson_feature,
        )
