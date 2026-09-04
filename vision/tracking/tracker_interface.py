"""
Tracking and Detection Interfaces for YUTA.

Derived from TrackStudio (Apache 2.0) and adapted for multi-camera vehicle intelligence,
Sentinel CCTV streams, and multi-modal evidence graphs.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any
import numpy as np


@dataclass
class Detection:
    """
    Object detection result from an image frame.
    bbox format: [x1, y1, x2, y2]
    """
    bbox: Tuple[float, float, float, float]
    confidence: float
    class_name: str
    class_id: int
    camera_id: str
    timestamp: float
    feature_vector: Optional[np.ndarray] = None


@dataclass
class LocalTrack:
    """
    Single-camera local track maintaining temporal continuity across frames.
    """
    track_id: str
    camera_id: str
    bbox: Tuple[float, float, float, float]
    confidence: float
    age: int = 1
    hits: int = 1
    time_since_update: int = 0
    state: str = "active"  # active, lost, finished
    bev_coord: Optional[Tuple[float, float]] = None
    feature_vector: Optional[np.ndarray] = None
    plate_text: Optional[str] = None
    plate_confidence: float = 0.0
    trajectory: List[Tuple[float, float, float]] = field(default_factory=list)  # (x, y, timestamp)


@dataclass
class BEVTrack:
    """
    Track projected into normalized or metric Bird's Eye View (BEV) coordinates.
    """
    local_track_id: str
    camera_id: str
    bev_x: float
    bev_y: float
    confidence: float
    global_id: Optional[str] = None
    feature_vector: Optional[np.ndarray] = None
    timestamp: float = 0.0
    trajectory: List[Tuple[float, float, float]] = field(default_factory=list)


@dataclass
class GlobalTrack:
    """
    Cross-camera global vehicle/object identity with multi-modal evidence.
    """
    global_id: str
    camera_tracks: Dict[str, str] = field(default_factory=dict)  # camera_id -> local_track_id
    first_seen: float = 0.0
    last_seen: float = 0.0
    positions: List[Tuple[float, float, float, str]] = field(default_factory=list)  # (bev_x, bev_y, timestamp, camera_id)
    appearance_features: Optional[np.ndarray] = None
    vehicle_class: str = "unknown"
    vehicle_color: str = "unknown"
    plate_number: Optional[str] = None
    plate_confidence: float = 0.0
    evidence_nodes: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 1.0
