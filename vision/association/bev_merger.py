"""
Multi-Camera BEV Cluster Association Engine for YUTA.

Derived from TrackStudio (Apache 2.0) BEVClusterMerger.
Associates local single-camera tracks into unified Global Vehicle Identities
using 2D ground plane (BEV) spatial proximity, appearance cosine similarity,
and temporal gating.
"""

import logging
import time
from typing import Dict, List, Optional, Set, Tuple
import numpy as np

from vision.tracking.tracker_interface import BEVTrack, GlobalTrack

logger = logging.getLogger(__name__)


class BEVClusterMerger:
    """
    Multi-camera track association using Bird's Eye View clustering and appearance fusion.
    """

    def __init__(
        self,
        spatial_threshold: float = 60.0,  # Max BEV ground distance in pixels/units
        appearance_threshold: float = 0.35,  # Max cosine distance (1 - sim)
        track_timeout_sec: float = 12.0,  # Keep global track alive across camera handoffs
    ):
        self.spatial_threshold = spatial_threshold
        self.appearance_threshold = appearance_threshold
        self.track_timeout_sec = track_timeout_sec

        self.global_tracks: Dict[str, GlobalTrack] = {}
        self.track_id_mapping: Dict[Tuple[str, str], str] = {}  # (camera_id, local_track_id) -> global_id
        self._next_id_counter = 1

    def _generate_global_id(self) -> str:
        gid = f"VEH-{self._next_id_counter:04d}"
        self._next_id_counter += 1
        return gid

    def cleanup_expired_tracks(self, current_time: float):
        """Removes global tracks that have not been observed in any camera for track_timeout_sec."""
        expired_ids = [
            gid for gid, track in self.global_tracks.items()
            if (current_time - track.last_seen) > self.track_timeout_sec
        ]
        for gid in expired_ids:
            # Clean reverse mappings
            self.track_id_mapping = {
                key: mapped_gid for key, mapped_gid in self.track_id_mapping.items()
                if mapped_gid != gid
            }
            del self.global_tracks[gid]

    def _cosine_distance(self, feat1: Optional[np.ndarray], feat2: Optional[np.ndarray]) -> float:
        if feat1 is None or feat2 is None:
            return 0.0  # Unknown appearance: rely purely on spatial & temporal gating
        norm1 = np.linalg.norm(feat1)
        norm2 = np.linalg.norm(feat2)
        if norm1 < 1e-6 or norm2 < 1e-6:
            return 0.0
        cosine_sim = float(np.dot(feat1, feat2) / (norm1 * norm2))
        return max(0.0, 1.0 - cosine_sim)

    def merge_tracks(
        self,
        bev_tracks: List[BEVTrack],
        timestamp: Optional[float] = None,
    ) -> List[BEVTrack]:
        """
        Clusters simultaneous tracks from different cameras and assigns consistent global IDs.
        """
        now = timestamp if timestamp is not None else time.time()
        self.cleanup_expired_tracks(now)

        if not bev_tracks:
            return []

        n = len(bev_tracks)
        adj_matrix = np.zeros((n, n), dtype=bool)

        # Build adjacency graph between candidate tracks
        for i in range(n):
            for j in range(i + 1, n):
                t1 = bev_tracks[i]
                t2 = bev_tracks[j]

                # Tracks from the same camera cannot be the same physical object
                if t1.camera_id == t2.camera_id:
                    continue

                # 1. Spatial distance in BEV space
                dx = t1.bev_x - t2.bev_x
                dy = t1.bev_y - t2.bev_y
                dist = np.sqrt(dx * dx + dy * dy)
                if dist > self.spatial_threshold:
                    continue

                # 2. Appearance cosine distance check
                app_dist = self._cosine_distance(t1.feature_vector, t2.feature_vector)
                if app_dist > self.appearance_threshold and (t1.feature_vector is not None and t2.feature_vector is not None):
                    continue

                adj_matrix[i, j] = adj_matrix[j, i] = True

        # Find connected components via DFS
        visited = [False] * n
        clusters: List[List[BEVTrack]] = []

        for i in range(n):
            if not visited[i]:
                cluster: List[BEVTrack] = []
                stack = [i]
                visited[i] = True
                while stack:
                    curr = stack.pop()
                    cluster.append(bev_tracks[curr])
                    for neighbor in range(n):
                        if adj_matrix[curr, neighbor] and not visited[neighbor]:
                            visited[neighbor] = True
                            stack.append(neighbor)
                clusters.append(cluster)

        # Assign or resolve global IDs for each cluster
        result_tracks: List[BEVTrack] = []

        for cluster in clusters:
            # Check if any member in the cluster already has a known global track
            existing_global_ids: Set[str] = set()
            for t in cluster:
                key = (t.camera_id, t.local_track_id)
                if key in self.track_id_mapping:
                    existing_global_ids.add(self.track_id_mapping[key])

            if existing_global_ids:
                # Use the oldest or most established global ID
                target_gid = min(existing_global_ids)
            else:
                target_gid = self._generate_global_id()
                self.global_tracks[target_gid] = GlobalTrack(
                    global_id=target_gid,
                    first_seen=now,
                    last_seen=now,
                )

            global_entry = self.global_tracks[target_gid]
            global_entry.last_seen = now

            for t in cluster:
                key = (t.camera_id, t.local_track_id)
                self.track_id_mapping[key] = target_gid
                global_entry.camera_tracks[t.camera_id] = t.local_track_id
                global_entry.positions.append((t.bev_x, t.bev_y, now, t.camera_id))

                # Update running appearance feature average if available
                if t.feature_vector is not None:
                    if global_entry.appearance_features is None:
                        global_entry.appearance_features = t.feature_vector.copy()
                    else:
                        global_entry.appearance_features = 0.8 * global_entry.appearance_features + 0.2 * t.feature_vector

                t.global_id = target_gid
                result_tracks.append(t)

        return result_tracks
