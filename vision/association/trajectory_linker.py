"""
Long-Term Global Trajectory Linker for YUTA.

Derived and adapted from ZIOVISION AIC2025 Track 1 (1st Place MCMT solution).
Provides:
1. Multi-modal gating matrix (dominant class, spatio-temporal boundary gating, ReID similarity, ANPR plate evidence).
2. Hungarian algorithm-based global assignment (scipy linear_sum_assignment).
3. Iterative hierarchical merging across camera handoffs and temporal occlusion gaps.
4. Explainable link evidence generation for YUTA's Evidence Graph.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)


from vision.association.attention_association import AttentionAssociationHead


@dataclass
class Tracklet:
    """
    Continuous single-camera tracklet representation for global linking.
    """
    tracklet_id: str
    camera_id: str
    dominant_class: str
    first_timestamp: float
    last_timestamp: float
    first_pos: Tuple[float, float]  # Entry point (x, y) in ground/BEV or image
    last_pos: Tuple[float, float]   # Exit point (x, y)
    mean_feature: Optional[np.ndarray] = None
    plate_text: Optional[str] = None
    plate_confidence: float = 0.0
    detection_count: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrajectoryLink:
    """
    Explainable link connecting two tracklets across space/time.
    """
    parent_id: str
    child_id: str
    cost: float
    time_gap_sec: float
    spatial_dist: float
    appearance_sim: float
    plate_match: Optional[bool]
    reason: str


class GlobalTrajectoryLinker:
    """
    Iterative Hungarian trajectory linking engine for multi-camera vehicle tracking.
    """

    def __init__(
        self,
        min_tracklet_detections: int = 2,
        max_time_gap_sec: float = 60.0,
        max_spatial_distance: float = 150.0,
        appearance_similarity_threshold: float = 0.65,
        max_speed_units_per_sec: float = 30.0,  # Physical velocity constraint
        attention_head: Optional[AttentionAssociationHead] = None,
    ):
        self.min_tracklet_detections = min_tracklet_detections
        self.max_time_gap_sec = max_time_gap_sec
        self.max_spatial_distance = max_spatial_distance
        self.appearance_similarity_threshold = appearance_similarity_threshold
        self.max_speed_units_per_sec = max_speed_units_per_sec
        self.attention_head = attention_head

    def _compute_appearance_similarity(self, f1: Optional[np.ndarray], f2: Optional[np.ndarray]) -> float:
        if f1 is None or f2 is None:
            return 0.5  # Neutral fallback when appearance feature is missing
        if self.attention_head is not None:
            try:
                q = f1.reshape(1, -1)
                k = f2.reshape(1, -1)
                aff = self.attention_head.compute_affinity(q, k)
                return float(aff[0, 0])
            except Exception:
                pass
        return self._cosine_similarity(f1, f2)

    def _cosine_similarity(self, f1: Optional[np.ndarray], f2: Optional[np.ndarray]) -> float:
        if f1 is None or f2 is None:
            return 0.5  # Neutral fallback when appearance feature is missing
        n1 = np.linalg.norm(f1)
        n2 = np.linalg.norm(f2)
        if n1 < 1e-6 or n2 < 1e-6:
            return 0.5
        return float(np.dot(f1, f2) / (n1 * n2))

    def link_tracklets(
        self,
        tracklets: List[Tracklet],
    ) -> Tuple[Dict[str, List[Tracklet]], List[TrajectoryLink]]:
        """
        Iteratively links tracklets into global vehicle trajectories.
        Returns:
            - Dict mapping global_id -> list of Tracklets belonging to that identity
            - List of TrajectoryLink evidence records
        """
        # Filter spurious single-frame tracklets
        valid_tracklets = [t for t in tracklets if t.detection_count >= self.min_tracklet_detections]
        if not valid_tracklets:
            return {}, []

        # Maintain clusters of tracklets (each cluster starts with 1 tracklet)
        # cluster_id -> list of tracklets
        clusters: Dict[str, List[Tracklet]] = {
            t.tracklet_id: [t] for t in valid_tracklets
        }
        all_links: List[TrajectoryLink] = []

        max_iterations = 10
        iteration = 0

        while iteration < max_iterations:
            iteration += 1
            cluster_keys = list(clusters.keys())
            num_clusters = len(cluster_keys)
            if num_clusters <= 1:
                break

            # Build cost matrix between clusters
            # Cost represents mismatch: lower is better; thresholded out values get large penalty
            INF_COST = 1e5
            cost_matrix = np.full((num_clusters, num_clusters), INF_COST, dtype=np.float32)
            candidate_links: Dict[Tuple[int, int], TrajectoryLink] = {}

            for i in range(num_clusters):
                c1 = clusters[cluster_keys[i]]
                # Find exit endpoint of cluster 1
                c1_last_time = max(t.last_timestamp for t in c1)
                t1_exit = max(c1, key=lambda t: t.last_timestamp)

                for j in range(num_clusters):
                    if i == j:
                        continue
                    c2 = clusters[cluster_keys[j]]
                    # Find entry endpoint of cluster 2
                    c2_first_time = min(t.first_timestamp for t in c2)
                    t2_entry = min(c2, key=lambda t: t.first_timestamp)

                    # Temporal causality gating: cluster 1 must occur BEFORE cluster 2
                    time_gap = c2_first_time - c1_last_time
                    if time_gap < 0.0 or time_gap > self.max_time_gap_sec:
                        continue

                    # Class gating
                    if t1_exit.dominant_class != t2_entry.dominant_class and (
                        t1_exit.dominant_class != "unknown" and t2_entry.dominant_class != "unknown"
                    ):
                        continue

                    # ANPR Plate gating
                    plate_match = None
                    if t1_exit.plate_text and t2_entry.plate_text:
                        if t1_exit.plate_text == t2_entry.plate_text:
                            plate_match = True
                        elif t1_exit.plate_confidence > 0.85 and t2_entry.plate_confidence > 0.85:
                            # Contradicting plates with high confidence: reject association
                            continue
                        else:
                            plate_match = False

                    # Spatial gating: exit point of c1 to entry point of c2
                    dx = t2_entry.first_pos[0] - t1_exit.last_pos[0]
                    dy = t2_entry.first_pos[1] - t1_exit.last_pos[1]
                    dist = float(np.sqrt(dx * dx + dy * dy))

                    # Velocity plausibility check
                    if time_gap > 0.1:
                        implied_speed = dist / time_gap
                        if implied_speed > self.max_speed_units_per_sec and not plate_match:
                            continue

                    # Appearance similarity check (cosine or GMT cross-attention affinity)
                    app_sim = self._compute_appearance_similarity(t1_exit.mean_feature, t2_entry.mean_feature)
                    if (
                        t1_exit.mean_feature is not None
                        and t2_entry.mean_feature is not None
                        and app_sim < self.appearance_similarity_threshold
                        and not plate_match
                    ):
                        continue

                    # Compute composite cost
                    # Base cost from time and distance
                    cost = (dist / (self.max_spatial_distance + 1e-5)) * 0.4 + (time_gap / self.max_time_gap_sec) * 0.3
                    # Bonus for strong appearance similarity
                    cost += (1.0 - app_sim) * 0.3

                    # Strong bonus for matching ANPR plate
                    if plate_match is True:
                        cost = max(0.01, cost * 0.1)

                    cost_matrix[i, j] = cost
                    reason = "anpr_and_reid" if plate_match else "spatiotemporal_reid"
                    candidate_links[(i, j)] = TrajectoryLink(
                        parent_id=cluster_keys[i],
                        child_id=cluster_keys[j],
                        cost=cost,
                        time_gap_sec=time_gap,
                        spatial_dist=dist,
                        appearance_sim=app_sim,
                        plate_match=plate_match,
                        reason=reason,
                    )

            # Solve assignment via Hungarian algorithm
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            merged_in_this_step = 0

            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] < 1.0:  # Accepted threshold
                    parent_key = cluster_keys[r]
                    child_key = cluster_keys[c]
                    if parent_key in clusters and child_key in clusters and parent_key != child_key:
                        link_info = candidate_links[(r, c)]
                        all_links.append(link_info)
                        # Merge child into parent
                        clusters[parent_key].extend(clusters[child_key])
                        # Update running feature centroid of merged cluster
                        valid_feats = [t.mean_feature for t in clusters[parent_key] if t.mean_feature is not None]
                        if valid_feats:
                            centroid = np.mean(valid_feats, axis=0)
                            for t in clusters[parent_key]:
                                t.mean_feature = centroid
                        del clusters[child_key]
                        merged_in_this_step += 1

            if merged_in_this_step == 0:
                break

        # Re-key global trajectories with clean Global IDs
        final_trajectories: Dict[str, List[Tracklet]] = {}
        for idx, (original_key, track_list) in enumerate(clusters.items(), start=1):
            gid = f"GLOBAL-VEH-{idx:04d}"
            # Sort tracklets chronologically
            track_list.sort(key=lambda t: t.first_timestamp)
            final_trajectories[gid] = track_list

        return final_trajectories, all_links
