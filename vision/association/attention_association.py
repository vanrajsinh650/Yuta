"""
Attention-Based Global Association Module for YUTA.

Derived and adapted from GMT (FoxCanned/GMT - CVPR 2026 - Apache 2.0).
Computes learned feature projections and temperature-scaled affinity weights
between tracklet query embeddings and candidate key embeddings across cameras.
"""

from typing import Optional, Tuple
import numpy as np


class AttentionAssociationHead:
    """
    Computes cross-attention affinity matrix between query tracks and key candidates.
    Can operate in pure NumPy for CPU inference or with learned projection matrices.
    """

    def __init__(self, feature_dim: int = 128, temperature: Optional[float] = None):
        self.feature_dim = feature_dim
        self.temperature = temperature if temperature is not None else np.sqrt(feature_dim)
        # Default orthogonal projection weights (identity-like mapping)
        self.W_q = np.eye(feature_dim, dtype=np.float32)
        self.W_k = np.eye(feature_dim, dtype=np.float32)

    def set_weights(self, w_q: np.ndarray, w_k: np.ndarray):
        """Sets custom or fine-tuned linear projection matrices."""
        self.W_q = w_q.astype(np.float32)
        self.W_k = w_k.astype(np.float32)

    def compute_affinity(
        self, queries: np.ndarray, keys: np.ndarray
    ) -> np.ndarray:
        """
        Computes pairwise attention weights between M queries and N keys.
        queries: (M, feature_dim)
        keys: (N, feature_dim)
        Returns: (M, N) softmax affinity matrix.
        """
        if len(queries) == 0 or len(keys) == 0:
            return np.zeros((len(queries), len(keys)), dtype=np.float32)

        # Linear projections
        Q = queries @ self.W_q  # (M, D)
        K = keys @ self.W_k    # (N, D)

        # Scaled dot product
        scores = (Q @ K.T) / self.temperature  # (M, N)

        # Row-wise softmax with numerical stability
        exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        affinity = exp_scores / (np.sum(exp_scores, axis=-1, keepdims=True) + 1e-8)

        return affinity
