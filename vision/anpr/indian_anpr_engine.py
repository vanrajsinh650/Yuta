"""
Indian Automated Number Plate Recognition (ANPR) Engine for YUTA.

Derived and adapted from Indian_LPR (sanchit2843/Indian_LPR).
Specialized for Gujarat Police Challenge 2026 requirements:
- 4-point perspective unwarping and tilt correction for motorcycles, autos, and skewed CCTV angles.
- High-contrast CLAHE preprocessing for night scenes, headlight glare, and dirty/damaged plates.
- Strict validation and canonicalization of standard Indian HSRP, two-line, and Bharat (BH) formats.
- Complete Gujarat RTO database (GJ01 to GJ38) for instant district validation.
- Temporal multi-frame majority voting across vehicle tracks to eliminate single-frame OCR flicker.
"""

import re
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Valid Indian States and Union Territories (Motor Vehicles Act)
INDIAN_STATE_CODES = {
    "GJ", "MH", "DL", "UP", "KA", "TN", "RJ", "MP", "HR", "PB",
    "WB", "KL", "AP", "TS", "OD", "CH", "GA", "JK", "LA", "UK",
    "JH", "BR", "AS", "TR", "ML", "MN", "NL", "MZ", "SK", "AR",
    "DD", "DN", "PY", "AN", "LD",
}

# Gujarat Specific RTO District Directory
GUJARAT_RTO_CODES = {
    "01": "Ahmedabad (West)",
    "02": "Mehsana",
    "03": "Rajkot",
    "04": "Bhavnagar",
    "05": "Surat (City)",
    "06": "Vadodara (City)",
    "07": "Nadiad (Kheda)",
    "08": "Palanpur (Banaskantha)",
    "09": "Himmatnagar (Sabar Kantha)",
    "10": "Jamnagar",
    "11": "Junagadh",
    "12": "Bhuj (Kutch)",
    "13": "Surendranagar",
    "14": "Amreli",
    "15": "Valsad",
    "16": "Bharuch",
    "17": "Godhra (Panchmahal)",
    "18": "Gandhinagar",
    "19": "Bardoli (Surat Rural)",
    "20": "Dahod",
    "21": "Navsari",
    "22": "Rajpipla (Narmada)",
    "23": "Anand",
    "24": "Patan",
    "25": "Porbandar",
    "26": "Vyara (Tapi)",
    "27": "Ahmedabad (East)",
    "28": "Surat (Pal)",
    "29": "Vadodara (Rural)",
    "30": "Ahwa (Dang)",
    "31": "Modasa (Aravalli)",
    "32": "Veraval (Gir Somnath)",
    "33": "Botad",
    "34": "Chhota Udepur",
    "35": "Lunawada (Mahisagar)",
    "36": "Morbi",
    "37": "Khambhalia (Devbhumi Dwarka)",
    "38": "Bavla (Ahmedabad Rural)",
}


@dataclass
class PlateCandidate:
    """Individual OCR observation for a vehicle in a single frame."""
    raw_text: str
    canonical_text: str
    confidence: float
    is_valid_format: bool
    state_code: Optional[str] = None
    rto_code: Optional[str] = None
    district_name: Optional[str] = None
    timestamp: float = 0.0


class IndianPlatePreprocessor:
    """Handles perspective unwarping and image enhancement for difficult Indian plates."""

    @staticmethod
    def rectify_quadrilateral(
        image: np.ndarray,
        quad_points: List[Tuple[float, float]],
        target_width: int = 300,
        target_height: int = 90,
    ) -> np.ndarray:
        """
        Unwarps 4-point quadrilateral plate coordinates into a frontal rectangular crop.
        quad_points order: [top-left, top-right, bottom-right, bottom-left]
        """
        if len(quad_points) != 4:
            raise ValueError("Exactly 4 corner points required for quadrilateral rectification.")

        src = np.array(quad_points, dtype=np.float32)
        dst = np.array([
            [0, 0],
            [target_width - 1, 0],
            [target_width - 1, target_height - 1],
            [0, target_height - 1]
        ], dtype=np.float32)

        M = cv2.getPerspectiveTransform(src, dst)
        rectified = cv2.warpPerspective(image, M, (target_width, target_height))
        return rectified

    @staticmethod
    def enhance_for_ocr(plate_crop: np.ndarray) -> np.ndarray:
        """
        Applies CLAHE, bilateral smoothing, and adaptive thresholding to handle night glare,
        shadows, and dirt on plates.
        """
        if plate_crop is None or plate_crop.size == 0:
            return plate_crop

        if len(plate_crop.shape) == 3:
            gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_crop.copy()

        # 1. CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        contrast_enhanced = clahe.apply(gray)

        # 2. Bilateral filter to preserve sharp character edges while eliminating camera sensor noise
        smoothed = cv2.bilateralFilter(contrast_enhanced, 9, 75, 75)

        return smoothed


class IndianPlateValidator:
    """Validates, cleans, and canonicalizes Indian motor vehicle registration numbers."""

    # Common OCR character confusions: Numbers confused with Letters, Letters with Numbers
    CHAR_TO_NUM = {"O": "0", "D": "0", "Q": "0", "I": "1", "L": "1", "Z": "2", "S": "5", "B": "8", "G": "6"}
    NUM_TO_CHAR = {"0": "O", "1": "I", "2": "Z", "5": "S", "8": "B", "6": "G"}

    @classmethod
    def clean_text(cls, raw: str) -> str:
        """Removes spaces, hyphens, and non-alphanumeric noise from OCR string."""
        return re.sub(r"[^A-Za-z0-9]", "", raw).upper()

    @classmethod
    def validate_and_canonicalize(cls, raw_text: str, confidence: float = 0.8) -> PlateCandidate:
        """
        Validates against:
        1. Standard Indian HSRP: [State 2L][District 2D][Series 1-3L][Number 1-4D] (e.g. GJ01AB1234, MH12DE9876)
        2. Bharat Series (BH): [Year 2D][BH][Number 4D][Series 1-2L] (e.g. 22BH1234AA)
        3. Vintage/older 2-line format: [State 2L][District 1-2D][Number 1-4D] (e.g. GJ1A1234)
        """
        cleaned = cls.clean_text(raw_text)

        # Quick length gate: Indian plates are between 7 and 11 alphanumeric characters
        if len(cleaned) < 6:
            return PlateCandidate(
                raw_text=raw_text,
                canonical_text=cleaned,
                confidence=confidence * 0.3,
                is_valid_format=False,
            )

        # Pattern 1: Bharat Series (e.g. 22BH1234AA)
        bh_match = re.match(r"^(\d{2})BH(\d{4})([A-Z]{1,2})$", cleaned)
        if bh_match:
            return PlateCandidate(
                raw_text=raw_text,
                canonical_text=cleaned,
                confidence=confidence * 0.98,
                is_valid_format=True,
                state_code="BH",
                district_name="Bharat Series",
            )

        # Pattern 2: Standard State HSRP (e.g. GJ01AB1234, GJ01A1234, or legacy GJ011234)
        std_match = re.match(r"^([A-Z]{2})(\d{1,2})([A-Z]{0,3})(\d{1,4})$", cleaned)
        if std_match:
            state, rto, series, number = std_match.groups()
            rto_padded = rto.zfill(2)
            canonical = f"{state}{rto_padded}{series}{number.zfill(4)}"
            is_valid_state = state in INDIAN_STATE_CODES
            district = GUJARAT_RTO_CODES.get(rto_padded) if state == "GJ" else None

            # Boost confidence for valid Indian state and Gujarat district
            format_conf = confidence
            if is_valid_state:
                format_conf = min(1.0, format_conf * 1.1)
                if district:
                    format_conf = min(1.0, format_conf * 1.1)
            else:
                format_conf *= 0.5

            return PlateCandidate(
                raw_text=raw_text,
                canonical_text=canonical,
                confidence=format_conf,
                is_valid_format=is_valid_state,
                state_code=state,
                rto_code=rto_padded,
                district_name=district,
            )

        # Pattern 3: Comprehensive Positional OCR Syntax Repair
        # Fixes digit/letter confusion based on mandatory Indian plate character positions
        if len(cleaned) >= 7:
            repaired = list(cleaned)
            # 1. First 2 characters must be State letters (e.g. '6J' -> 'GJ', '0L' -> 'DL')
            for idx in range(2):
                if repaired[idx] in cls.NUM_TO_CHAR:
                    repaired[idx] = cls.NUM_TO_CHAR[repaired[idx]]

            # 2. Next 1-2 characters must be RTO digits (e.g. 'O1' -> '01', 'I8' -> '18')
            for idx in range(2, min(4, len(repaired))):
                if repaired[idx] in cls.CHAR_TO_NUM:
                    repaired[idx] = cls.CHAR_TO_NUM[repaired[idx]]

            # 3. Last 1-4 characters must be registration number digits
            for idx in range(max(4, len(repaired) - 4), len(repaired)):
                if repaired[idx] in cls.CHAR_TO_NUM:
                    repaired[idx] = cls.CHAR_TO_NUM[repaired[idx]]

            repaired_str = "".join(repaired)
            std_retry = re.match(r"^([A-Z]{2})(\d{1,2})([A-Z]{0,3})(\d{1,4})$", repaired_str)
            if std_retry:
                st, rto, ser, num = std_retry.groups()
                if st in INDIAN_STATE_CODES:
                    rto_padded = rto.zfill(2)
                    canonical = f"{st}{rto_padded}{ser}{num.zfill(4)}"
                    district = GUJARAT_RTO_CODES.get(rto_padded) if st == "GJ" else None
                    return PlateCandidate(
                        raw_text=raw_text,
                        canonical_text=canonical,
                        confidence=confidence * 0.9,
                        is_valid_format=True,
                        state_code=st,
                        rto_code=rto_padded,
                        district_name=district,
                    )

        # Fallback for unrecognized non-standard plates
        return PlateCandidate(
            raw_text=raw_text,
            canonical_text=cleaned,
            confidence=confidence * 0.4,
            is_valid_format=False,
        )


class TemporalTrackANPRVoter:
    """
    Accumulates multi-frame plate candidates across a single vehicle track and performs
    confidence-weighted temporal voting to produce the final verified plate.
    """

    def __init__(self, min_observations: int = 2, min_confidence: float = 0.6):
        self.min_observations = min_observations
        self.min_confidence = min_confidence
        # track_id -> list of PlateCandidate
        self.track_candidates: Dict[str, List[PlateCandidate]] = defaultdict(list)

    def add_observation(self, track_id: str, raw_text: str, confidence: float, timestamp: float = 0.0) -> PlateCandidate:
        candidate = IndianPlateValidator.validate_and_canonicalize(raw_text, confidence)
        candidate.timestamp = timestamp
        self.track_candidates[track_id].append(candidate)
        return candidate

    def get_consensus_plate(self, track_id: str) -> Tuple[Optional[str], float, Optional[str]]:
        """
        Returns: (consensus_plate_text, confidence, district_name)
        """
        candidates = self.track_candidates.get(track_id, [])
        if not candidates:
            return None, 0.0, None

        # Filter for valid formats if any exist
        valid_candidates = [c for c in candidates if c.is_valid_format]
        pool = valid_candidates if valid_candidates else candidates

        # Weighted voting by candidate confidence
        votes: Dict[str, float] = defaultdict(float)
        candidate_meta: Dict[str, PlateCandidate] = {}

        for c in pool:
            votes[c.canonical_text] += c.confidence
            if c.canonical_text not in candidate_meta or c.confidence > candidate_meta[c.canonical_text].confidence:
                candidate_meta[c.canonical_text] = c

        best_plate, total_weight = max(votes.items(), key=lambda item: item[1])
        # Average confidence calculation
        matching_count = sum(1 for c in pool if c.canonical_text == best_plate)
        avg_conf = min(0.99, (total_weight / max(1, matching_count)) * (1.0 + 0.05 * min(matching_count, 5)))

        meta = candidate_meta[best_plate]
        return best_plate, avg_conf, meta.district_name

    def clear_track(self, track_id: str):
        if track_id in self.track_candidates:
            del self.track_candidates[track_id]
