"""
Test Suite for Indian ANPR Engine Integration in YUTA (derived from Indian_LPR).
"""

import pytest
import numpy as np

from vision.anpr.indian_anpr_engine import (
    IndianPlatePreprocessor,
    IndianPlateValidator,
    TemporalTrackANPRVoter,
    GUJARAT_RTO_CODES,
)


def test_plate_perspective_unwarping():
    # Synthetic image with a skewed plate
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    quad = [(100.0, 150.0), (350.0, 170.0), (340.0, 260.0), (90.0, 240.0)]

    rectified = IndianPlatePreprocessor.rectify_quadrilateral(img, quad, target_width=300, target_height=90)
    assert rectified.shape == (90, 300, 3)

    enhanced = IndianPlatePreprocessor.enhance_for_ocr(rectified)
    assert enhanced.shape == (90, 300)
    assert enhanced.dtype == np.uint8


def test_indian_plate_validator_standard_gj():
    res = IndianPlateValidator.validate_and_canonicalize("GJ-01-AB-1234", confidence=0.85)
    assert res.is_valid_format is True
    assert res.canonical_text == "GJ01AB1234"
    assert res.state_code == "GJ"
    assert res.rto_code == "01"
    assert res.district_name == "Ahmedabad (West)"
    assert res.confidence > 0.85


def test_indian_plate_validator_gandhinagar_rto():
    res = IndianPlateValidator.validate_and_canonicalize("GJ18-Z-9999", confidence=0.90)
    assert res.is_valid_format is True
    assert res.canonical_text == "GJ18Z9999"
    assert res.rto_code == "18"
    assert res.district_name == "Gandhinagar"


def test_indian_plate_validator_bharat_series():
    res = IndianPlateValidator.validate_and_canonicalize("22 BH 4567 CD", confidence=0.88)
    assert res.is_valid_format is True
    assert res.canonical_text == "22BH4567CD"
    assert res.district_name == "Bharat Series"


def test_indian_plate_ocr_correction():
    # OCR misread 'G' as '6': '6J01AB1234'
    res = IndianPlateValidator.validate_and_canonicalize("6J01AB1234", confidence=0.80)
    assert res.is_valid_format is True
    assert res.canonical_text == "GJ01AB1234"


def test_temporal_track_voting_eliminates_ocr_flicker():
    voter = TemporalTrackANPRVoter()
    track_id = "trk_cam01_42"

    # Frame 1: Clean read
    voter.add_observation(track_id, "GJ 01 AB 1234", confidence=0.88)
    # Frame 2: Minor whitespace / hyphen variation
    voter.add_observation(track_id, "GJ-01-AB-1234", confidence=0.92)
    # Frame 3: One blurry corrupted read
    voter.add_observation(track_id, "MH99ZZ0000", confidence=0.35)
    # Frame 4: Clean read again
    voter.add_observation(track_id, "GJ01AB1234", confidence=0.90)

    consensus_plate, consensus_conf, district = voter.get_consensus_plate(track_id)
    assert consensus_plate == "GJ01AB1234"
    assert district == "Ahmedabad (West)"
    assert consensus_conf > 0.90
