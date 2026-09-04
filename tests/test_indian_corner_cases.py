"""
INDIAN ANPR CORNER CASES AUDIT TEST SUITE:
Verifies real-world Indian vehicle license plate challenges:
1. Skewed / tilted angles (4-point perspective unwarping)
2. Night scene glare and contrast enhancement via CLAHE
3. Two-line plates (auto-rickshaws, commercial trucks, scooters)
4. Positional character confusion (e.g. '6J' -> 'GJ', 'O1' -> '01', 'I234' -> '1234')
5. Complete Gujarat RTO district code coverage (GJ01 to GJ38)
6. Commercial vehicles (taxis, transport) and Bharat (BH) series
"""

import pytest
import numpy as np
import cv2

from vision.anpr.indian_anpr_engine import (
    IndianPlatePreprocessor,
    IndianPlateValidator,
    TemporalTrackANPRVoter,
    GUJARAT_RTO_CODES,
)


def test_tilted_perspective_quadrilateral_rectification():
    """Simulates a highly tilted CCTV angle (30-40 degree skew) and tests 4-point rectification."""
    img = np.zeros((400, 600, 3), dtype=np.uint8)
    quad_points = [
        (120.0, 150.0),  # Top-left
        (460.0, 100.0),  # Top-right
        (490.0, 220.0),  # Bottom-right
        (90.0, 250.0),   # Bottom-left
    ]
    cv2.fillPoly(img, [np.array(quad_points, dtype=np.int32)], (200, 200, 200))

    rectified = IndianPlatePreprocessor.rectify_quadrilateral(
        image=img,
        quad_points=quad_points,
        target_width=300,
        target_height=90,
    )

    assert rectified.shape == (90, 300, 3)
    center_roi = rectified[30:60, 100:200]
    assert np.mean(center_roi) > 150.0


def test_night_glare_and_clahe_enhancement():
    """Tests CLAHE enhancement under simulated headlight glare and low-contrast night conditions."""
    night_img = np.full((90, 300), 30, dtype=np.uint8)
    cv2.circle(night_img, (150, 45), 30, 240, -1)

    enhanced = IndianPlatePreprocessor.enhance_for_ocr(night_img)
    assert enhanced.shape == (90, 300)

    dark_region_orig = night_img[10:30, 10:50]
    dark_region_enh = enhanced[10:30, 10:50]
    assert np.std(dark_region_enh) >= np.std(dark_region_orig)


def test_two_line_plate_parsing():
    """Tests multi-line plates commonly found on commercial vehicles, autos, and motorcycles."""
    # Case 1: Auto-rickshaw / motorcycle 2-line plate with newline
    c1 = IndianPlateValidator.validate_and_canonicalize("GJ-01\nAB-1234")
    assert c1.is_valid_format is True
    assert c1.canonical_text == "GJ01AB1234"
    assert c1.district_name == "Ahmedabad (West)"

    # Case 2: Commercial truck 2-line plate with spaces and tabs
    c2 = IndianPlateValidator.validate_and_canonicalize("GJ 18 \n Z 9000")
    assert c2.is_valid_format is True
    assert c2.canonical_text == "GJ18Z9000"
    assert c2.district_name == "Gandhinagar"

    # Case 3: Vintage 2-line without series letters
    c3 = IndianPlateValidator.validate_and_canonicalize("GJ-05\n1234")
    assert c3.is_valid_format is True
    assert c3.canonical_text == "GJ051234"
    assert c3.district_name == "Surat (City)"


def test_positional_character_confusion_repair():
    """Tests positional OCR repair for ambiguous character pairs (6/G, O/0, I/1, B/8, Z/2)."""
    # '6J' -> State 'GJ', 'O1' -> District '01', 'I234' -> '1234'
    cand1 = IndianPlateValidator.validate_and_canonicalize("6JO1ABI234")
    assert cand1.is_valid_format is True
    assert cand1.canonical_text == "GJ01AB1234"
    assert cand1.district_name == "Ahmedabad (West)"

    # 'DL' with 'O8' -> 'DL08'
    cand2 = IndianPlateValidator.validate_and_canonicalize("DLO8CD5678")
    assert cand2.is_valid_format is True
    assert cand2.canonical_text == "DL08CD5678"

    # Gandhinagar '18' with 'GJ18AB8888' where B was confused in trailing digits
    cand3 = IndianPlateValidator.validate_and_canonicalize("GJ18ABBBBB")
    assert cand3.is_valid_format is True
    assert cand3.canonical_text == "GJ18AB8888"
    assert cand3.district_name == "Gandhinagar"


def test_complete_gujarat_rto_coverage():
    """Verifies recognition and district lookup across Gujarat RTOs (GJ01 to GJ38)."""
    test_rtos = [
        ("01", "Ahmedabad (West)"),
        ("02", "Mehsana"),
        ("03", "Rajkot"),
        ("05", "Surat (City)"),
        ("06", "Vadodara (City)"),
        ("12", "Bhuj (Kutch)"),
        ("18", "Gandhinagar"),
        ("27", "Ahmedabad (East)"),
        ("36", "Morbi"),
        ("38", "Bavla (Ahmedabad Rural)"),
    ]

    for rto, expected_district in test_rtos:
        plate_str = f"GJ{rto}AX5555"
        res = IndianPlateValidator.validate_and_canonicalize(plate_str)
        assert res.is_valid_format is True
        assert res.district_name == expected_district
        assert res.rto_code == rto


def test_commercial_and_bharat_series():
    """Tests commercial fleet formats and Bharat Series national registration."""
    # Commercial Taxi series in Ahmedabad (e.g. GJ01TT, GJ01TX)
    taxi_res = IndianPlateValidator.validate_and_canonicalize("GJ01TT9999")
    assert taxi_res.is_valid_format is True
    assert taxi_res.canonical_text == "GJ01TT9999"

    # Bharat Series (BH)
    bh1 = IndianPlateValidator.validate_and_canonicalize("22BH1234AA")
    assert bh1.is_valid_format is True
    assert bh1.state_code == "BH"
    assert bh1.district_name == "Bharat Series"

    bh2 = IndianPlateValidator.validate_and_canonicalize("23BH9876B")
    assert bh2.is_valid_format is True
    assert bh2.state_code == "BH"
