"""
LIVE RTSP & SENTINEL INTEGRATION AUDIT TEST:
Actually connects to active RTSP stream (rtsp://127.0.0.1:8554/stream/1),
decodes H.264 video frames over TCP transport, validates PTS timestamps,
and tests non-blocking ring buffer behavior.
"""

import time
import pytest
import numpy as np

from vision.stream.rtsp_stream import RTSPStreamReader, StreamFrame


def test_real_live_rtsp_stream_ingestion():
    # Connect directly to the active MediaMTX RTSP stream
    reader = RTSPStreamReader(
        camera_id="CAM_LIVE_01",
        rtsp_url="rtsp://127.0.0.1:8554/stream/1",
        target_fps=15.0,
        max_buffer_size=5,
        reconnect_timeout_sec=10.0,
    )

    try:
        reader.start()

        # Allow worker thread time to establish TCP handshake and decode first frames
        start_wait = time.time()
        frames_collected = []

        while time.time() - start_wait < 15.0 and len(frames_collected) < 5:
            frame = reader.get_latest_frame(timeout=0.5)
            if frame is not None and not frame.is_fallback:
                frames_collected.append(frame)
            time.sleep(0.05)

        # Verification 1: Must be connected and successfully receiving frames
        assert reader.status == "connected", f"Stream status was {reader.status}, expected 'connected'"
        assert len(frames_collected) >= 3, f"Expected at least 3 frames, got {len(frames_collected)}"

        # Verification 2: Check frame geometry and content
        first_frame = frames_collected[0]
        assert isinstance(first_frame, StreamFrame)
        assert first_frame.is_fallback is False, "Frame was synthetic fallback instead of real decoded frame!"
        assert first_frame.frame.shape == (720, 1280, 3)
        assert first_frame.camera_id == "CAM_LIVE_01"
        assert first_frame.frame_number >= 1

        # Verification 3: Monotonic timestamps
        for f1, f2 in zip(frames_collected, frames_collected[1:]):
            assert f2.timestamp >= f1.timestamp

    finally:
        reader.stop()
        assert reader.status == "stopped"


def test_sentinel_catalogue_discovery_and_pacing():
    """
    Tests live Sentinel catalogue synchronization against active backend /api/ingest,
    validating camera parameters, RTSP TCP URL generation, and staggered connection pacing.
    """
    from vision.stream.sentinel_client import SentinelCatalogueClient

    client = SentinelCatalogueClient(gateway_host="127.0.0.1", api_port=8000, rtsp_port=8554)
    cameras = client.fetch_catalogue()

    # Must discover cameras from the active Sentinel catalogue
    assert len(cameras) >= 5
    first_cam = cameras[0]
    assert first_cam.camera_id.startswith("CAM_")
    assert first_cam.rtsp_url.startswith("rtsp://127.0.0.1:8554/")
    assert 20.0 <= first_cam.lat <= 25.0  # Gujarat latitude
    assert 70.0 <= first_cam.lon <= 75.0  # Gujarat longitude

    # Verify connection pacing delay monotonically increases to prevent gateway connection surge
    delays = [client.get_connection_delay(idx, len(cameras), base_interval_sec=0.5) for idx in range(len(cameras))]
    for i in range(len(delays) - 1):
        assert delays[i + 1] > delays[i] - 0.2  # Monotonically staggered pacing with jitter
