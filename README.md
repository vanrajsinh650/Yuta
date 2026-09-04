# YUTA — Unified Video Intelligence Platform

[![License: Apache 2.0 / MIT](https://img.shields.io/badge/License-Apache%202.0%20%2F%20MIT-blue.svg)](docs/THIRD_PARTY_LICENSES.md)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://python.org)
[![Tests: 29 Passed](https://img.shields.io/badge/Tests-29%20Passed-brightgreen.svg)](tests/)

**YUTA** is an end-to-end, multi-camera video intelligence and investigation platform engineered for the **Gujarat Police Innovation Challenge 2026**.

Instead of stopping at simple per-camera detection (*"Vehicle detected"*), YUTA answers the operational law-enforcement question: **"Where did this vehicle go?"**

---

## 🏛️ End-to-End System Pipeline

```text
Sentinel CCTV / RTSP Gateway (TCP)
        ↓
Resilient Stream Ingestion (Zero-Lag Ring Buffer, Auto-Reconnect)
        ↓
ByteTrack 8D Kalman Local Tracking (Low-Confidence Occlusion Recovery)
        ↓
Indian ANPR & 4-Point Homography Rectification (CLAHE, Gujarat RTO DB, Multi-Frame Voter)
        ↓
Cross-Camera BEV Association & Feature Projection (Bird's Eye View Clustering)
        ↓
Global Trajectory Linker (Iterative Hungarian Assignment, Spatio-Temporal Gating)
        ↓
Camera Topology & Route Reconstruction (Gaussian Travel-Time Likelihood, GeoJSON LineStrings)
        ↓
Multi-Camera Evidence Graph (Causal Transition Links, Node-to-Edge Explanations)
        ↓
FastAPI Backend & Police Watchlist Engine (Real-Time Hotlist Alerts)
        ↓
Grounded AI Investigation Assistant & Interactive Leaflet GIS Dashboard
```

---

## 🔬 Provenance & Research Integration

YUTA was synthesized through a rigorous **"Research → Extract → Integrate → Test → Clean"** pipeline, combining the strongest proven components from leading computer vision repositories:

| Component | Source Repository | Purpose in YUTA |
| :--- | :--- | :--- |
| **Stream Engine & BEV Calibration** | [`playbox-dev/trackstudio`](https://github.com/playbox-dev/trackstudio) | Resilient RTSP over TCP, synthetic fallback frames, 4-point ground homography. |
| **Global Trajectory Linker** | [`ZIOVISION/AIC2025_Track1_ZV`](https://github.com/ZIOVISION/AIC2025_Track1_ZV) (1st Place AI City 2025) | Iterative Hungarian tracklet assignment, velocity & multi-modal gating. |
| **Local Tracking & Attention** | [`FoxCanned/GMT`](https://github.com/FoxCanned/GMT) (CVPR 2026) | ByteTrack 2-stage association with 8D Kalman filter and cross-attention affinity. |
| **Route Reconstruction** | [`tsinghua-fib-lab/Cam-Traj-Rec`](https://github.com/tsinghua-fib-lab/Cam-Traj-Rec) (KDD 2022) | Camera topology network, Gaussian travel-time likelihood, road graph routing. |
| **Indian ANPR & Rectification** | [`sanchit2843/Indian_LPR`](https://github.com/sanchit2843/Indian_LPR) | 4-point plate unwarping, CLAHE enhancement, Gujarat RTO database (GJ01–GJ38), track voting. |
| **Evidence Graph & AI Query** | [`siri-rouser/TAU-Agent`](https://github.com/siri-rouser/TAU-Agent) (AI City 2026) | Multi-camera Evidence Graph with grounded natural language search. |

For detailed license terms and attribution notices, see [docs/THIRD_PARTY_LICENSES.md](docs/THIRD_PARTY_LICENSES.md).

---

## 🚀 Quick Start

### 1. Local Setup
```bash
# Clone repository
git clone https://github.com/vanrajsinh650/Yuta.git
cd Yuta

# Setup virtual environment with uv (or python -m venv)
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -r requirements.txt  # or run deployment/run_yuta.sh

# Run all unit and integration tests (29 tests)
PYTHONPATH=. pytest tests/ -v

# Launch the unified server and dashboard
bash deployment/run_yuta.sh
```

Open **http://localhost:8000** in your browser to access the interactive investigation console.

### 2. Docker Deployment
```bash
cd deployment
docker compose up -d
```

---

## 📡 REST API & WebSocket Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Healthcheck and system status |
| `GET` | `/api/cameras` | List registered Sentinel camera nodes and live streams |
| `GET` | `/api/vehicles` | List all tracked global vehicle identities |
| `GET` | `/api/vehicles/search?q=...` | Multi-attribute search (Plate, Class, Color, Camera) |
| `GET` | `/api/vehicles/{id}/route` | Spatio-temporal route reconstruction (GeoJSON format) |
| `GET` | `/api/vehicles/{id}/evidence` | Grand-Finale Evidence Graph (Causal camera hops & metrics) |
| `GET` | `/api/watchlist` | Active police vehicle hotlist |
| `POST` | `/api/watchlist` | Add suspect vehicle to real-time alert hotlist |
| `GET` | `/api/alerts` | Real-time alert feed (CRITICAL, HIGH severity) |
| `POST` | `/api/investigate/nlq` | Grounded natural-language police query engine |
| `WS` | `/ws/live-events` | WebSocket stream for live sightings, tracks, and alerts |

---

## 🇮🇳 Special Indian Traffic Corner Cases Handled
- **High Security Registration Plates (HSRP)** & 2-line older plates.
- **Gujarat RTO Verification**: Canonicalizes RTO district codes (e.g. `GJ01` Ahmedabad West, `GJ05` Surat, `GJ06` Vadodara, `GJ18` Gandhinagar).
- **Bharat Series (`BH`) Support**: Correctly identifies and parses central government `22BH1234AA` registrations.
- **Perspective Distortion**: 4-point quadrilateral rectification unwarps skewed plates on motorcycles, autorickshaws, and high-angle pole CCTV.
- **Temporal Voting**: Eliminates single-frame OCR flicker by computing weighted consensus across the vehicle's entire tracklet.
- **Night & Glare Preprocessing**: CLAHE contrast expansion and bilateral filtering to recover text through headlight glare and road dirt.

---

## 🛡️ License
YUTA is released under Apache 2.0 / MIT compatible open source licensing. Third-party algorithm provenance is fully documented in [docs/THIRD_PARTY_LICENSES.md](docs/THIRD_PARTY_LICENSES.md).
