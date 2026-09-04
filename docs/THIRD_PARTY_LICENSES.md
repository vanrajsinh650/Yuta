# Third-Party Notices & Licenses

YUTA (Unified Video Intelligence Platform) incorporates algorithms, architectures, and components from open-source projects in accordance with their respective licenses. This document preserves required attribution and licensing notices.

---

## 1. TrackStudio (`playbox-dev/trackstudio`)

- **Source**: https://github.com/playbox-dev/trackstudio
- **License**: Apache License 2.0
- **Copyright**: Copyright (c) 2024 Playbox Dev
- **Integrated Components**:
  - RTSP ingestion engine with TCP transport and auto-reconnection (`vision/stream/rtsp_stream.py`)
  - 4-Point Homography & Bird's Eye View (BEV) Calibration (`vision/calibration/calibration.py`)
  - Multi-Camera BEV Cluster Association & Graph Matching (`vision/association/bev_merger.py`)
  - Base Tracker & Track Data Models (`vision/tracking/tracker_interface.py`)

A copy of the Apache License 2.0 is retained below:

```text
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

---

## 2. ZIOVISION AIC2025 (`ZIOVISION/AIC2025_Track1_ZV`)

- **Source**: https://github.com/ZIOVISION/AIC2025_Track1_ZV
- **Reference**: AI City Challenge 2025 Track 1 (1st Place MCMT Solution)
- **Integrated Components**:
  - Global Trajectory Linking Engine (`vision/association/trajectory_linker.py`)
  - Multi-Modal Spatio-Temporal Gating & Hungarian Tracklet Association
  - Iterative Hierarchical Trajectory Merging Algorithm

---

## 3. GMT (`FoxCanned/GMT`)

- **Source**: https://github.com/FoxCanned/GMT
- **Reference**: CVPR 2026 Paper: "GMT: Effective Global Framework for Multi-Camera Multi-Target Tracking"
- **License**: Apache License 2.0
- **Integrated Components**:
  - Attention-Based Global Association Affinity Head (`vision/association/attention_association.py`)
  - Two-Stage ByteTrack-Style Kalman Filter Local Tracker (`vision/tracking/byte_tracker.py`)

---

## 4. Cam-Traj-Rec (`tsinghua-fib-lab/Cam-Traj-Rec`)

- **Source**: https://github.com/tsinghua-fib-lab/Cam-Traj-Rec
- **Reference**: KDD 2022 Paper: "Spatio-Temporal Vehicle Trajectory Recovery on Road Network Based on Traffic Camera Video Data"
- **License**: MIT License
- **Copyright**: Copyright (c) 2023 FIB LAB, Tsinghua University
- **Integrated Components**:
  - Camera Topology Network & Road Constraints Graph (`vision/trajectory/route_reconstructor.py`)
  - Gaussian Travel-Time Transition Likelihood & Velocity Filtering
  - Spatio-Temporal Route Reconstruction & Intermediate Waypoint Recovery

---

## 5. Indian_LPR (`sanchit2843/Indian_LPR`)

- **Source**: https://github.com/sanchit2843/Indian_LPR
- **Reference**: Indian Number Plate dataset in wild & LPR benchmark
- **Integrated Components**:
  - 4-Point Plate Homography Rectification & Orientation Correction (`vision/anpr/indian_anpr_engine.py`)
  - Indian Registration Pattern Validation & Gujarat RTO Normalizer (`vision/anpr/indian_anpr_engine.py`)
  - Multi-Frame Track Voting & Confidence Aggregation Engine (`vision/anpr/indian_anpr_engine.py`)
