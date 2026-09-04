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
