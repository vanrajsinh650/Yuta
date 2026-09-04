// YUTA Frontend Investigation Dashboard
const API_BASE = "http://localhost:8000";

let map, routeLayer, cameraMarkersGroup;
let networkInstance = null;

// Initialize when DOM ready
document.addEventListener("DOMContentLoaded", () => {
  initMap();
  loadCameras();
  loadVehicleRoute("GLOBAL-VEH-0001");
  performVehicleSearch();
  loadAlerts();
  loadEvidenceGraph("GLOBAL-VEH-0001");
});

// === TAB SWITCHER ===
function switchTab(tabName) {
  document.querySelectorAll(".tab-content").forEach(el => el.classList.add("hidden"));
  document.querySelectorAll(".tab-btn").forEach(btn => {
    btn.classList.remove("bg-amber-500", "text-slate-950", "shadow");
    btn.classList.add("text-slate-400");
  });

  const activeContent = document.getElementById(`tab-${tabName}`);
  const activeBtn = document.getElementById(`tab-btn-${tabName}`);
  if (activeContent) activeContent.classList.remove("hidden");
  if (activeBtn) {
    activeBtn.classList.remove("text-slate-400");
    activeBtn.classList.add("bg-amber-500", "text-slate-950", "shadow");
  }

  if (tabName === "gis" && map) {
    setTimeout(() => map.invalidateSize(), 200);
  }
}

// === GIS MAP ENGINE (Leaflet) ===
function initMap() {
  map = L.map("map").setView([23.055, 72.573], 14);

  // CartoDB Dark Matter tiles
  L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", {
    attribution: '&copy; <a href="https://carto.com/">CARTO</a> | YUTA GPIC 2026',
    maxZoom: 19
  }).addTo(map);

  cameraMarkersGroup = L.layerGroup().addTo(map);
}

async function loadCameras() {
  try {
    const res = await fetch(`${API_BASE}/api/cameras`);
    const cameras = await res.json();
    cameraMarkersGroup.clearLayers();

    cameras.forEach(cam => {
      const icon = L.divIcon({
        className: 'custom-cam-icon',
        html: `<div class="bg-amber-500/20 border-2 border-amber-400 text-amber-300 w-8 h-8 rounded-full flex items-center justify-center shadow-lg shadow-amber-500/20"><i class="fa-solid fa-video text-xs"></i></div>`,
        iconSize: [32, 32],
        iconAnchor: [16, 16]
      });

      const marker = L.marker([cam.lat, cam.lon], { icon }).addTo(cameraMarkersGroup);
      marker.bindPopup(`
        <div class="p-1 space-y-1">
          <div class="font-bold text-amber-400">${cam.name}</div>
          <div class="text-xs text-slate-300">ID: ${cam.camera_id}</div>
          <div class="text-xs text-slate-400">GPS: ${cam.lat.toFixed(4)}, ${cam.lon.toFixed(4)}</div>
          <div class="text-xs text-emerald-400 font-semibold mt-1">● RTSP TCP Live</div>
        </div>
      `);
    });
  } catch (err) {
    console.error("Failed to load cameras", err);
  }
}

async function loadVehicleRoute(globalId) {
  try {
    const res = await fetch(`${API_BASE}/api/vehicles/${globalId}/route`);
    const data = await res.json();

    if (routeLayer) map.removeLayer(routeLayer);

    // Draw route LineString
    routeLayer = L.geoJSON(data.geojson, {
      style: {
        color: "#f59e0b",
        weight: 5,
        opacity: 0.85,
        dashArray: "8, 8"
      }
    }).addTo(map);

    map.fitBounds(routeLayer.getBounds(), { padding: [40, 40] });

    document.getElementById("route-stat-badge").innerHTML =
      `<span class="text-amber-400 font-bold">${data.plate_number}</span> | ${data.total_distance_m}m | ${data.average_speed_kmh} km/h | ${(data.overall_confidence * 100).toFixed(1)}% Conf`;

    // Render segment list
    const listEl = document.getElementById("route-segment-list");
    listEl.innerHTML = data.segments.map((seg, idx) => `
      <div class="bg-slate-950 p-3 rounded-xl border border-slate-800/80 space-y-1">
        <div class="flex items-center justify-between font-semibold text-slate-200">
          <span>Hop ${idx + 1}: ${seg.from_camera} → ${seg.to_camera}</span>
          <span class="text-emerald-400">${(seg.likelihood * 100).toFixed(0)}% Match</span>
        </div>
        <div class="text-slate-400 flex justify-between">
          <span>Dist: ${seg.distance_m}m (${seg.time_sec}s)</span>
          <span class="text-amber-400 font-medium">${seg.speed_kmh} km/h</span>
        </div>
      </div>
    `).join("");

  } catch (err) {
    console.error("Failed to load route", err);
  }
}

// === GRAND-FINALE EVIDENCE GRAPH (Vis.js) ===
async function loadEvidenceGraph(globalId) {
  try {
    const res = await fetch(`${API_BASE}/api/vehicles/${globalId}/evidence`);
    const data = await res.json();

    const nodes = data.nodes.map(n => ({
      id: n.node_id,
      label: `${n.camera_name}\n[${n.plate_number}]`,
      color: { background: "#1e293b", border: "#f59e0b" },
      font: { color: "#ffffff", size: 12 },
      shape: "box",
      raw: n
    }));

    const edges = data.edges.map(e => ({
      from: e.source,
      to: e.target,
      label: `${e.time_gap_sec}s | ${e.implied_speed_kmh} km/h\nReID: ${(e.appearance_similarity * 100).toFixed(0)}%`,
      color: { color: "#38bdf8", highlight: "#eab308" },
      arrows: "to",
      font: { color: "#94a3b8", size: 10, align: "horizontal" },
      raw: e
    }));

    const container = document.getElementById("evidence-network");
    const networkData = {
      nodes: new vis.DataSet(nodes),
      edges: new vis.DataSet(edges)
    };

    const options = {
      physics: { stabilization: true },
      layout: { hierarchical: { direction: "LR", sortMethod: "directed", levelSeparation: 180 } }
    };

    networkInstance = new vis.Network(container, networkData, options);

    networkInstance.on("click", params => {
      const detailEl = document.getElementById("evidence-detail-panel");
      if (params.nodes.length > 0) {
        const nId = params.nodes[0];
        const nObj = nodes.find(n => n.id === nId).raw;
        detailEl.innerHTML = `
          <div class="space-y-1">
            <span class="font-bold text-amber-400">Camera Sighting: ${nObj.camera_name} (${nObj.camera_id})</span>
            <div>Vehicle: ${nObj.vehicle_color} ${nObj.vehicle_class.toUpperCase()} | Plate: <span class="text-white font-semibold">${nObj.plate_number}</span> (Conf: ${(nObj.plate_confidence * 100).toFixed(1)}%)</div>
            <div class="text-slate-400">Timestamp: ${new Date(nObj.timestamp * 1000).toLocaleTimeString()}</div>
          </div>
        `;
      } else if (params.edges.length > 0) {
        const eId = params.edges[0];
        const eObj = edges[params.edges[0]] ? edges[params.edges[0]].raw : data.edges[0];
        detailEl.innerHTML = `
          <div class="space-y-1">
            <span class="font-bold text-emerald-400">Cross-Camera Link Verification: ${eObj.from_camera} → ${eObj.to_camera}</span>
            <div>Spatio-Temporal Feasibility: ${eObj.distance_m}m in ${eObj.time_gap_sec}s (${eObj.implied_speed_kmh} km/h) | Gaussian Road Likelihood: ${(eObj.road_likelihood * 100).toFixed(1)}%</div>
            <div>Appearance Match: ReID Cosine ${(eObj.appearance_similarity * 100).toFixed(1)}% | Plate Match: <span class="text-emerald-400 font-bold">${eObj.plate_match ? 'VERIFIED' : 'N/A'}</span></div>
          </div>
        `;
      }
    });

  } catch (err) {
    console.error("Failed to load evidence graph", err);
  }
}

// === VEHICLE SEARCH ===
async function performVehicleSearch() {
  const query = document.getElementById("search-plate-input").value;
  try {
    const url = query ? `${API_BASE}/api/vehicles/search?q=${encodeURIComponent(query)}` : `${API_BASE}/api/vehicles`;
    const res = await fetch(url);
    const list = await res.json();

    const tbody = document.getElementById("vehicle-search-results");
    if (list.length === 0) {
      tbody.innerHTML = `<tr><td colspan="7" class="p-4 text-center text-slate-500">No vehicles matching query</td></tr>`;
      return;
    }

    tbody.innerHTML = list.map(v => `
      <tr class="hover:bg-slate-900/60 transition">
        <td class="p-3 font-mono text-amber-400 font-semibold">${v.global_id}</td>
        <td class="p-3 font-bold text-white tracking-wider">${v.plate_number || 'N/A'}</td>
        <td class="p-3"><span class="bg-emerald-500/20 text-emerald-400 px-2 py-0.5 rounded font-semibold">${((v.confidence || v.plate_confidence || 0.9) * 100).toFixed(0)}%</span></td>
        <td class="p-3 capitalize">${v.vehicle_color || ''} ${v.vehicle_class || ''}</td>
        <td class="p-3">${v.sightings_count || v.total_sightings || 1} sightings</td>
        <td class="p-3 text-slate-400">${v.last_camera || v.last_camera_name || 'N/A'}</td>
        <td class="p-3 text-right space-x-2">
          <button onclick="inspectVehicle('${v.global_id}')" class="bg-slate-800 hover:bg-slate-700 text-slate-200 px-2.5 py-1 rounded transition">
            <i class="fa-solid fa-map-pin mr-1"></i> Route
          </button>
        </td>
      </tr>
    `).join("");

  } catch (err) {
    console.error("Search failed", err);
  }
}

function inspectVehicle(globalId) {
  switchTab('gis');
  loadVehicleRoute(globalId);
  loadEvidenceGraph(globalId);
}

// === NATURAL LANGUAGE AI INVESTIGATION (TAU-Agent) ===
async function submitNLQ() {
  const query = document.getElementById("nlq-input").value;
  if (!query) return;
  runSampleNLQ(query);
}

async function runSampleNLQ(query) {
  document.getElementById("nlq-input").value = query;
  const resultBox = document.getElementById("nlq-result-box");
  const answerText = document.getElementById("nlq-answer-text");
  const intentBadge = document.getElementById("nlq-intent-badge");
  const citationsEl = document.getElementById("nlq-citations");

  resultBox.classList.remove("hidden");
  answerText.innerText = "Querying grounded Evidence Graph...";

  try {
    const res = await fetch(`${API_BASE}/api/investigate/nlq`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query })
    });
    const data = await res.json();

    intentBadge.innerText = `Intent: ${data.intent.toUpperCase()}`;
    answerText.innerText = data.summary;

    if (data.evidence && data.evidence.length > 0) {
      citationsEl.innerHTML = `<span class="font-bold text-slate-300">Citations:</span><br>` +
        data.evidence.map(e => `• [Camera ${e.camera_id || e.from_camera} | ${e.camera_name || ''}] Sightings at timestamp ${e.timestamp ? new Date(e.timestamp * 1000).toLocaleTimeString() : 'N/A'}`).join("<br>");
    } else {
      citationsEl.innerHTML = "";
    }
  } catch (err) {
    answerText.innerText = "Error executing investigation query.";
  }
}

// === WATCHLIST ALERTS ===
async function loadAlerts() {
  try {
    const res = await fetch(`${API_BASE}/api/alerts`);
    const alerts = await res.json();
    const listEl = document.getElementById("active-alerts-list");

    if (alerts.length === 0) {
      listEl.innerHTML = `<div class="text-slate-500 text-xs text-center py-6">No active watchlist alerts.</div>`;
      return;
    }

    listEl.innerHTML = alerts.map(a => `
      <div class="p-4 rounded-xl bg-slate-950 border border-red-500/30 shadow-lg flex items-start justify-between">
        <div class="space-y-1 text-xs">
          <div class="flex items-center gap-2">
            <span class="bg-red-500/20 text-red-400 font-bold px-2 py-0.5 rounded border border-red-500/40">${a.severity}</span>
            <span class="font-mono font-bold text-white text-sm">${a.plate_number}</span>
            <span class="text-slate-400">(${a.global_vehicle_id})</span>
          </div>
          <div class="text-slate-300 font-medium">${a.reason}</div>
          <div class="text-slate-400">
            <i class="fa-solid fa-location-dot text-amber-400 mr-1"></i> ${a.camera_name} (${a.camera_id})
            • ${new Date(a.timestamp * 1000).toLocaleTimeString()}
          </div>
        </div>
        <button onclick="inspectVehicle('${a.global_vehicle_id}')" class="text-xs bg-amber-500 hover:bg-amber-600 text-slate-950 font-bold px-3 py-1.5 rounded-lg transition">
          Track Route
        </button>
      </div>
    `).join("");

  } catch (err) {
    console.error("Failed to load alerts", err);
  }
}

async function addWatchlistPlate() {
  const plate = document.getElementById("wl-plate").value;
  const reason = document.getElementById("wl-reason").value;
  const severity = document.getElementById("wl-severity").value;
  if (!plate || !reason) return alert("Please specify plate and reason.");

  try {
    await fetch(`${API_BASE}/api/watchlist`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ plate_number: plate, reason, severity })
    });
    alert(`Plate ${plate} added to police hotlist.`);
    document.getElementById("wl-plate").value = "";
    document.getElementById("wl-reason").value = "";
    loadAlerts();
  } catch (err) {
    console.error("Failed to add watchlist entry", err);
  }
}
