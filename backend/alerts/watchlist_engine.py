"""
Police Watchlist and Real-Time Alert Engine for YUTA.

Monitors vehicle detections and ANPR plates against law enforcement watchlists:
- Exact and normalized Indian plate matching (e.g. GJ01AB1234, GJ-01-AB-1234).
- Severity tiers: CRITICAL, HIGH, MEDIUM, LOW.
- Instant alert generation with camera ID, GPS coordinates, timestamp, and case references.
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


@dataclass
class WatchlistEntry:
    """Entry in police vehicle hotlist."""
    plate_number: str  # Canonical uppercase, no spaces
    reason: str        # e.g. "Stolen Vehicle", "Suspect in FIR 104/2026", "Hit and Run"
    severity: str      # "CRITICAL", "HIGH", "MEDIUM", "LOW"
    case_id: Optional[str] = None
    vehicle_model: Optional[str] = None
    vehicle_color: Optional[str] = None
    registered_at: float = field(default_factory=time.time)
    active: bool = True


@dataclass
class AlertEvent:
    """Real-time alert triggered upon watchlist match or severe traffic anomaly."""
    alert_id: str
    global_vehicle_id: str
    plate_number: str
    camera_id: str
    camera_name: str
    lat: float
    lon: float
    timestamp: float
    reason: str
    severity: str
    confidence: float
    case_id: Optional[str] = None
    acknowledged: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class WatchlistAlertEngine:
    """
    Manages watchlist registry and checks incoming vehicle observations in real-time.
    """

    def __init__(self):
        self.watchlist: Dict[str, WatchlistEntry] = {}
        self.alerts: List[AlertEvent] = []
        self._alert_counter = 1

        # Seed initial sample watchlist entries for Gujarat Police testing
        self.add_entry(
            plate_number="GJ01AB1234",
            reason="Wanted: Vehicle involved in Hit & Run on Ashram Road",
            severity="CRITICAL",
            case_id="FIR-2026-0891",
            vehicle_model="White SUV",
            vehicle_color="White",
        )
        self.add_entry(
            plate_number="GJ18Z9999",
            reason="Stolen Commercial Vehicle reported in Gandhinagar",
            severity="HIGH",
            case_id="FIR-2026-0412",
            vehicle_model="Silver Sedan",
            vehicle_color="Silver",
        )

    def _normalize(self, plate: str) -> str:
        return plate.replace(" ", "").replace("-", "").upper()

    def add_entry(
        self,
        plate_number: str,
        reason: str,
        severity: str = "HIGH",
        case_id: Optional[str] = None,
        vehicle_model: Optional[str] = None,
        vehicle_color: Optional[str] = None,
    ) -> WatchlistEntry:
        canonical = self._normalize(plate_number)
        entry = WatchlistEntry(
            plate_number=canonical,
            reason=reason,
            severity=severity,
            case_id=case_id,
            vehicle_model=vehicle_model,
            vehicle_color=vehicle_color,
        )
        self.watchlist[canonical] = entry
        logger.info(f"Added plate {canonical} to police watchlist ({severity}: {reason}).")
        return entry

    def remove_entry(self, plate_number: str) -> bool:
        canonical = self._normalize(plate_number)
        if canonical in self.watchlist:
            del self.watchlist[canonical]
            return True
        return False

    def check_observation(
        self,
        global_vehicle_id: str,
        plate_number: Optional[str],
        camera_id: str,
        camera_name: str,
        lat: float,
        lon: float,
        confidence: float,
        timestamp: Optional[float] = None,
    ) -> Optional[AlertEvent]:
        """
        Checks vehicle plate against the watchlist and triggers an alert if matched.
        """
        if not plate_number:
            return None

        canonical = self._normalize(plate_number)
        entry = self.watchlist.get(canonical)

        if entry and entry.active:
            now = timestamp if timestamp is not None else time.time()
            alert = AlertEvent(
                alert_id=f"ALT-{self._alert_counter:05d}",
                global_vehicle_id=global_vehicle_id,
                plate_number=canonical,
                camera_id=camera_id,
                camera_name=camera_name,
                lat=lat,
                lon=lon,
                timestamp=now,
                reason=entry.reason,
                severity=entry.severity,
                confidence=confidence,
                case_id=entry.case_id,
            )
            self._alert_counter += 1
            self.alerts.append(alert)
            logger.warning(
                f"🚨 WATCHLIST ALERT [{alert.severity}]: Vehicle {canonical} spotted at {camera_name} (ID: {alert.alert_id})"
            )
            return alert

        return None

    def get_recent_alerts(self, limit: int = 50) -> List[AlertEvent]:
        return sorted(self.alerts, key=lambda a: a.timestamp, reverse=True)[:limit]
