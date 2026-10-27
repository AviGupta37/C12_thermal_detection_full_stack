"""
Mock PX4 Telemetry Provider for UI development and offline demonstration.
Simulates quadrotor flight within the specified 20-50m operating envelope.
Clearly marked as SIMULATED to distinguish from actual vehicle state.
"""

import math
import threading
import time
from typing import Callable, Optional

from .telemetry_interface import TelemetryData, TelemetryProvider


class MockTelemetryProvider(TelemetryProvider):
    """Generates synthetic telemetry data representing a drone hovering/patrolling at 20-50m."""

    def __init__(self, update_rate_hz: float = 4.0, on_update: Optional[Callable[[TelemetryData], None]] = None):
        self._rate_hz = update_rate_hz
        self._on_update = on_update
        self._is_running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        # Initial simulation state
        self._data = TelemetryData(
            is_connected=True,
            is_simulated=True,
            flight_mode="OFFBOARD",
            altitude_m=32.4,
            battery_percent=88.0,
            gps_status="3D FIX (14 sats)",
            last_update_time=time.time(),
        )

    def start(self) -> bool:
        if self._is_running:
            return True
        self._is_running = True
        self._thread = threading.Thread(target=self._worker_loop, daemon=True, name="MockTelemetryWorker")
        self._thread.start()
        return True

    def stop(self) -> None:
        self._is_running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None
        with self._lock:
            self._data = TelemetryData(is_connected=False, is_simulated=True)

    def _worker_loop(self) -> None:
        interval = 1.0 / self._rate_hz
        step = 0.0

        while self._is_running:
            step += 0.05
            # Altitude sinusoidally oscillates within target envelope [24m, 44m]
            sim_alt = 34.0 + 10.0 * math.sin(step * 0.4)
            # Battery slowly discharges
            sim_batt = max(10.0, 88.0 - (step * 0.02))

            packet = TelemetryData(
                is_connected=True,
                is_simulated=True,
                flight_mode="OFFBOARD",
                altitude_m=round(sim_alt, 1),
                battery_percent=round(sim_batt, 1),
                gps_status="3D FIX (14 sats)",
                last_update_time=time.time(),
            )

            with self._lock:
                self._data = packet

            if self._on_update:
                try:
                    self._on_update(packet)
                except Exception:
                    pass

            time.sleep(interval)

    def get_latest_telemetry(self) -> TelemetryData:
        with self._lock:
            return self._data

    @property
    def is_active(self) -> bool:
        return self._is_running

