"""
PX4 Flight Telemetry Interface.
Strictly read-only monitoring interface.
Does not send commands, does not alter flight modes, and does not interfere with PX4 Offboard setpoints.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import time
from typing import Callable, Optional


@dataclass
class TelemetryData:
    """Read-only quadrotor telemetry snapshot."""
    is_connected: bool = False
    is_simulated: bool = False
    flight_mode: str = "N/A"
    altitude_m: Optional[float] = None
    battery_percent: Optional[float] = None
    gps_status: str = "N/A"
    last_update_time: float = field(default_factory=time.time)

    @property
    def age_seconds(self) -> float:
        return max(0.0, time.time() - self.last_update_time)

    @property
    def altitude_str(self) -> str:
        if self.altitude_m is not None:
            return f"{self.altitude_m:.1f} m"
        return "N/A"

    @property
    def battery_str(self) -> str:
        if self.battery_percent is not None:
            return f"{int(round(self.battery_percent))}%"
        return "N/A"

    @property
    def connection_status_str(self) -> str:
        if not self.is_connected:
            return "DISCONNECTED"
        if self.is_simulated:
            return "SIMULATED"
        return "CONNECTED"


class TelemetryProvider(ABC):
    """Abstract interface for PX4 telemetry listeners."""

    @abstractmethod
    def start(self) -> bool:
        """Start listening for telemetry."""
        pass

    @abstractmethod
    def stop(self) -> None:
        """Stop listening for telemetry."""
        pass

    @abstractmethod
    def get_latest_telemetry(self) -> TelemetryData:
        """Retrieve most recent telemetry packet."""
        pass

    @property
    @abstractmethod
    def is_active(self) -> bool:
        """Check if provider is running."""
        pass

