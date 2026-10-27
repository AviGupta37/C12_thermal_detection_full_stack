"""Flight telemetry interface package."""

from .telemetry_interface import TelemetryData, TelemetryProvider
from .mock_telemetry import MockTelemetryProvider

__all__ = ["TelemetryData", "TelemetryProvider", "MockTelemetryProvider"]

