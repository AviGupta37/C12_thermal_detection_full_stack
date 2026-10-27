"""
Event and audit logger for C12 Thermal Human Detection Ground Station.
Records operator actions, detection alerts, stream errors, and system state transitions.
Maintains an in-memory ring-buffer for UI display and writes to disk.
"""

from collections import deque
import csv
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import threading
from typing import List, Optional


@dataclass
class LogEntry:
    """Represents a single audit/system log event."""
    timestamp: str
    level: str       # INFO, WARNING, ERROR, ALERT
    category: str    # SYSTEM, CAMERA, DETECTOR, TELEMETRY, RECORDING
    message: str

    def to_csv_row(self) -> List[str]:
        return [self.timestamp, self.level, self.category, self.message]


class EventLogger:
    """Thread-safe event logger with bounded history and file export."""

    def __init__(self, log_dir: str = "logs", max_history: int = 500):
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._max_history = max_history
        self._history = deque(maxlen=max_history)
        self._lock = threading.Lock()

        # Session log file
        session_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._session_file = self._log_dir / f"session_{session_time}.log"
        self._csv_file = self._log_dir / f"session_{session_time}.csv"

        # Initialize CSV header
        with open(self._csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "Level", "Category", "Message"])

    def log(self, level: str, category: str, message: str) -> LogEntry:
        """Record a structured log event."""
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        entry = LogEntry(timestamp=now_str, level=level.upper(), category=category.upper(), message=message)

        with self._lock:
            self._history.append(entry)

            # Append to text log
            try:
                with open(self._session_file, "a", encoding="utf-8") as f:
                    f.write(f"[{entry.timestamp}] [{entry.level:<5}] [{entry.category:<9}] {entry.message}\n")
            except Exception:
                pass

            # Append to CSV
            try:
                with open(self._csv_file, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(entry.to_csv_row())
            except Exception:
                pass

        return entry

    def info(self, category: str, message: str) -> LogEntry:
        return self.log("INFO", category, message)

    def warning(self, category: str, message: str) -> LogEntry:
        return self.log("WARNING", category, message)

    def error(self, category: str, message: str) -> LogEntry:
        return self.log("ERROR", category, message)

    def alert(self, category: str, message: str) -> LogEntry:
        return self.log("ALERT", category, message)

    def get_recent_entries(self, limit: Optional[int] = None) -> List[LogEntry]:
        """Return shallow copy of recent entries for UI table binding."""
        with self._lock:
            entries = list(self._history)
            if limit is not None and limit > 0:
                return entries[-limit:]
            return entries

    def export_csv(self, target_path: Path) -> bool:
        """Export current log history to a user-chosen destination."""
        with self._lock:
            entries = list(self._history)

        try:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with open(target_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["Timestamp", "Level", "Category", "Message"])
                for e in entries:
                    writer.writerow(e.to_csv_row())
            return True
        except Exception:
            return False

