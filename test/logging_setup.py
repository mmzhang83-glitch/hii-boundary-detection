"""Shared logging setup for HII boundary detection.

Provides formatters and initialization functions used by the main
pipeline (detect_hii_boundary.py) and the test runner (run_test_plan.py).

Logger hierarchy::

    hii_boundary
    ├── .extract
    │   ├── .polar
    │   └── .dp
    ├── .scan
    ├── .bootstrap
    └── .test
        ├── .baseline
        ├── .noise
        ├── .center
        └── .rmax
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------

class ConsoleFormatter(logging.Formatter):
    """Compact console formatter — no logger name prefix, milestone-oriented."""

    # Mapping of ASCII fallbacks for checkmarks
    _MILESTONE = "✓"
    _SEPARATOR = "─"

    def format(self, record: logging.LogRecord) -> str:
        if record.levelno == logging.INFO:
            # Concise, no prefix
            return record.getMessage()
        elif record.levelno == logging.WARNING:
            return f"Warning: {record.getMessage()}"
        elif record.levelno == logging.ERROR:
            return f"Error: {record.getMessage()}"
        return record.getMessage()


class DetailedFormatter(logging.Formatter):
    """Timestamped, detailed formatter for file logs.

    Format: [2026-05-29 10:15:23.456] [logger.name] LEVEL  message
    """

    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        level = record.levelname.ljust(5)
        return f"[{ts}] [{record.name}] {level} {record.getMessage()}"


class JsonFormatter(logging.Formatter):
    """One JSON object per line for structured log indexing."""

    def format(self, record: logging.LogRecord) -> str:
        return json.dumps({
            "t": datetime.fromtimestamp(record.created).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3],
            "logger": record.name,
            "level": record.levelname,
            "msg": record.getMessage(),
        }, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Progress handler — inline \r-based progress for terminal
# ---------------------------------------------------------------------------

class ProgressHandler(logging.Handler):
    """Handler that writes \r-based inline progress to stderr.

    Only handles DEBUG-level records (used by bootstrap progress).
    After each emit, the StreamHandler on the same stream should be
    flushed so the next \r overwrites the line.
    """

    def __init__(self, stream=None):
        super().__init__(level=logging.DEBUG)
        self.stream = stream or sys.stderr

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            self.stream.write(f"\r{msg}")
            self.stream.flush()
        except Exception:
            self.handleError(record)


# ---------------------------------------------------------------------------
# Root logger
# ---------------------------------------------------------------------------

logger = logging.getLogger("hii_boundary")


def _setup_default_logger():
    """Auto-configure a basic ConsoleHandler if no handlers exist.

    Called at import time so standalone use of detect_hii_boundary()
    gets visible output without extra configuration.
    """
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(logging.INFO)
        handler.setFormatter(ConsoleFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)


def _setup_test_logging(output_dir: Path):
    """Replace handlers on the hii_boundary root logger for test mode.

    Three handlers:
      - ConsoleHandler (INFO) — compact terminal output
      - FileHandler (DEBUG)  — timestamped detailed log
      - FileHandler (INFO)   — JSON-structured log

    An additional ProgressHandler captures \r-based inline progress
    (DEBUG) for long-running steps like bootstrap.
    """
    root = logging.getLogger("hii_boundary")
    root.handlers.clear()

    # Remove the progress handler from the hii_boundary.bootstrap logger too
    bootstrap_log = logging.getLogger("hii_boundary.bootstrap")
    for h in list(bootstrap_log.handlers):
        if isinstance(h, ProgressHandler):
            bootstrap_log.handlers.remove(h)

    # Console: INFO, compact
    console = logging.StreamHandler(sys.stderr)
    console.setLevel(logging.INFO)
    console.setFormatter(ConsoleFormatter())
    root.addHandler(console)

    # File: DEBUG, detailed
    file_h = logging.FileHandler(output_dir / "test_plan.log", encoding="utf-8")
    file_h.setLevel(logging.DEBUG)
    file_h.setFormatter(DetailedFormatter())
    root.addHandler(file_h)

    # JSON: INFO, structured
    json_h = logging.FileHandler(output_dir / "test_plan.jsonl", encoding="utf-8")
    json_h.setLevel(logging.INFO)
    json_h.setFormatter(JsonFormatter())
    root.addHandler(json_h)

    # Progress: DEBUG, inline \r (only on bootstrap logger)
    progress = ProgressHandler(sys.stderr)
    progress.setLevel(logging.DEBUG)
    progress.setFormatter(ConsoleFormatter())
    bootstrap_log.addHandler(progress)
    bootstrap_log.propagate = True

    root.setLevel(logging.DEBUG)


# Auto-configure for standalone use
_setup_default_logger()