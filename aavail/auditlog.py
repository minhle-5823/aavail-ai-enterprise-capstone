"""Separated append-only JSONL audit logs, no user identifiers or personal names."""
from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path


def append_event(log_dir, category: str, event: dict):
    if category not in {"predictions", "feedback", "training"}:
        raise ValueError("Invalid log category")
    path = Path(log_dir)
    path.mkdir(parents=True, exist_ok=True)
    payload = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "category": category, **event}
    # A single append operation avoids truncating existing log contents.
    fd = os.open(str(path / f"{category}.jsonl"), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(fd, (json.dumps(payload, allow_nan=False, separators=(",", ":")) + "\n").encode("utf-8"))
    finally:
        os.close(fd)
    return payload


def read_events(log_dir, category):
    if category not in {"predictions", "feedback", "training"}:
        raise ValueError("Invalid log category")
    p = Path(log_dir) / f"{category}.jsonl"
    if not p.exists():
        return []
    with p.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def new_prediction_id():
    return uuid.uuid4().hex
