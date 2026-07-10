"""Tiny JSON-backed model registry.

Each entry records:
    version, path, metrics, params, trained_at.

We append-only — old versions are kept around for audit and
rollback.  ``latest`` is computed at read time as the entry with
the most recent ``trained_at``.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from ..utils.config import get_config
from ..utils.logger import get_logger

log = get_logger(__name__)


def _load(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        log.warning("Registry %s corrupt — starting fresh", path)
        return []


def _save(path: Path, entries: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries, indent=2), encoding="utf-8")


def register(model_path: Path, metadata: Dict, metrics: Dict) -> str:
    """Add a new entry and return its version string."""
    cfg = get_config()
    reg_path = cfg.paths["registry_path"]
    entries = _load(reg_path)
    version = f"v{len(entries) + 1:04d}-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
    entry = {
        "version": version,
        "path": str(model_path),
        "trained_at": metadata.get("trained_at"),
        "params": metadata.get("params"),
        "val_metrics": metadata.get("val_metrics"),
        "test_metrics": metrics,
    }
    entries.append(entry)
    _save(reg_path, entries)
    log.info("registry: added %s", version)
    return version


def latest(reg_path: Optional[Path] = None) -> Optional[Dict]:
    """Return the most recently trained registry entry, if any."""
    cfg = get_config()
    entries = _load(reg_path or cfg.paths["registry_path"])
    if not entries:
        return None
    return sorted(entries, key=lambda e: e.get("trained_at") or "", reverse=True)[0]
