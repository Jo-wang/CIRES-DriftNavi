"""
Lightweight dataset snapshot manager.

A snapshot represents a pair of datasets (primary and secondary) captured at a
point in time, together with minimal metadata. This module integrates tightly
with the app-wide global state (`global_vars`) but keeps snapshot bookkeeping in
one place for clarity and reuse.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, List

from UI.functions.global_vars import global_vars


@dataclass
class SnapshotMeta:
    ver: str              # 'original' or 'v1'...'vn'
    desc: str             # human friendly description
    time: str             # ISO8601 string
    has_secondary: bool   # whether a secondary dataset exists


def _ensure_store() -> None:
    """Ensure snapshot containers exist on global_vars."""
    if not hasattr(global_vars, 'snapshots'):
        global_vars.snapshots = []            # list[dict]
    if not hasattr(global_vars, 'snapshot_payloads'):
        global_vars.snapshot_payloads = {}    # ver -> payload dict


def _next_version() -> str:
    """Return the next version label like 'v3'."""
    _ensure_store()
    count = sum(1 for s in global_vars.snapshots if isinstance(s, dict) and str(s.get('ver', '')).startswith('v'))
    return f"v{count + 1}"


def create_original_if_absent() -> None:
    """Create an 'original' snapshot if none exists yet."""
    _ensure_store()
    for s in global_vars.snapshots:
        if s.get('ver') == 'original':
            # If secondary arrives later, enrich original payload to be a complete pair
            payload = global_vars.snapshot_payloads.get('original', {})
            if payload.get('secondary') is None and getattr(global_vars, 'secondary_df', None) is not None:
                payload['secondary'] = getattr(global_vars, 'secondary_df', None)
                payload['secondary_file_name'] = getattr(global_vars, 'secondary_file_name', None)
                global_vars.snapshot_payloads['original'] = payload
                s['has_secondary'] = True
            return
    meta = SnapshotMeta(
        ver='original',
        desc='Original data',
        time=datetime.now().isoformat(timespec='seconds'),
        has_secondary=bool(getattr(global_vars, 'secondary_df', None) is not None)
    )
    global_vars.snapshots.insert(0, asdict(meta))
    # store DEEP COPIES to avoid in-place preview edits mutating snapshots
    _primary = getattr(global_vars, 'df', None)
    _secondary = getattr(global_vars, 'secondary_df', None)
    global_vars.snapshot_payloads['original'] = {
        'primary': _primary.copy(deep=True) if _primary is not None else None,
        'secondary': _secondary.copy(deep=True) if _secondary is not None else None,
        'file_name': getattr(global_vars, 'file_name', None),
        'secondary_file_name': getattr(global_vars, 'secondary_file_name', None),
        'target_attribute': getattr(global_vars, 'target_attribute', None),
    }


def create_snapshot(desc: str = "") -> str:
    """Create a new version snapshot from current global datasets.

    Returns the created version label (e.g., 'v1').
    """
    _ensure_store()
    ver = _next_version()
    meta = SnapshotMeta(
        ver=ver,
        desc=desc or f"Snapshot {ver}",
        time=datetime.now().isoformat(timespec='seconds'),
        has_secondary=bool(getattr(global_vars, 'secondary_df', None) is not None)
    )
    global_vars.snapshots.append(asdict(meta))
    # store DEEP COPIES to make versions immutable
    _primary = getattr(global_vars, 'df', None)
    _secondary = getattr(global_vars, 'secondary_df', None)
    global_vars.snapshot_payloads[ver] = {
        'primary': _primary.copy(deep=True) if _primary is not None else None,
        'secondary': _secondary.copy(deep=True) if _secondary is not None else None,
        'file_name': getattr(global_vars, 'file_name', None),
        'secondary_file_name': getattr(global_vars, 'secondary_file_name', None),
        'target_attribute': getattr(global_vars, 'target_attribute', None),
    }
    return ver


def list_snapshots() -> List[Dict[str, Any]]:
    """Return shallow list of snapshot metadata dicts."""
    _ensure_store()
    return list(global_vars.snapshots)


def restore_snapshot(ver: str) -> bool:
    """Restore a snapshot into global_vars and invalidate metrics cache."""
    _ensure_store()
    payload = global_vars.snapshot_payloads.get(ver)
    if not payload:
        return False

    # 1) Write datasets and related context back
    global_vars.df = payload.get('primary')
    global_vars.secondary_df = payload.get('secondary')
    global_vars.file_name = payload.get('file_name')
    global_vars.secondary_file_name = payload.get('secondary_file_name')
    if payload.get('target_attribute') is not None:
        global_vars.target_attribute = payload['target_attribute']

    # 2) Recreate fingerprints and invalidate metrics cache
    try:
        global_vars.initialize_dataset_fingerprints(force=True)
    except Exception:
        # Be resilient; continue to clear cache
        pass
    global_vars.clear_metrics_cache(f"Restored snapshot {ver}")
    # Explicitly mark metrics as outdated so chat Detect recomputes immediately
    if hasattr(global_vars, 'dataset_change_flags'):
        try:
            global_vars.dataset_change_flags['metrics_outdated'] = True
        except Exception:
            pass

    return True


