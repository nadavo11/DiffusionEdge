"""Utilities to persist train/eval parameters for reproducibility."""

from __future__ import annotations

import argparse
import json
import platform
import socket
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union


def _git_short_commit() -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if out.returncode == 0:
            value = out.stdout.strip()
            if value:
                return value
    except Exception:
        pass
    return None


def _git_branch() -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if out.returncode == 0:
            value = out.stdout.strip()
            if value:
                return value
    except Exception:
        pass
    return None


def _serialize(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: _serialize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize(v) for v in value]
    return value


def save_run_params(
    args: Union[argparse.Namespace, Dict[str, Any]],
    out_dir: Union[str, Path],
    prefix: str = "run",
) -> Dict[str, Path]:
    """Save JSON/YAML parameter snapshots with runtime metadata."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    params = vars(args).copy() if isinstance(args, argparse.Namespace) else dict(args)
    params = {k: _serialize(v) for k, v in params.items()}

    metadata: Dict[str, Any] = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
    }

    commit = _git_short_commit()
    branch = _git_branch()
    if commit:
        metadata["git_commit"] = commit
    if branch:
        metadata["git_branch"] = branch

    payload = {"_metadata": metadata, **params}
    json_path = out_dir / f"{prefix}_params.json"
    yaml_path = out_dir / f"{prefix}_params.yaml"

    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)

    try:
        import yaml  # type: ignore

        with open(yaml_path, "w", encoding="utf-8") as handle:
            yaml.dump(payload, handle, default_flow_style=False, sort_keys=False, allow_unicode=True)
    except Exception:
        with open(yaml_path, "w", encoding="utf-8") as handle:
            handle.write(f"# Saved at {metadata['timestamp']}\n")
            for key, value in payload.items():
                handle.write(f"{key}: {value}\n")

    return {"json": json_path, "yaml": yaml_path}


def load_run_params(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a JSON params snapshot produced by save_run_params."""
    with open(Path(path), "r", encoding="utf-8") as handle:
        return json.load(handle)

