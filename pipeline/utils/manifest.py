"""Pipeline reproducibility manifest — tracks inputs, code version, and environment.

Every pipeline run generates a JSON manifest alongside the outputs so that
any result can be traced back to exact inputs, code, and package versions.
"""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


def _sha256(filepath: Path) -> str:
    """Compute SHA256 checksum of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_info() -> dict:
    """Get current git commit hash and branch."""
    info: dict = {}
    try:
        info["commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True,
        ).strip()
        info["branch"] = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL, text=True,
        ).strip()
        info["dirty"] = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, text=True,
        ).strip())
    except Exception:
        info["commit"] = "unknown"
        info["branch"] = "unknown"
        info["dirty"] = None
    return info


def _package_versions() -> dict[str, str]:
    """Get versions of key installed packages."""
    pkgs: dict[str, str] = {}
    for name in ["pandas", "numpy", "pyarrow", "torch", "esm", "evo2", "scikit-learn", "requests"]:
        try:
            mod = __import__(name)
            pkgs[name] = getattr(mod, "__version__", "installed")
        except ImportError:
            pkgs[name] = "not installed"
    return pkgs


def generate_pipeline_manifest(
    data_dir: Path,
    output_dir: Path,
    extra: dict | None = None,
) -> dict:
    """Generate a reproducibility manifest and save to output_dir.

    Returns the manifest dict.
    """
    manifest: dict = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "git": _git_info(),
        "packages": _package_versions(),
        "input_checksums": {},
    }

    # Checksum all input data files
    if data_dir.exists():
        for ext in ("*.csv", "*.csv.gz", "*.tsv", "*.tsv.gz"):
            for f in sorted(data_dir.glob(ext)):
                manifest["input_checksums"][f.name] = _sha256(f)

    if extra:
        manifest["extra"] = extra

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "pipeline_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Pipeline manifest saved: %s", manifest_path)

    return manifest
