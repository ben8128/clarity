"""Persist experiment metrics to results/metrics/ as versioned JSON.

Every experiment MUST save its metrics through save_metrics() — stdout-only
results are not recoverable (the 2026-02-18 runs of 01b/01c were lost this way).
"""

import json
import platform
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
METRICS_DIR = RESULTS_DIR / "metrics"


def _git_sha() -> Optional[str]:
    """Return the current git commit SHA, or None if unavailable."""
    try:
        return (
            subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
                cwd=Path(__file__).resolve().parent,
            )
            .stdout.strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _library_versions() -> dict:
    """Collect versions of the libraries that affect experiment results."""
    versions = {"python": platform.python_version()}
    for lib in ("torch", "transformers", "numpy", "pesq", "pystoi"):
        try:
            module = __import__(lib)
            versions[lib] = getattr(module, "__version__", "unknown")
        except ImportError:
            versions[lib] = None
    return versions


def save_metrics(
    experiment_id: str, payload: dict, metrics_dir: Optional[Path] = None
) -> Path:
    """Save an experiment's metrics payload as timestamped + latest JSON.

    Args:
        experiment_id: Short experiment identifier, e.g. "01_codebook_sweep".
        payload: JSON-serializable dict of metrics and metadata.
        metrics_dir: Override output directory (used in tests).

    Returns:
        Path to the timestamped JSON file that was written.
    """
    out_dir = metrics_dir if metrics_dir is not None else METRICS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    record = {
        "experiment_id": experiment_id,
        "timestamp": datetime.now().isoformat(),
        "git_sha": _git_sha(),
        "device": platform.node(),
        "versions": _library_versions(),
        "metrics": payload,
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped = out_dir / f"{experiment_id}_{timestamp}.json"
    latest = out_dir / f"{experiment_id}_latest.json"

    text = json.dumps(record, indent=2, default=_json_fallback)
    timestamped.write_text(text)
    latest.write_text(text)

    print(f"Metrics saved: {timestamped}")
    return timestamped


def load_latest(experiment_id: str, metrics_dir: Optional[Path] = None) -> dict:
    """Load the most recent metrics record for an experiment.

    Args:
        experiment_id: Experiment identifier used at save time.
        metrics_dir: Override directory (used in tests).

    Returns:
        The full record dict (with "metrics" holding the payload).

    Raises:
        FileNotFoundError: If no metrics were ever saved for this experiment.
    """
    out_dir = metrics_dir if metrics_dir is not None else METRICS_DIR
    latest = out_dir / f"{experiment_id}_latest.json"
    if not latest.exists():
        raise FileNotFoundError(f"No saved metrics for experiment '{experiment_id}'")
    return json.loads(latest.read_text())


def _json_fallback(obj: Any) -> Any:
    """Serialize numpy scalars/arrays and Paths that json can't handle."""
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if hasattr(obj, "item"):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
