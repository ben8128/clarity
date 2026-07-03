"""Tests for results_io metrics persistence."""

import json

import numpy as np
import pytest

from src.results_io import load_latest, save_metrics


class TestSaveMetrics:
    def test_round_trip(self, tmp_path):
        payload = {"pesq": 3.2, "stoi": 0.91, "n_codebooks": 8}
        path = save_metrics("test_exp", payload, metrics_dir=tmp_path)

        assert path.exists()
        record = json.loads(path.read_text())
        assert record["experiment_id"] == "test_exp"
        assert record["metrics"] == payload
        assert "timestamp" in record
        assert "versions" in record

    def test_latest_pointer_updated(self, tmp_path):
        save_metrics("test_exp", {"run": 1}, metrics_dir=tmp_path)
        save_metrics("test_exp", {"run": 2}, metrics_dir=tmp_path)

        record = load_latest("test_exp", metrics_dir=tmp_path)
        assert record["metrics"] == {"run": 2}

    def test_numpy_values_serialized(self, tmp_path):
        payload = {
            "score": np.float32(1.5),
            "counts": np.array([1, 2, 3]),
        }
        path = save_metrics("test_np", payload, metrics_dir=tmp_path)

        record = json.loads(path.read_text())
        assert record["metrics"]["score"] == 1.5
        assert record["metrics"]["counts"] == [1, 2, 3]

    def test_load_latest_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_latest("never_saved", metrics_dir=tmp_path)
