"""Tests for reconstruction-track additions: DNSMOS, WER, degradation."""

import numpy as np
import pytest

from src.degrade import DegradeConfig, degrade_phone
from src.quality import _word_error_rate, compute_dnsmos


def _speechy_signal(seconds: float = 2.0, sr: int = 24000) -> np.ndarray:
    """Amplitude-modulated tone stack — enough structure for the metrics."""
    t = np.linspace(0, seconds, int(seconds * sr), endpoint=False)
    carrier = 0.3 * np.sin(2 * np.pi * 220 * t) + 0.15 * np.sin(2 * np.pi * 440 * t)
    envelope = 0.5 * (1 + np.sin(2 * np.pi * 3 * t))
    return (carrier * envelope).astype(np.float32)


class TestWer:
    def test_identical(self):
        assert _word_error_rate("hello there world", "hello there world") == 0.0

    def test_case_and_punctuation_insensitive(self):
        assert _word_error_rate("Hello, there!", "hello there") == 0.0

    def test_substitution(self):
        assert _word_error_rate("a b c d", "a x c d") == pytest.approx(0.25)

    def test_empty_reference(self):
        assert _word_error_rate("", "something") == 1.0
        assert _word_error_rate("", "") == 0.0


class TestDegrade:
    def test_preserves_length_and_dtype(self):
        audio = _speechy_signal()
        out = degrade_phone(audio, 24000)
        assert len(out) == len(audio)
        assert out.dtype == np.float32
        assert np.max(np.abs(out)) <= 0.99 + 1e-6

    def test_deterministic_with_seed(self):
        audio = _speechy_signal()
        a = degrade_phone(audio, 24000, DegradeConfig(seed=7))
        b = degrade_phone(audio, 24000, DegradeConfig(seed=7))
        np.testing.assert_array_equal(a, b)

    def test_actually_degrades(self):
        audio = _speechy_signal()
        out = degrade_phone(audio, 24000, DegradeConfig(snr_db=10))
        # Signal must have changed substantially (noise added, filtered)
        assert not np.allclose(out, audio, atol=1e-3)


class TestDnsmos:
    def test_returns_plausible_score(self):
        audio = _speechy_signal(seconds=3.0)
        score = compute_dnsmos(audio, 24000)
        assert 1.0 <= score <= 5.0

    def test_noise_scores_lower_than_tone(self):
        clean = _speechy_signal(seconds=3.0)
        noisy = degrade_phone(clean, 24000, DegradeConfig(snr_db=0))
        assert compute_dnsmos(noisy, 24000) <= compute_dnsmos(clean, 24000) + 0.3
