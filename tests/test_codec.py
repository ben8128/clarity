"""Basic tests for the Mimi codec wrapper.

Tests are split into:
- Unit tests: test token manipulation logic with synthetic tensors (always run)
- Integration tests: require the Mimi model from HuggingFace (skipped if unavailable)
"""

import numpy as np
import pytest
import torch

from src.codec import MimiCodec


def _model_available() -> bool:
    """Check if the Mimi model can be loaded."""
    try:
        from transformers import MimiModel
        MimiModel.from_pretrained("kyutai/mimi")
        return True
    except Exception:
        return False


requires_model = pytest.mark.skipif(
    not _model_available(),
    reason="Mimi model not available (no network access or model not cached)",
)


# --- Unit tests (no model needed) ---


class TestTokenManipulation:
    """Test token extraction/manipulation using synthetic tensors."""

    def _make_tokens(self, num_codebooks: int = 8, num_frames: int = 100) -> torch.Tensor:
        """Create a synthetic token tensor."""
        return torch.randint(0, 2048, (1, num_codebooks, num_frames))

    def test_extract_semantic(self):
        codec = object.__new__(MimiCodec)  # skip __init__
        tokens = self._make_tokens(8, 50)
        semantic = codec.extract_semantic(tokens)

        assert semantic.shape == (1, 1, 50)
        assert torch.equal(semantic, tokens[:, 0:1, :])

    def test_extract_acoustic(self):
        codec = object.__new__(MimiCodec)
        tokens = self._make_tokens(8, 50)
        acoustic = codec.extract_acoustic(tokens)

        assert acoustic.shape == (1, 7, 50)
        assert torch.equal(acoustic, tokens[:, 1:, :])

    def test_extract_semantic_32_codebooks(self):
        codec = object.__new__(MimiCodec)
        tokens = self._make_tokens(32, 100)
        semantic = codec.extract_semantic(tokens)

        assert semantic.shape == (1, 1, 100)

    def test_extract_acoustic_32_codebooks(self):
        codec = object.__new__(MimiCodec)
        tokens = self._make_tokens(32, 100)
        acoustic = codec.extract_acoustic(tokens)

        assert acoustic.shape == (1, 31, 100)

    def test_codebook_info_report(self):
        codec = object.__new__(MimiCodec)
        tokens = self._make_tokens(8, 150)
        info = codec.report_codebook_info(tokens)

        assert info["num_codebooks"] == 8
        assert info["num_frames"] == 150
        assert info["bits_per_token"] == 11
        assert info["frame_rate_hz"] == 12.5
        assert info["full_bitrate_bps"] == 12.5 * 8 * 11
        assert info["semantic_only_bitrate_bps"] == 137.5

    def test_codebook_info_32_codebooks(self):
        codec = object.__new__(MimiCodec)
        tokens = self._make_tokens(32, 150)
        info = codec.report_codebook_info(tokens)

        assert info["num_codebooks"] == 32
        assert info["full_bitrate_bps"] == 12.5 * 32 * 11
        assert info["semantic_only_bitrate_bps"] == 137.5


class TestUtilityFunctions:
    """Test utility functions that don't need network access."""

    def test_audio_stats(self):
        from src.utils import audio_stats

        audio = np.random.randn(24000).astype(np.float32) * 0.5
        stats = audio_stats(audio, 24000)

        assert stats["duration_s"] == pytest.approx(1.0)
        assert stats["sample_rate"] == 24000
        assert stats["num_samples"] == 24000
        assert 0 < stats["rms_energy"] < 1
        assert stats["peak"] > 0

    def test_save_and_load_audio(self, tmp_path):
        from src.utils import load_audio, save_audio

        audio = np.random.randn(24000).astype(np.float32) * 0.1
        path = tmp_path / "test.wav"
        save_audio(audio, path, sr=24000)
        loaded, sr = load_audio(path, target_sr=24000)

        assert sr == 24000
        assert len(loaded) == len(audio)
        np.testing.assert_allclose(loaded, audio, atol=1e-4)

    def test_resample(self):
        from src.utils import resample

        audio = np.random.randn(16000).astype(np.float32)
        resampled = resample(audio, 16000, 24000)

        expected_len = int(len(audio) * 24000 / 16000)
        assert abs(len(resampled) - expected_len) <= 1


class TestQualityMetrics:
    """Test quality metric functions with synthetic audio."""

    def test_pesq_identical(self):
        from src.utils import resample
        from src.quality import compute_pesq

        # Create a tone signal (PESQ needs speech-like signal length)
        sr = 24000
        t = np.linspace(0, 2, 2 * sr, endpoint=False)
        audio = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

        score = compute_pesq(audio, audio, sr)
        assert score > 3.0, f"Identical signals should have high PESQ, got {score}"

    def test_stoi_identical(self):
        from src.quality import compute_stoi

        sr = 24000
        t = np.linspace(0, 2, 2 * sr, endpoint=False)
        audio = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

        score = compute_stoi(audio, audio, sr)
        assert score > 0.9, f"Identical signals should have high STOI, got {score}"


# --- Integration tests (require model) ---


@requires_model
class TestMimiCodecIntegration:
    """Integration tests that require the actual Mimi model."""

    @pytest.fixture(scope="class")
    def codec(self):
        return MimiCodec(device="cpu")

    @pytest.fixture(scope="class")
    def sample_audio(self):
        from src.utils import download_librispeech_sample
        return download_librispeech_sample()

    def test_encode_produces_tokens(self, codec, sample_audio):
        audio, sr = sample_audio
        tokens = codec.encode(audio, sr=sr)

        assert tokens is not None
        assert tokens.ndim == 3
        assert tokens.shape[0] == 1
        assert tokens.shape[1] > 0
        assert tokens.shape[2] > 0

    def test_decode_produces_audio(self, codec, sample_audio):
        audio, sr = sample_audio
        tokens = codec.encode(audio, sr=sr)
        reconstructed = codec.decode(tokens)

        assert isinstance(reconstructed, np.ndarray)
        assert reconstructed.ndim == 1
        assert len(reconstructed) > 0
        assert np.isfinite(reconstructed).all()
        assert np.max(np.abs(reconstructed)) > 0

    def test_semantic_only_reconstruction(self, codec, sample_audio):
        audio, sr = sample_audio
        tokens = codec.encode(audio, sr=sr)
        recon = codec.reconstruct_semantic_only(tokens)

        assert isinstance(recon, np.ndarray)
        assert len(recon) > 0
        assert np.isfinite(recon).all()
