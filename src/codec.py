"""Mimi neural audio codec wrapper for Clarity experiments."""

from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch

MODEL_ID = "kyutai/mimi"


class MimiCodec:
    """Wrapper around the Kyutai Mimi neural audio codec.

    Provides encode/decode/token-manipulation methods for Clarity experiments.
    """

    def __init__(self, device: Optional[str] = None) -> None:
        """Load the Mimi model and feature extractor.

        Args:
            device: PyTorch device string. Auto-detected if None.
        """
        from transformers import AutoFeatureExtractor, MimiModel

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = device
        print(f"Loading Mimi model ({MODEL_ID}) on {device}...")

        try:
            self.model = MimiModel.from_pretrained(MODEL_ID).to(device)
        except OSError:
            print("Network unavailable, loading from cache...")
            self.model = MimiModel.from_pretrained(MODEL_ID, local_files_only=True).to(device)
        self.model.eval()
        try:
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID)
        except OSError:
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID, local_files_only=True)

        self._num_codebooks: Optional[int] = None
        print("Mimi model loaded successfully.")

    @property
    def num_codebooks(self) -> Optional[int]:
        """Number of codebooks used by the model (known after first encode)."""
        return self._num_codebooks

    @property
    def sample_rate(self) -> int:
        """Expected input sample rate."""
        return 24000

    @torch.no_grad()
    def encode(self, audio: Union[str, Path, np.ndarray], sr: int = 24000) -> torch.Tensor:
        """Encode audio to Mimi token tensor.

        Args:
            audio: Path to audio file or numpy array (24kHz mono float32).
            sr: Sample rate of the input array (ignored if audio is a path).

        Returns:
            Token tensor of shape (1, num_codebooks, num_frames).
        """
        if isinstance(audio, (str, Path)):
            from .utils import load_audio

            audio, sr = load_audio(audio, target_sr=self.sample_rate)

        if sr != self.sample_rate:
            from .utils import resample

            audio = resample(audio, sr, self.sample_rate)

        inputs = self.feature_extractor(
            raw_audio=audio, sampling_rate=self.sample_rate, return_tensors="pt"
        )
        input_values = inputs["input_values"].to(self.device)

        encoder_outputs = self.model.encode(input_values)
        audio_codes = encoder_outputs.audio_codes  # (batch, num_codebooks, num_frames)

        if self._num_codebooks is None:
            self._num_codebooks = audio_codes.shape[1]
            print(f"Mimi codebook count: {self._num_codebooks}")

        return audio_codes

    @torch.no_grad()
    def decode(self, tokens: torch.Tensor) -> np.ndarray:
        """Decode token tensor back to audio.

        Args:
            tokens: Token tensor of shape (batch, num_codebooks, num_frames).

        Returns:
            Audio numpy array (mono, float32, 24kHz).
        """
        tokens = tokens.to(self.device)
        decoded = self.model.decode(tokens)
        audio = decoded.audio_values.squeeze().cpu().numpy()
        return audio

    def extract_semantic(self, tokens: torch.Tensor) -> torch.Tensor:
        """Extract semantic tokens only (codebook 0).

        Args:
            tokens: Full token tensor (batch, num_codebooks, num_frames).

        Returns:
            Semantic token tensor (batch, 1, num_frames).
        """
        return tokens[:, 0:1, :]

    def extract_acoustic(self, tokens: torch.Tensor) -> torch.Tensor:
        """Extract acoustic tokens (codebooks 1+).

        Args:
            tokens: Full token tensor (batch, num_codebooks, num_frames).

        Returns:
            Acoustic token tensor (batch, num_codebooks-1, num_frames).
        """
        return tokens[:, 1:, :]

    def reconstruct_with_n_codebooks(self, tokens: torch.Tensor, n: int) -> np.ndarray:
        """Reconstruct audio using only the first n codebooks (truncating the rest).

        Truncation (not zeroing) is essential: token id 0 is a valid codebook
        entry, so zeroing dropped codebooks would make the RVQ decoder sum in
        wrong embeddings instead of omitting those quantizer layers.

        Args:
            tokens: Full token tensor (batch, num_codebooks, num_frames).
            n: Number of codebooks to keep (1 = semantic only, all = full quality).

        Returns:
            Reconstructed audio numpy array.

        Raises:
            ValueError: If n is out of valid range.
        """
        num_codebooks = tokens.shape[1]
        if not (1 <= n <= num_codebooks):
            raise ValueError(
                f"n must be between 1 and {num_codebooks}, got {n}"
            )
        return self.decode(tokens[:, :n, :])

    def reconstruct_semantic_only(self, tokens: torch.Tensor) -> np.ndarray:
        """Reconstruct audio using only semantic tokens (codebook 0).

        Args:
            tokens: Full token tensor.

        Returns:
            Reconstructed audio numpy array.
        """
        return self.reconstruct_with_n_codebooks(tokens, 1)

    @staticmethod
    def get_bitrate(n_codebooks: int) -> float:
        """Calculate bitrate for a given number of codebooks.

        Args:
            n_codebooks: Number of codebooks transmitted.

        Returns:
            Bitrate in bits per second.
        """
        return 12.5 * n_codebooks * 11

    @staticmethod
    def get_bandwidth_savings_vs_opus(
        n_codebooks: int, opus_bitrate: int = 24000
    ) -> float:
        """Calculate bandwidth savings compared to Opus.

        Args:
            n_codebooks: Number of Mimi codebooks transmitted.
            opus_bitrate: Opus bitrate in bps (default 24 kbps).

        Returns:
            Savings as a fraction (e.g. 0.96 means 96% smaller than Opus).
        """
        return 1.0 - (MimiCodec.get_bitrate(n_codebooks) / opus_bitrate)

    def reconstruct_acoustic_only(self, tokens: torch.Tensor) -> np.ndarray:
        """Reconstruct audio using only acoustic tokens, zeroing codebook 0.

        Args:
            tokens: Full token tensor.

        Returns:
            Reconstructed audio numpy array.
        """
        modified = tokens.clone()
        modified[:, 0:1, :] = 0
        return self.decode(modified)

    def reconstruct_with_modified_acoustic(
        self, semantic_tokens: torch.Tensor, acoustic_tokens: torch.Tensor
    ) -> np.ndarray:
        """Combine semantic and acoustic tokens from potentially different sources and decode.

        Args:
            semantic_tokens: Semantic tokens (batch, 1, num_frames).
            acoustic_tokens: Acoustic tokens (batch, num_codebooks-1, num_frames).

        Returns:
            Reconstructed audio numpy array.
        """
        # Handle frame length mismatch by truncating to shorter
        min_frames = min(semantic_tokens.shape[2], acoustic_tokens.shape[2])
        sem = semantic_tokens[:, :, :min_frames]
        aco = acoustic_tokens[:, :, :min_frames]

        combined = torch.cat([sem, aco], dim=1)
        return self.decode(combined)

    def report_codebook_info(self, tokens: torch.Tensor) -> dict:
        """Report information about the token tensor structure.

        Args:
            tokens: Token tensor from encode().

        Returns:
            Dict with codebook structure details and bitrate calculations.
        """
        num_codebooks = tokens.shape[1]
        num_frames = tokens.shape[2]
        entries_per_codebook = 2048
        bits_per_token = 11  # log2(2048) ≈ 11
        frame_rate = 12.5  # Hz

        full_bitrate = frame_rate * num_codebooks * bits_per_token
        semantic_bitrate = frame_rate * 1 * bits_per_token

        info = {
            "num_codebooks": num_codebooks,
            "num_frames": num_frames,
            "entries_per_codebook": entries_per_codebook,
            "bits_per_token": bits_per_token,
            "frame_rate_hz": frame_rate,
            "full_bitrate_bps": full_bitrate,
            "semantic_only_bitrate_bps": semantic_bitrate,
            "token_shape": list(tokens.shape),
        }

        print(f"\n--- Mimi Codebook Info ---")
        print(f"  Codebooks:           {num_codebooks}")
        print(f"  Frames:              {num_frames}")
        print(f"  Entries/codebook:    {entries_per_codebook}")
        print(f"  Bits/token:          {bits_per_token}")
        print(f"  Frame rate:          {frame_rate} Hz")
        print(f"  Full bitrate:        {full_bitrate:.1f} bps")
        print(f"  Semantic-only rate:  {semantic_bitrate:.1f} bps")
        print(f"  Token tensor shape:  {list(tokens.shape)}")
        print(f"-------------------------\n")

        return info
