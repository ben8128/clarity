"""Audio I/O, resampling, visualization, and utility helpers for Clarity."""

from pathlib import Path
from typing import Optional, Union

import numpy as np
import soundfile as sf


def load_audio(path: Union[str, Path], target_sr: int = 24000) -> tuple[np.ndarray, int]:
    """Load an audio file and resample to target sample rate.

    Args:
        path: Path to audio file (.wav, .mp3, .flac, etc.)
        target_sr: Target sample rate (default 24000 for Mimi)

    Returns:
        Tuple of (audio_array, sample_rate). Audio is mono, float32, normalized.
    """
    import librosa

    path = Path(path)
    audio, sr = librosa.load(str(path), sr=target_sr, mono=True)
    return audio.astype(np.float32), sr


def save_audio(
    audio: np.ndarray, path: Union[str, Path], sr: int = 24000, verbose: bool = True
) -> Path:
    """Save audio array to a .wav file.

    Args:
        audio: Audio array (float32, mono)
        path: Output file path
        sr: Sample rate
        verbose: Print the saved path (disable for bulk writes)

    Returns:
        Path to saved file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sr)
    if verbose:
        print(f"Saved audio: {path} ({len(audio) / sr:.2f}s, {sr}Hz)")
    return path


def resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample audio to a different sample rate.

    Args:
        audio: Input audio array
        orig_sr: Original sample rate
        target_sr: Target sample rate

    Returns:
        Resampled audio array.
    """
    if orig_sr == target_sr:
        return audio
    import librosa

    return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)


def audio_stats(audio: np.ndarray, sr: int) -> dict:
    """Compute basic audio statistics.

    Args:
        audio: Audio array
        sr: Sample rate

    Returns:
        Dictionary with duration, sample_rate, rms_energy, peak, and num_samples.
    """
    return {
        "duration_s": len(audio) / sr,
        "sample_rate": sr,
        "num_samples": len(audio),
        "rms_energy": float(np.sqrt(np.mean(audio**2))),
        "peak": float(np.max(np.abs(audio))),
    }


def plot_waveform_and_spectrogram(
    audio_dict: dict[str, tuple[np.ndarray, int]],
    title: str = "Audio Comparison",
    save_path: Optional[Union[str, Path]] = None,
) -> None:
    """Plot waveforms and spectrograms for multiple audio signals side by side.

    Args:
        audio_dict: Dict mapping label → (audio_array, sample_rate)
        title: Plot title
        save_path: If provided, save the figure to this path
    """
    import matplotlib.pyplot as plt

    n = len(audio_dict)
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 6))
    if n == 1:
        axes = axes.reshape(2, 1)

    fig.suptitle(title, fontsize=14)

    for i, (label, (audio, sr)) in enumerate(audio_dict.items()):
        t = np.arange(len(audio)) / sr

        # Waveform
        axes[0, i].plot(t, audio, linewidth=0.3)
        axes[0, i].set_title(f"{label}\n(waveform)")
        axes[0, i].set_xlabel("Time (s)")
        axes[0, i].set_ylabel("Amplitude")
        axes[0, i].set_ylim(-1, 1)

        # Spectrogram
        axes[1, i].specgram(audio, Fs=sr, NFFT=1024, noverlap=512, cmap="viridis")
        axes[1, i].set_title(f"{label}\n(spectrogram)")
        axes[1, i].set_xlabel("Time (s)")
        axes[1, i].set_ylabel("Frequency (Hz)")

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
        print(f"Saved plot: {save_path}")

    plt.close(fig)


def download_librispeech_sample() -> tuple[np.ndarray, int]:
    """Download a single sample from LibriSpeech for testing.

    Returns:
        Tuple of (audio_array at 24kHz, sample_rate).
    """
    from datasets import load_dataset

    print("Downloading LibriSpeech sample...")
    ds = load_dataset(
        "hf-internal-testing/librispeech_asr_dummy",
        "clean",
        split="validation",
        trust_remote_code=True,
    )
    sample = ds[0]
    audio = np.array(sample["audio"]["array"], dtype=np.float32)
    orig_sr = sample["audio"]["sampling_rate"]

    audio_24k = resample(audio, orig_sr, 24000)
    print(f"Loaded LibriSpeech sample: {len(audio_24k) / 24000:.2f}s at 24kHz")
    return audio_24k, 24000
