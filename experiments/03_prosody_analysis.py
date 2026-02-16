"""Experiment 03: Prosody & Emotion Analysis.

Tests whether semantic tokens preserve HOW something is said, not just WHAT.
Compares pitch contours (F0) between original and semantic-only reconstruction.
"""

import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.codec import MimiCodec
from src.utils import download_librispeech_sample, load_audio, save_audio

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
AUDIO_DIR = Path(__file__).resolve().parent.parent / "audio"


def extract_pitch(audio: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """Extract pitch contour (F0) using librosa's pyin.

    Returns:
        Tuple of (times, f0_values). f0_values may contain NaN for unvoiced frames.
    """
    import librosa

    f0, voiced_flag, voiced_probs = librosa.pyin(
        audio, fmin=50, fmax=500, sr=sr
    )
    times = librosa.times_like(f0, sr=sr)
    return times, f0


def pitch_correlation(f0_a: np.ndarray, f0_b: np.ndarray) -> float:
    """Compute correlation between two pitch contours, ignoring NaN frames."""
    # Align lengths
    min_len = min(len(f0_a), len(f0_b))
    a = f0_a[:min_len]
    b = f0_b[:min_len]

    # Only compare frames where both are voiced
    mask = ~(np.isnan(a) | np.isnan(b))
    if mask.sum() < 10:
        return float("nan")

    return float(np.corrcoef(a[mask], b[mask])[0, 1])


def run_experiment():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print("=" * 70)
    print("EXPERIMENT 03: Prosody & Emotion Analysis")
    print("=" * 70)

    codec = MimiCodec(device="cpu")

    # Collect test audio
    test_samples = {}

    # Check for local emotional audio
    emotional_files = {
        "happy": "emotional_happy",
        "question": "emotional_question",
        "frustrated": "emotional_frustrated",
    }

    for label, prefix in emotional_files.items():
        if AUDIO_DIR.exists():
            matches = list(AUDIO_DIR.glob(f"{prefix}.*"))
            if matches:
                audio, sr = load_audio(matches[0])
                test_samples[label] = (audio, sr)

    # Always include a LibriSpeech sample as baseline
    print("Loading LibriSpeech baseline sample...")
    audio, sr = download_librispeech_sample()
    test_samples["librispeech_baseline"] = (audio, sr)

    results = {}

    for label, (audio, sr) in test_samples.items():
        print(f"\n--- Processing: {label} ---")

        # Encode and reconstruct
        tokens = codec.encode(audio, sr=sr)
        full_recon = codec.decode(tokens)
        semantic_recon = codec.reconstruct_semantic_only(tokens)

        # Extract pitch contours
        print(f"  Extracting pitch contours...")
        t_orig, f0_orig = extract_pitch(audio, sr)
        t_full, f0_full = extract_pitch(full_recon, sr)
        t_sem, f0_sem = extract_pitch(semantic_recon, sr)

        # Compute correlations
        corr_full = pitch_correlation(f0_orig, f0_full)
        corr_semantic = pitch_correlation(f0_orig, f0_sem)

        results[label] = {
            "pitch_corr_full": corr_full,
            "pitch_corr_semantic": corr_semantic,
        }

        print(f"  Pitch correlation (full recon):     {corr_full:.3f}")
        print(f"  Pitch correlation (semantic-only):   {corr_semantic:.3f}")

        # Save audio
        save_audio(semantic_recon, RESULTS_DIR / f"03_{label}_semantic_{timestamp}.wav", sr)

        # Plot pitch contours
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

            axes[0].plot(t_orig, f0_orig, "b-", alpha=0.7)
            axes[0].set_ylabel("F0 (Hz)")
            axes[0].set_title(f"{label} — Original")
            axes[0].set_ylim(50, 400)

            axes[1].plot(t_full, f0_full, "g-", alpha=0.7)
            axes[1].set_ylabel("F0 (Hz)")
            axes[1].set_title(f"{label} — Full Reconstruction (corr={corr_full:.3f})")
            axes[1].set_ylim(50, 400)

            axes[2].plot(t_sem, f0_sem, "r-", alpha=0.7)
            axes[2].set_ylabel("F0 (Hz)")
            axes[2].set_title(f"{label} — Semantic Only (corr={corr_semantic:.3f})")
            axes[2].set_ylim(50, 400)
            axes[2].set_xlabel("Time (s)")

            plt.tight_layout()
            plt.savefig(RESULTS_DIR / f"03_{label}_pitch_{timestamp}.png", dpi=150)
            plt.close()
        except Exception as e:
            print(f"  Plotting failed: {e}")

    # Summary
    print("\n" + "=" * 65)
    print(f"{'Sample':<25} {'Full Corr':>12} {'Semantic Corr':>14}")
    print("-" * 65)
    for label, r in results.items():
        fc = f"{r['pitch_corr_full']:.3f}" if not np.isnan(r["pitch_corr_full"]) else "N/A"
        sc = f"{r['pitch_corr_semantic']:.3f}" if not np.isnan(r["pitch_corr_semantic"]) else "N/A"
        print(f"{label:<25} {fc:>12} {sc:>14}")
    print("=" * 65)

    return results


if __name__ == "__main__":
    run_experiment()
