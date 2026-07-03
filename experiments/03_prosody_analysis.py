"""Experiment 03: Prosody vs Codebook Count.

Finds the codebook count at which prosody (pitch contour) survives transmission.
Sweeps codebook counts and measures F0 correlation between the original and each
reconstruction. Success criterion: correlation > 0.7.
"""

import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.codec import MimiCodec
from src.results_io import save_metrics
from src.utils import download_librispeech_sample, load_audio, save_audio

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
AUDIO_DIR = Path(__file__).resolve().parent.parent / "audio"

SWEEP_POINTS = [1, 2, 4, 8, 16, 32]
CORRELATION_THRESHOLD = 0.7


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


def collect_test_samples() -> dict:
    """Collect prosody test samples: local emotional recordings + LibriSpeech."""
    test_samples = {}

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

    print("Loading LibriSpeech baseline sample...")
    audio, sr = download_librispeech_sample()
    test_samples["librispeech_baseline"] = (audio, sr)
    return test_samples


def run_experiment():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print("=" * 70)
    print("EXPERIMENT 03: Prosody vs Codebook Count")
    print("=" * 70)

    codec = MimiCodec(device="cpu")
    test_samples = collect_test_samples()
    if len(test_samples) == 1:
        print("NOTE: no emotional recordings in audio/ (emotional_happy.wav etc.);")
        print("      running on LibriSpeech only. Record varied clips for a stronger test.")

    results: dict[str, dict] = {}

    for label, (audio, sr) in test_samples.items():
        print(f"\n--- Processing: {label} ---")
        tokens = codec.encode(audio, sr=sr)
        sweep_points = [n for n in SWEEP_POINTS if n <= tokens.shape[1]]

        print("  Extracting original pitch contour...")
        t_orig, f0_orig = extract_pitch(audio, sr)

        per_count = []
        contours = {}
        for n in sweep_points:
            recon = codec.reconstruct_with_n_codebooks(tokens, n)
            t_rec, f0_rec = extract_pitch(recon, sr)
            corr = pitch_correlation(f0_orig, f0_rec)
            per_count.append({"codebooks": n, "pitch_correlation": corr})
            contours[n] = (t_rec, f0_rec)
            corr_str = f"{corr:.3f}" if not np.isnan(corr) else "N/A"
            print(f"  {n:>2} codebook(s): F0 correlation = {corr_str}")
            if n in (1, 4, 32):
                save_audio(recon, RESULTS_DIR / f"03_{label}_cb{n:02d}_{timestamp}.wav", sr)

        # First count clearing the threshold
        prosody_emerges_at = next(
            (
                e["codebooks"]
                for e in per_count
                if not np.isnan(e["pitch_correlation"])
                and e["pitch_correlation"] > CORRELATION_THRESHOLD
            ),
            None,
        )
        results[label] = {
            "sweep": per_count,
            "prosody_emerges_at": prosody_emerges_at,
        }

        # Plot original vs selected reconstructions
        try:
            import matplotlib.pyplot as plt

            plot_counts = [n for n in (1, 4, 32) if n in contours]
            fig, axes = plt.subplots(
                1 + len(plot_counts), 1, figsize=(12, 3 * (1 + len(plot_counts))), sharex=True
            )
            axes[0].plot(t_orig, f0_orig, "b-", alpha=0.7)
            axes[0].set_ylabel("F0 (Hz)")
            axes[0].set_title(f"{label} — Original")
            axes[0].set_ylim(50, 400)

            for ax, n in zip(axes[1:], plot_counts):
                t_rec, f0_rec = contours[n]
                corr = next(
                    e["pitch_correlation"] for e in per_count if e["codebooks"] == n
                )
                ax.plot(t_rec, f0_rec, "r-", alpha=0.7)
                ax.set_ylabel("F0 (Hz)")
                ax.set_title(f"{label} — {n} codebook(s) (corr={corr:.3f})")
                ax.set_ylim(50, 400)
            axes[-1].set_xlabel("Time (s)")

            plt.tight_layout()
            plt.savefig(RESULTS_DIR / f"03_{label}_pitch_{timestamp}.png", dpi=150)
            plt.close()
        except Exception as e:
            print(f"  Plotting failed: {e}")

    # Summary
    print("\n" + "=" * 65)
    print(f"{'Sample':<25} {'Prosody emerges at':>20}")
    print("-" * 65)
    for label, r in results.items():
        at = r["prosody_emerges_at"]
        print(f"{label:<25} {str(at) + ' codebook(s)' if at else 'never (>0.7)':>20}")
    print("=" * 65)

    save_metrics(
        "03_prosody_sweep",
        {
            "correlation_threshold": CORRELATION_THRESHOLD,
            "per_sample": results,
        },
    )
    return results


if __name__ == "__main__":
    run_experiment()
