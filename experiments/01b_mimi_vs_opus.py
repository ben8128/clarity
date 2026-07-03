"""Experiment 01b: Mimi vs Opus — Codec Comparison & Noise Resilience.

Tests the real finding: Mimi at 4.4 kbps delivers near-perfect quality compared to
Opus at 24 kbps, and token-based reconstruction may strip background noise.

Tests:
  1. Mimi (full 32 codebooks, 4.4 kbps) vs Opus at various bitrates
  2. Quality sweep across codebook counts (1, 2, 4, 8, 16, 32)
  3. Noisy audio through Mimi — does token reconstruction denoise?
"""

import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.codec import MimiCodec
from src.quality import compute_pesq, compute_stoi
from src.utils import (
    audio_stats,
    download_librispeech_sample,
    load_audio,
    plot_waveform_and_spectrogram,
    resample,
    save_audio,
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
AUDIO_DIR = Path(__file__).resolve().parent.parent / "audio"


def encode_opus(audio: np.ndarray, sr: int, bitrate_kbps: int) -> np.ndarray:
    """Encode/decode audio through Opus at a given bitrate using ffmpeg.

    Args:
        audio: Input audio array (mono float32).
        sr: Sample rate.
        bitrate_kbps: Target Opus bitrate in kbps.

    Returns:
        Audio after Opus encode→decode round-trip.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        in_path = Path(tmpdir) / "input.wav"
        opus_path = Path(tmpdir) / "encoded.opus"
        out_path = Path(tmpdir) / "output.wav"

        save_audio(audio, in_path, sr)

        # Encode to Opus
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(in_path),
                "-c:a", "libopus", "-b:a", f"{bitrate_kbps}k",
                "-ar", str(sr), "-ac", "1",
                str(opus_path),
            ],
            capture_output=True, check=True,
        )

        # Decode back to WAV
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(opus_path),
                "-ar", str(sr), "-ac", "1",
                str(out_path),
            ],
            capture_output=True, check=True,
        )

        result, _ = load_audio(out_path, target_sr=sr)
        return result


def run_quality_sweep(codec: MimiCodec, tokens, original: np.ndarray, sr: int, label: str, timestamp: str):
    """Run quality sweep across codebook counts and save results."""
    codebook_counts = [1, 2, 4, 8, 16, 32]
    results = {}

    print(f"\n--- Quality Sweep: {label} ---")
    print(f"{'Codebooks':<12} {'Bitrate':>10} {'PESQ':>8} {'STOI':>8}")
    print("-" * 42)

    for n_cb in codebook_counts:
        if n_cb > tokens.shape[1]:
            continue

        modified = tokens.clone()
        modified[:, n_cb:, :] = 0
        recon = codec.decode(modified)

        # Align lengths for comparison
        min_len = min(len(original), len(recon))
        ref = original[:min_len]
        deg = recon[:min_len]

        try:
            pesq_score = compute_pesq(ref, deg, sr)
        except Exception:
            pesq_score = float("nan")

        try:
            stoi_score = compute_stoi(ref, deg, sr)
        except Exception:
            stoi_score = float("nan")

        bitrate = 12.5 * n_cb * 11
        results[n_cb] = {"pesq": pesq_score, "stoi": stoi_score, "bitrate_bps": bitrate}

        pesq_str = f"{pesq_score:.3f}" if not np.isnan(pesq_score) else "N/A"
        stoi_str = f"{stoi_score:.3f}" if not np.isnan(stoi_score) else "N/A"
        bitrate_str = f"{bitrate:.0f} bps" if bitrate < 1000 else f"{bitrate/1000:.1f} kbps"
        print(f"{n_cb:<12} {bitrate_str:>10} {pesq_str:>8} {stoi_str:>8}")

        # Save audio for key codebook counts
        if n_cb in (1, 2, 4, 8, 32):
            save_audio(recon, RESULTS_DIR / f"01b_{label}_cb{n_cb:02d}_{timestamp}.wav", sr)

    return results


def run_experiment():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("EXPERIMENT 01b: Mimi vs Opus — Codec Comparison & Noise Resilience")
    print("=" * 70)

    # --- Load audio ---
    audio_files = {}

    # Clean audio (LibriSpeech)
    print("\n--- Loading clean audio (LibriSpeech) ---")
    clean_audio, sr = download_librispeech_sample()
    audio_files["clean"] = clean_audio
    save_audio(clean_audio, RESULTS_DIR / f"01b_clean_original_{timestamp}.wav", sr)

    # Noisy audio (airplane recording)
    airplane_path = AUDIO_DIR / "airplane_handheld.wav"
    if airplane_path.exists():
        print(f"\n--- Loading noisy audio ({airplane_path}) ---")
        noisy_audio, _ = load_audio(airplane_path, target_sr=sr)
        audio_files["airplane"] = noisy_audio
        stats = audio_stats(noisy_audio, sr)
        print(f"Airplane audio stats: {stats}")
        save_audio(noisy_audio, RESULTS_DIR / f"01b_airplane_original_{timestamp}.wav", sr)
    else:
        print(f"\nNo airplane audio found at {airplane_path}, skipping noisy test.")

    # --- Initialize Mimi ---
    print("\n--- Loading Mimi model ---")
    codec = MimiCodec(device="cpu")

    # === TEST 1: Mimi vs Opus on clean audio ===
    print("\n" + "=" * 70)
    print("TEST 1: Mimi vs Opus on Clean Audio")
    print("=" * 70)

    tokens_clean = codec.encode(clean_audio, sr=sr)
    codec.report_codebook_info(tokens_clean)

    # Mimi full reconstruction
    mimi_full = codec.decode(tokens_clean)
    save_audio(mimi_full, RESULTS_DIR / f"01b_clean_mimi_full_{timestamp}.wav", sr)

    # Opus at various bitrates
    opus_bitrates = [6, 12, 24, 48]
    opus_results = {}

    for kbps in opus_bitrates:
        print(f"  Encoding clean audio with Opus at {kbps} kbps...")
        try:
            opus_audio = encode_opus(clean_audio, sr, kbps)
            save_audio(opus_audio, RESULTS_DIR / f"01b_clean_opus_{kbps}k_{timestamp}.wav", sr)
            opus_results[kbps] = opus_audio
        except Exception as e:
            print(f"    Opus {kbps}k failed: {e}")

    # Compare quality
    print(f"\n{'Codec':<25} {'Bitrate':>10} {'PESQ':>8} {'STOI':>8}")
    print("-" * 55)

    min_len = min(len(clean_audio), len(mimi_full))
    mimi_pesq = compute_pesq(clean_audio[:min_len], mimi_full[:min_len], sr)
    mimi_stoi = compute_stoi(clean_audio[:min_len], mimi_full[:min_len], sr)
    print(f"{'Mimi (32 cb)':<25} {'4.4 kbps':>10} {mimi_pesq:>8.3f} {mimi_stoi:>8.3f}")

    for kbps, opus_audio in opus_results.items():
        min_len = min(len(clean_audio), len(opus_audio))
        pesq_s = compute_pesq(clean_audio[:min_len], opus_audio[:min_len], sr)
        stoi_s = compute_stoi(clean_audio[:min_len], opus_audio[:min_len], sr)
        print(f"{'Opus ' + str(kbps) + 'k':<25} {str(kbps) + ' kbps':>10} {pesq_s:>8.3f} {stoi_s:>8.3f}")

    # === TEST 2: Quality sweep on clean audio ===
    print("\n" + "=" * 70)
    print("TEST 2: Quality Sweep — Codebook Count vs Quality (Clean)")
    print("=" * 70)

    sweep_clean = run_quality_sweep(codec, tokens_clean, clean_audio, sr, "clean", timestamp)

    # === TEST 3: Noisy audio through Mimi ===
    if "airplane" in audio_files:
        noisy_audio = audio_files["airplane"]

        print("\n" + "=" * 70)
        print("TEST 3: Airplane Audio — Mimi Noise Resilience")
        print("=" * 70)

        tokens_noisy = codec.encode(noisy_audio, sr=sr)

        # Full Mimi reconstruction of noisy audio
        mimi_noisy_full = codec.decode(tokens_noisy)
        save_audio(mimi_noisy_full, RESULTS_DIR / f"01b_airplane_mimi_full_{timestamp}.wav", sr)

        # Opus comparison on noisy audio
        print(f"\n{'Codec':<25} {'Bitrate':>10} {'Note':>30}")
        print("-" * 70)
        print(f"{'Mimi (32 cb)':<25} {'4.4 kbps':>10} {'→ Listen for noise reduction':>30}")

        for kbps in opus_bitrates:
            try:
                opus_noisy = encode_opus(noisy_audio, sr, kbps)
                save_audio(opus_noisy, RESULTS_DIR / f"01b_airplane_opus_{kbps}k_{timestamp}.wav", sr)
                print(f"{'Opus ' + str(kbps) + 'k':<25} {str(kbps) + ' kbps':>10} {'→ Opus preserves noise':>30}")
            except Exception as e:
                print(f"  Opus {kbps}k failed: {e}")

        # Quality sweep on noisy audio
        print("\n--- Quality Sweep on Airplane Audio ---")
        sweep_noisy = run_quality_sweep(codec, tokens_noisy, noisy_audio, sr, "airplane", timestamp)

        # Generate comparison plot: original noisy vs Mimi reconstructed
        plot_waveform_and_spectrogram(
            {
                "Airplane Original": (noisy_audio, sr),
                "Mimi Full (4.4kbps)": (mimi_noisy_full, sr),
            },
            title="Airplane Audio: Original vs Mimi Reconstruction",
            save_path=RESULTS_DIR / f"01b_airplane_comparison_{timestamp}.png",
        )

    # === Summary ===
    print("\n" + "=" * 70)
    print("EXPERIMENT 01b SUMMARY")
    print("=" * 70)

    print(f"""
KEY FINDINGS:
  1. Mimi at 4.4 kbps vs Opus at 24 kbps on clean speech:
     Mimi — PESQ: {mimi_pesq:.3f}, STOI: {mimi_stoi:.3f}
     (Compare by listening to the saved audio files)

  2. Quality sweep: Check results above for the codebook "knee"
     — How few codebooks can we use and still sound good?

  3. Noise resilience: Listen to airplane audio files!
     — Does Mimi strip airplane cabin noise via token reconstruction?
     — Compare 01b_airplane_original vs 01b_airplane_mimi_full

OUTPUT FILES: {RESULTS_DIR}/01b_*_{timestamp}.*
Listen to the audio files — your ears are the ultimate judge.
""")
    print("=" * 70)


if __name__ == "__main__":
    run_experiment()
