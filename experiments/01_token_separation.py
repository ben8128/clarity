"""Experiment 01: Token Separation Validation.

Validates the core Clarity hypothesis — that Mimi's semantic tokens (codebook 0) alone
produce intelligible speech.

Produces four reconstructions:
  (a) Full reconstruction (all codebooks) — baseline
  (b) Semantic-only (codebook 0, rest zeroed) — the Clarity hypothesis
  (c) Acoustic-only (codebook 0 zeroed, rest kept) — what we discard
  (d) First-half semantic + second-half acoustic — boundary analysis
"""

import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.codec import MimiCodec
from src.quality import quality_report
from src.utils import (
    audio_stats,
    download_librispeech_sample,
    load_audio,
    plot_waveform_and_spectrogram,
    save_audio,
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
AUDIO_DIR = Path(__file__).resolve().parent.parent / "audio"


def find_test_audio() -> tuple[np.ndarray, int]:
    """Find a test audio file. Uses local audio if available, else downloads LibriSpeech."""
    # Look for local audio files
    if AUDIO_DIR.exists():
        for ext in ("*.wav", "*.mp3", "*.flac"):
            files = list(AUDIO_DIR.glob(ext))
            if files:
                print(f"Using local audio: {files[0]}")
                return load_audio(files[0], target_sr=24000)

    # Fall back to LibriSpeech
    print("No local audio found. Downloading LibriSpeech sample...")
    return download_librispeech_sample()


def run_experiment():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print("=" * 70)
    print("EXPERIMENT 01: Token Separation Validation")
    print("=" * 70)

    # Step 1: Load audio
    print("\n--- Step 1: Loading test audio ---")
    audio, sr = find_test_audio()
    stats = audio_stats(audio, sr)
    print(f"Audio stats: {stats}")

    # Step 2: Initialize codec and encode
    print("\n--- Step 2: Encoding with Mimi ---")
    codec = MimiCodec(device="cpu")
    tokens = codec.encode(audio, sr=sr)
    info = codec.report_codebook_info(tokens)

    # Step 3: Create four reconstructions
    print("\n--- Step 3: Creating reconstructions ---")

    # (a) Full reconstruction
    print("  (a) Full reconstruction (all codebooks)...")
    full_recon = codec.decode(tokens)

    # (b) Semantic-only
    print("  (b) Semantic-only (codebook 0, rest zeroed)...")
    semantic_recon = codec.reconstruct_semantic_only(tokens)

    # (c) Acoustic-only
    print("  (c) Acoustic-only (codebook 0 zeroed, rest kept)...")
    acoustic_recon = codec.reconstruct_acoustic_only(tokens)

    # (d) First-half semantic + second-half acoustic
    print("  (d) Half-half construction...")
    num_frames = tokens.shape[2]
    mid = num_frames // 2
    half_tokens = tokens.clone()
    # First half: keep only semantic (zero acoustic)
    half_tokens[:, 1:, :mid] = 0
    # Second half: keep only acoustic (zero semantic)
    half_tokens[:, 0:1, mid:] = 0
    half_recon = codec.decode(half_tokens)

    # Step 4: Save all audio files
    print("\n--- Step 4: Saving audio files ---")
    save_audio(audio, RESULTS_DIR / f"01_original_{timestamp}.wav", sr)
    save_audio(full_recon, RESULTS_DIR / f"01_full_reconstruction_{timestamp}.wav", sr)
    save_audio(semantic_recon, RESULTS_DIR / f"01_semantic_only_{timestamp}.wav", sr)
    save_audio(acoustic_recon, RESULTS_DIR / f"01_acoustic_only_{timestamp}.wav", sr)
    save_audio(half_recon, RESULTS_DIR / f"01_half_half_{timestamp}.wav", sr)

    # Step 5: Compute quality metrics
    print("\n--- Step 5: Computing quality metrics ---")
    reconstructions = {
        "Full (all codebooks)": full_recon,
        "Semantic-only (cb 0)": semantic_recon,
        "Acoustic-only (cb 1+)": acoustic_recon,
        "Half-half": half_recon,
    }
    metrics = quality_report(audio, reconstructions, sr=sr)

    # Step 6: Generate comparison plot
    print("\n--- Step 6: Generating comparison plot ---")
    plot_data = {
        "Original": (audio, sr),
        "Full Recon": (full_recon, sr),
        "Semantic Only": (semantic_recon, sr),
        "Acoustic Only": (acoustic_recon, sr),
    }
    plot_waveform_and_spectrogram(
        plot_data,
        title="Experiment 01: Token Separation Comparison",
        save_path=RESULTS_DIR / f"01_comparison_{timestamp}.png",
    )

    # Step 7: Bitrate calculations
    print("\n--- Step 7: Bitrate Summary ---")
    num_codebooks = info["num_codebooks"]
    bits_per_token = info["bits_per_token"]
    frame_rate = info["frame_rate_hz"]

    bitrates = {
        "Semantic-only (1 cb)": frame_rate * 1 * bits_per_token,
        "2 codebooks": frame_rate * 2 * bits_per_token,
        "4 codebooks": frame_rate * 4 * bits_per_token,
        f"Full ({num_codebooks} cb)": frame_rate * num_codebooks * bits_per_token,
        "Opus (typical voice)": 24000,  # 24 kbps typical
        "Raw PCM 24kHz 16bit": 24000 * 16,
    }

    print(f"\n{'Configuration':<25} {'Bitrate':>12} {'vs Raw PCM':>12} {'vs Opus':>12}")
    print("-" * 65)
    raw_bps = bitrates["Raw PCM 24kHz 16bit"]
    opus_bps = bitrates["Opus (typical voice)"]
    for label, bps in bitrates.items():
        if bps >= 1000:
            rate_str = f"{bps/1000:.1f} kbps"
        else:
            rate_str = f"{bps:.1f} bps"
        compression = f"{raw_bps / bps:.0f}x" if bps > 0 else "N/A"
        vs_opus = f"{opus_bps / bps:.1f}x" if bps > 0 else "N/A"
        print(f"{label:<25} {rate_str:>12} {compression:>12} {vs_opus:>12}")

    # Step 8: Summary
    print("\n" + "=" * 70)
    print("EXPERIMENT 01 SUMMARY")
    print("=" * 70)
    print(f"\nMimi uses {num_codebooks} codebooks (paper claims 8).")
    print(f"Semantic-only bitrate: {frame_rate * bits_per_token:.1f} bps")
    print(f"Full Mimi bitrate: {frame_rate * num_codebooks * bits_per_token:.1f} bps")

    sem_metrics = metrics.get("Semantic-only (cb 0)", {})
    full_metrics = metrics.get("Full (all codebooks)", {})

    print(f"\nKey metrics:")
    print(f"  Full reconstruction  — PESQ: {full_metrics.get('pesq', 'N/A')}, "
          f"STOI: {full_metrics.get('stoi', 'N/A')}")
    print(f"  Semantic-only        — PESQ: {sem_metrics.get('pesq', 'N/A')}, "
          f"STOI: {sem_metrics.get('stoi', 'N/A')}")

    stoi_val = sem_metrics.get("stoi", 0)
    if isinstance(stoi_val, float) and not np.isnan(stoi_val):
        if stoi_val > 0.5:
            print(f"\n✓ STOI > 0.5 ({stoi_val:.3f}): Semantic-only reconstruction "
                  "appears intelligible. Core hypothesis shows promise.")
        else:
            print(f"\n✗ STOI ≤ 0.5 ({stoi_val:.3f}): Semantic-only reconstruction "
                  "may not be intelligible. Consider using more codebooks (Experiment 02).")

    print(f"\nOutput files saved to: {RESULTS_DIR}/")
    print("LISTEN to 01_semantic_only_*.wav — your ears are the ultimate judge.")
    print("=" * 70)

    return metrics


if __name__ == "__main__":
    run_experiment()
