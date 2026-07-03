"""Experiment 01c: Voice Isolation / Noise Reduction via Mimi.

Tests whether Mimi's token-based reconstruction strips background noise.

Method:
  1. Take clean speech (LibriSpeech)
  2. Add noise at various SNR levels (white noise, pink noise, babble)
  3. Run noisy audio through Mimi encode→decode
  4. Measure quality of Mimi output vs CLEAN original
  5. If Mimi output scores better than the noisy input, it's doing denoising

Also tests the real airplane recording by comparing spectral characteristics
before and after Mimi reconstruction.
"""

import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.codec import MimiCodec
from src.quality import compute_pesq, compute_stoi
from src.utils import (
    download_librispeech_sample,
    load_audio,
    plot_waveform_and_spectrogram,
    save_audio,
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
AUDIO_DIR = Path(__file__).resolve().parent.parent / "audio"


def generate_white_noise(length: int) -> np.ndarray:
    """Generate white noise."""
    return np.random.randn(length).astype(np.float32)


def generate_pink_noise(length: int) -> np.ndarray:
    """Generate pink noise (1/f) using the Voss-McCartney algorithm."""
    # Simple approximation: filter white noise
    white = np.random.randn(length).astype(np.float32)
    # Apply 1/f rolloff via cumulative filtering
    from scipy.signal import lfilter
    b = np.array([0.049922035, -0.095993537, 0.050612699, -0.004709510])
    a = np.array([1.0, -2.494956002, 2.017265875, -0.522189400])
    pink = lfilter(b, a, white).astype(np.float32)
    # Normalize
    pink = pink / (np.max(np.abs(pink)) + 1e-8)
    return pink


def generate_babble_noise(length: int, n_voices: int = 8) -> np.ndarray:
    """Generate babble noise by summing multiple modulated noise signals."""
    babble = np.zeros(length, dtype=np.float32)
    for _ in range(n_voices):
        # Each "voice" is amplitude-modulated noise
        noise = np.random.randn(length).astype(np.float32)
        # Random modulation at speech-like rate (2-8 Hz)
        mod_freq = np.random.uniform(2, 8)
        t = np.arange(length) / 24000
        envelope = 0.5 * (1 + np.sin(2 * np.pi * mod_freq * t + np.random.uniform(0, 2 * np.pi)))
        babble += noise * envelope.astype(np.float32)
    babble = babble / (np.max(np.abs(babble)) + 1e-8)
    return babble


def mix_at_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Mix clean signal with noise at a specified SNR in dB."""
    # Match lengths
    if len(noise) < len(clean):
        noise = np.tile(noise, int(np.ceil(len(clean) / len(noise))))
    noise = noise[:len(clean)]

    clean_power = np.mean(clean ** 2)
    noise_power = np.mean(noise ** 2)

    if noise_power == 0:
        return clean.copy()

    # Scale noise to achieve target SNR
    target_noise_power = clean_power / (10 ** (snr_db / 10))
    noise_scaled = noise * np.sqrt(target_noise_power / noise_power)

    mixed = clean + noise_scaled
    # Prevent clipping
    peak = np.max(np.abs(mixed))
    if peak > 0.99:
        mixed = mixed * 0.99 / peak

    return mixed.astype(np.float32)


def compute_snr(clean: np.ndarray, noisy: np.ndarray) -> float:
    """Estimate SNR between clean and noisy signal."""
    min_len = min(len(clean), len(noisy))
    c = clean[:min_len]
    n = noisy[:min_len]
    noise = n - c
    signal_power = np.mean(c ** 2)
    noise_power = np.mean(noise ** 2)
    if noise_power == 0:
        return float('inf')
    return float(10 * np.log10(signal_power / noise_power))


def compute_spectral_noise_floor(audio: np.ndarray, sr: int) -> dict:
    """Compute spectral characteristics to estimate noise floor."""
    from scipy.signal import welch
    freqs, psd = welch(audio, fs=sr, nperseg=2048)

    # Overall energy
    total_energy = np.sum(psd)

    # Speech band energy (300-3400 Hz)
    speech_mask = (freqs >= 300) & (freqs <= 3400)
    speech_energy = np.sum(psd[speech_mask])

    # Noise floor: energy outside speech band
    noise_mask = ~speech_mask
    noise_energy = np.sum(psd[noise_mask])

    # Speech-to-noise ratio in spectral domain
    spectral_snr = 10 * np.log10(speech_energy / (noise_energy + 1e-10))

    return {
        "total_energy": float(total_energy),
        "speech_band_energy": float(speech_energy),
        "noise_floor_energy": float(noise_energy),
        "spectral_snr_db": float(spectral_snr),
    }


def run_experiment():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("EXPERIMENT 01c: Voice Isolation / Noise Reduction")
    print("=" * 70)

    # Load clean audio
    print("\n--- Loading clean audio ---")
    clean_audio, sr = download_librispeech_sample()

    # Initialize Mimi
    print("\n--- Loading Mimi ---")
    codec = MimiCodec(device="cpu")

    # ====================================================================
    # TEST 1: Synthetic noise at various SNR levels
    # ====================================================================
    print("\n" + "=" * 70)
    print("TEST 1: Mimi Denoising — Synthetic Noise at Various SNR Levels")
    print("=" * 70)

    noise_types = {
        "white": generate_white_noise(len(clean_audio)),
        "pink": generate_pink_noise(len(clean_audio)),
        "babble": generate_babble_noise(len(clean_audio)),
    }

    snr_levels = [20, 10, 5, 0, -5]  # dB

    print(f"\n{'Noise':<10} {'SNR':>5} │ {'Noisy→Clean':>14} {'Mimi→Clean':>14} │ {'Denoising?':>12}")
    print(f"{'Type':<10} {'(dB)':>5} │ {'PESQ / STOI':>14} {'PESQ / STOI':>14} │ {'ΔSTOI':>12}")
    print("─" * 72)

    for noise_name, noise in noise_types.items():
        for snr_db in snr_levels:
            # Mix clean + noise
            noisy = mix_at_snr(clean_audio, noise, snr_db)

            # Run through Mimi
            tokens = codec.encode(noisy, sr=sr)
            mimi_out = codec.decode(tokens)

            # Measure quality vs CLEAN original
            min_len = min(len(clean_audio), len(noisy), len(mimi_out))
            ref = clean_audio[:min_len]
            noisy_ref = noisy[:min_len]
            mimi_ref = mimi_out[:min_len]

            try:
                noisy_pesq = compute_pesq(ref, noisy_ref, sr)
            except Exception:
                noisy_pesq = float("nan")
            try:
                mimi_pesq = compute_pesq(ref, mimi_ref, sr)
            except Exception:
                mimi_pesq = float("nan")

            noisy_stoi = compute_stoi(ref, noisy_ref, sr)
            mimi_stoi = compute_stoi(ref, mimi_ref, sr)

            delta_stoi = mimi_stoi - noisy_stoi
            indicator = "✓ YES" if delta_stoi > 0.01 else ("~ same" if abs(delta_stoi) <= 0.01 else "✗ worse")

            np_str = f"{noisy_pesq:.2f}" if not np.isnan(noisy_pesq) else "N/A"
            mp_str = f"{mimi_pesq:.2f}" if not np.isnan(mimi_pesq) else "N/A"

            print(f"{noise_name:<10} {snr_db:>5} │ {np_str:>6} / {noisy_stoi:.3f} {mp_str:>6} / {mimi_stoi:.3f} │ {delta_stoi:>+.3f} {indicator}")

            # Save audio for key conditions
            if snr_db in (5, 0) and noise_name == "babble":
                save_audio(noisy, RESULTS_DIR / f"01c_{noise_name}_snr{snr_db}_noisy_{timestamp}.wav", sr)
                save_audio(mimi_out, RESULTS_DIR / f"01c_{noise_name}_snr{snr_db}_mimi_{timestamp}.wav", sr)

    # Save clean reference
    save_audio(clean_audio, RESULTS_DIR / f"01c_clean_reference_{timestamp}.wav", sr)

    # ====================================================================
    # TEST 2: Airplane recording — spectral analysis
    # ====================================================================
    airplane_path = AUDIO_DIR / "airplane_handheld.wav"
    if airplane_path.exists():
        print("\n" + "=" * 70)
        print("TEST 2: Airplane Recording — Spectral Noise Analysis")
        print("=" * 70)

        airplane_audio, _ = load_audio(airplane_path, target_sr=sr)

        # Run through Mimi
        tokens = codec.encode(airplane_audio, sr=sr)
        mimi_airplane = codec.decode(tokens)

        # Spectral analysis before/after
        print("\n--- Spectral Analysis ---")
        orig_spec = compute_spectral_noise_floor(airplane_audio, sr)
        mimi_spec = compute_spectral_noise_floor(mimi_airplane, sr)

        print(f"\n{'Metric':<30} {'Original':>12} {'After Mimi':>12} {'Change':>12}")
        print("-" * 70)
        print(f"{'Total energy':<30} {orig_spec['total_energy']:>12.4f} {mimi_spec['total_energy']:>12.4f} {mimi_spec['total_energy']/orig_spec['total_energy']*100-100:>+11.1f}%")
        print(f"{'Speech band (300-3400Hz)':<30} {orig_spec['speech_band_energy']:>12.4f} {mimi_spec['speech_band_energy']:>12.4f} {mimi_spec['speech_band_energy']/orig_spec['speech_band_energy']*100-100:>+11.1f}%")
        print(f"{'Noise floor (outside speech)':<30} {orig_spec['noise_floor_energy']:>12.4f} {mimi_spec['noise_floor_energy']:>12.4f} {mimi_spec['noise_floor_energy']/orig_spec['noise_floor_energy']*100-100:>+11.1f}%")
        print(f"{'Spectral SNR':<30} {orig_spec['spectral_snr_db']:>11.1f}dB {mimi_spec['spectral_snr_db']:>11.1f}dB {mimi_spec['spectral_snr_db']-orig_spec['spectral_snr_db']:>+11.1f}dB")

        # Plot spectrograms side by side
        plot_waveform_and_spectrogram(
            {
                "Airplane Original": (airplane_audio, sr),
                "After Mimi (32 cb)": (mimi_airplane, sr),
            },
            title="Voice Isolation: Airplane Audio Before/After Mimi",
            save_path=RESULTS_DIR / f"01c_airplane_spectral_{timestamp}.png",
        )

        # Also generate a spectrogram at different codebook levels
        print("\n--- Codebook sweep on airplane audio ---")
        for n_cb in [4, 8, 16, 32]:
            modified = tokens.clone()
            modified[:, n_cb:, :] = 0
            recon = codec.decode(modified)
            spec = compute_spectral_noise_floor(recon, sr)
            save_audio(recon, RESULTS_DIR / f"01c_airplane_cb{n_cb:02d}_{timestamp}.wav", sr)
            print(f"  {n_cb} codebooks: spectral SNR = {spec['spectral_snr_db']:.1f} dB "
                  f"(noise floor: {spec['noise_floor_energy']:.4f})")

    # ====================================================================
    # Summary
    # ====================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 01c SUMMARY")
    print("=" * 70)
    print("""
Listen to these key files:
  - 01c_clean_reference_*.wav         — clean speech
  - 01c_babble_snr5_noisy_*.wav       — clean + babble noise at 5dB SNR
  - 01c_babble_snr5_mimi_*.wav        — after Mimi reconstruction
  - 01c_babble_snr0_noisy_*.wav       — clean + babble noise at 0dB SNR
  - 01c_babble_snr0_mimi_*.wav        — after Mimi reconstruction
  - 01c_airplane_cb*_*.wav            — airplane at different codebook levels

If Mimi output sounds cleaner than the noisy input, the token representation
is acting as a voice-selective bottleneck — free noise reduction!
""")
    print("=" * 70)


if __name__ == "__main__":
    run_experiment()
