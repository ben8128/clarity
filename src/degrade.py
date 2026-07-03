"""Phone-grade degradation augmentation.

Simulates what a real incoming voice message sounds like (room noise,
bandlimited mic, a bit of room reverb) so reconstruction models can be
trained/evaluated on (degraded input → studio target) pairs.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.signal import butter, fftconvolve, sosfilt


@dataclass
class DegradeConfig:
    """Parameters for phone-grade degradation."""

    snr_db: float = 18.0          # additive room-noise level
    lowpass_hz: float = 7000.0    # phone mic bandwidth ceiling
    highpass_hz: float = 120.0    # phone mic low cut
    reverb_rt60_s: float = 0.25   # small-room tail; 0 disables
    reverb_wet: float = 0.12      # reverb mix
    seed: int = 0


def _pink_noise(length: int, rng: np.random.Generator) -> np.ndarray:
    """Pink (1/f) room-tone noise via filtered white noise."""
    from scipy.signal import lfilter

    white = rng.standard_normal(length).astype(np.float32)
    b = np.array([0.049922035, -0.095993537, 0.050612699, -0.004709510])
    a = np.array([1.0, -2.494956002, 2.017265875, -0.522189400])
    pink = lfilter(b, a, white).astype(np.float32)
    return pink / (np.max(np.abs(pink)) + 1e-8)


def _synthetic_rir(sr: int, rt60_s: float, rng: np.random.Generator) -> np.ndarray:
    """Exponentially-decaying noise burst as a small-room impulse response."""
    length = int(sr * rt60_s)
    t = np.arange(length) / sr
    decay = np.exp(-6.9 * t / rt60_s)  # -60 dB at rt60
    rir = rng.standard_normal(length).astype(np.float32) * decay.astype(np.float32)
    rir[0] = 1.0  # direct path
    return rir / (np.max(np.abs(rir)) + 1e-8)


def _mix_at_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    if len(noise) < len(clean):
        noise = np.tile(noise, int(np.ceil(len(clean) / len(noise))))
    noise = noise[: len(clean)]
    clean_power = np.mean(clean**2)
    noise_power = np.mean(noise**2) + 1e-12
    target_noise_power = clean_power / (10 ** (snr_db / 10))
    return clean + noise * np.sqrt(target_noise_power / noise_power)


def degrade_phone(
    audio: np.ndarray, sr: int = 24000, config: Optional[DegradeConfig] = None
) -> np.ndarray:
    """Apply phone-grade degradation: reverb → bandlimit → room noise.

    Args:
        audio: Clean mono float32 audio.
        sr: Sample rate.
        config: Degradation parameters (defaults to a typical handheld phone).

    Returns:
        Degraded audio, same length, peak-normalized below clipping.
    """
    cfg = config or DegradeConfig()
    rng = np.random.default_rng(cfg.seed)
    out = audio.astype(np.float32).copy()

    if cfg.reverb_rt60_s > 0 and cfg.reverb_wet > 0:
        rir = _synthetic_rir(sr, cfg.reverb_rt60_s, rng)
        wet = fftconvolve(out, rir)[: len(out)].astype(np.float32)
        wet = wet / (np.max(np.abs(wet)) + 1e-8) * (np.max(np.abs(out)) + 1e-8)
        out = (1 - cfg.reverb_wet) * out + cfg.reverb_wet * wet

    sos = butter(4, [cfg.highpass_hz, cfg.lowpass_hz], btype="band", fs=sr, output="sos")
    out = sosfilt(sos, out).astype(np.float32)

    out = _mix_at_snr(out, _pink_noise(len(out), rng), cfg.snr_db).astype(np.float32)

    peak = np.max(np.abs(out))
    if peak > 0.99:
        out = out * 0.99 / peak
    return out
