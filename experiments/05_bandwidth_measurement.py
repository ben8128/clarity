"""Experiment 05: Bandwidth & Packet Loss Resilience.

Bitrate table plus the core transport question: how does quality degrade when
token frames are lost, and which cheap strategy recovers it?

Loss simulation at the transmitted codebook count (default 8 = tier-B
candidate), loss rates {0, 5, 10, 20, 30}%, uniform and bursty patterns,
three arms:
  A "zero-fill":   lost frames decoded with token id 0 (naive baseline —
                   token 0 is a valid entry, so this decodes wrong audio)
  B "repeat-last": lost frames reuse the previous received frame's tokens
  C "redundancy":  DRED-style — each packet also carries the previous D
                   frames (D=1,2,4); a frame is lost only if all packets
                   carrying it drop. Residual losses fall back to repeat-last.
                   Effective bitrate multiplies by (D+1) but stays far below
                   Opus: 8 cb x 5 = 5.5 kbps vs 24 kbps.
"""

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.codec import MimiCodec
from src.quality import compute_pesq, compute_stoi
from src.results_io import save_metrics
from src.utils import download_librispeech_sample, save_audio

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

TRANSMIT_CODEBOOKS = 8
LOSS_RATES = [0.0, 0.05, 0.10, 0.20, 0.30]
REDUNDANCY_DEPTHS = [1, 2, 4]
MEAN_BURST_LEN = 4
SEED = 42


def uniform_loss_mask(num_frames: int, loss_rate: float, rng: np.random.Generator) -> np.ndarray:
    """Independent per-packet loss."""
    return rng.random(num_frames) < loss_rate


def bursty_loss_mask(
    num_frames: int, loss_rate: float, rng: np.random.Generator,
    mean_burst_len: int = MEAN_BURST_LEN,
) -> np.ndarray:
    """Two-state Gilbert model: losses arrive in bursts of ~mean_burst_len."""
    if loss_rate <= 0:
        return np.zeros(num_frames, dtype=bool)
    p_recover = 1.0 / mean_burst_len
    p_enter = loss_rate * p_recover / max(1.0 - loss_rate, 1e-9)
    mask = np.zeros(num_frames, dtype=bool)
    lost = False
    for i in range(num_frames):
        if lost:
            lost = rng.random() >= p_recover
        else:
            lost = rng.random() < p_enter
        mask[i] = lost
    return mask


def apply_zero_fill(tokens: torch.Tensor, lost: np.ndarray) -> torch.Tensor:
    """Arm A: decode lost frames with token id 0 (naive baseline)."""
    out = tokens.clone()
    out[:, :, torch.from_numpy(lost)] = 0
    return out


def apply_repeat_last(tokens: torch.Tensor, lost: np.ndarray) -> torch.Tensor:
    """Arm B: lost frames reuse the previous received frame's tokens."""
    out = tokens.clone()
    last_good = None
    for i in range(out.shape[2]):
        if lost[i]:
            if last_good is not None:
                out[:, :, i] = out[:, :, last_good]
            else:
                out[:, :, i] = 0  # nothing received yet
        else:
            last_good = i
    return out


def apply_redundancy(
    tokens: torch.Tensor, packet_lost: np.ndarray, depth: int
) -> tuple[torch.Tensor, float]:
    """Arm C: frame i rides in packets i..i+depth; recovered if any arrives.

    Returns the repaired tokens (residual losses handled by repeat-last)
    and the residual frame-loss rate after redundancy.
    """
    num_frames = tokens.shape[2]
    frame_lost = np.ones(num_frames, dtype=bool)
    for i in range(num_frames):
        for j in range(i, min(i + depth + 1, num_frames)):
            if not packet_lost[j]:
                frame_lost[i] = False
                break
    residual_rate = float(frame_lost.mean())
    return apply_repeat_last(tokens, frame_lost), residual_rate


def measure(original: np.ndarray, recon: np.ndarray, sr: int) -> dict:
    min_len = min(len(original), len(recon))
    ref, deg = original[:min_len], recon[:min_len]
    try:
        pesq_score = compute_pesq(ref, deg, sr)
    except Exception:
        pesq_score = float("nan")
    try:
        stoi_score = compute_stoi(ref, deg, sr)
    except Exception:
        stoi_score = float("nan")
    return {"pesq": pesq_score, "stoi": stoi_score}


def run_experiment():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print("=" * 70)
    print("EXPERIMENT 05: Bandwidth & Packet Loss Resilience")
    print("=" * 70)

    audio, sr = download_librispeech_sample()
    codec = MimiCodec(device="cpu")
    tokens_full = codec.encode(audio, sr=sr)
    tokens = tokens_full[:, :TRANSMIT_CODEBOOKS, :]
    num_frames = tokens.shape[2]
    base_bitrate = codec.get_bitrate(TRANSMIT_CODEBOOKS)
    print(f"\nTransmitting {TRANSMIT_CODEBOOKS} codebooks ({base_bitrate:.0f} bps), "
          f"{num_frames} frames")

    # Reference: lossless decode at the transmitted codebook count
    lossless = codec.decode(tokens)
    baseline = measure(audio, lossless, sr)
    print(f"Lossless at {TRANSMIT_CODEBOOKS} cb: PESQ={baseline['pesq']:.3f} "
          f"STOI={baseline['stoi']:.3f}")

    # --- Bitrate table ---
    print("\n--- Bitrate Comparison ---")
    num_codebooks = tokens_full.shape[1]
    bitrates = {
        "Raw PCM 24kHz/16bit": 24000 * 16,
        "Opus (voice, 24kbps)": 24000,
        "Opus (low, 6kbps)": 6000,
        f"Mimi full ({num_codebooks} cb)": codec.get_bitrate(num_codebooks),
        f"Mimi {TRANSMIT_CODEBOOKS} cb": base_bitrate,
        f"Mimi {TRANSMIT_CODEBOOKS} cb + 4x redundancy": base_bitrate * 5,
        "Mimi semantic-only": codec.get_bitrate(1),
    }
    for label, bps in bitrates.items():
        rate_str = f"{bps/1000:.1f} kbps" if bps >= 1000 else f"{bps:.1f} bps"
        print(f"  {label:<35} {rate_str:>12}")

    # --- Loss matrix ---
    rng = np.random.default_rng(SEED)
    results = []
    patterns = {"uniform": uniform_loss_mask, "bursty": bursty_loss_mask}

    for pattern_name, make_mask in patterns.items():
        print(f"\n--- Loss pattern: {pattern_name} ---")
        header = f"  {'Loss':>5} {'Arm':<22} {'PESQ':>7} {'STOI':>7}"
        print(header)
        print("  " + "-" * (len(header) - 2))

        for loss_rate in LOSS_RATES:
            packet_lost = make_mask(num_frames, loss_rate, rng)
            actual_rate = float(packet_lost.mean())

            arms: list[tuple[str, torch.Tensor, dict]] = [
                ("A_zero_fill", apply_zero_fill(tokens, packet_lost), {}),
                ("B_repeat_last", apply_repeat_last(tokens, packet_lost), {}),
            ]
            for depth in REDUNDANCY_DEPTHS:
                repaired, residual = apply_redundancy(tokens, packet_lost, depth)
                arms.append(
                    (
                        f"C_redundancy_d{depth}",
                        repaired,
                        {
                            "redundancy_depth": depth,
                            "residual_frame_loss": residual,
                            "effective_bitrate_bps": base_bitrate * (depth + 1),
                        },
                    )
                )

            for arm_name, arm_tokens, extra in arms:
                if loss_rate == 0.0 and arm_name != "A_zero_fill":
                    continue  # all arms identical at zero loss
                recon = codec.decode(arm_tokens)
                m = measure(audio, recon, sr)
                row = {
                    "pattern": pattern_name,
                    "loss_rate": loss_rate,
                    "actual_packet_loss": actual_rate,
                    "arm": arm_name,
                    **m,
                    **extra,
                }
                results.append(row)
                print(f"  {loss_rate*100:>4.0f}% {arm_name:<22} {m['pesq']:>7.3f} {m['stoi']:>7.3f}")

                if pattern_name == "uniform" and loss_rate == 0.10:
                    save_audio(
                        recon,
                        RESULTS_DIR / f"05_{pattern_name}_loss10_{arm_name}_{timestamp}.wav",
                        sr,
                    )

    save_metrics(
        "05_packet_loss",
        {
            "transmit_codebooks": TRANSMIT_CODEBOOKS,
            "base_bitrate_bps": base_bitrate,
            "mean_burst_len": MEAN_BURST_LEN,
            "lossless_baseline": baseline,
            "bitrates": bitrates,
            "loss_matrix": results,
        },
    )

    # Summary: quality retained at 10% loss per arm
    print("\n" + "=" * 70)
    print("EXPERIMENT 05 SUMMARY (10% uniform loss, STOI vs lossless)")
    print("=" * 70)
    for r in results:
        if r["pattern"] == "uniform" and r["loss_rate"] == 0.10:
            delta = r["stoi"] - baseline["stoi"]
            print(f"  {r['arm']:<22} STOI {r['stoi']:.3f} (Δ {delta:+.3f})")
    print("\nGate: some arm should keep ΔSTOI > -0.05 at 10% loss.")
    return results


if __name__ == "__main__":
    run_experiment()
