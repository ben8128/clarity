"""Experiment 05: Bandwidth & Latency Measurement.

Quantifies practical performance: bitrates, encoding/decoding latency,
and quality degradation under simulated packet loss.
"""

import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.codec import MimiCodec
from src.quality import compute_pesq, compute_stoi
from src.utils import download_librispeech_sample, save_audio

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def run_experiment():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print("=" * 70)
    print("EXPERIMENT 05: Bandwidth & Latency Measurement")
    print("=" * 70)

    audio, sr = download_librispeech_sample()
    codec = MimiCodec(device="cpu")
    duration_s = len(audio) / sr

    # --- Encoding latency ---
    print("\n--- Encoding Latency ---")
    encode_times = []
    for i in range(5):
        start = time.perf_counter()
        tokens = codec.encode(audio, sr=sr)
        elapsed = time.perf_counter() - start
        encode_times.append(elapsed)
        print(f"  Run {i+1}: {elapsed:.3f}s")

    info = codec.report_codebook_info(tokens)
    avg_encode = np.mean(encode_times[1:])  # skip first (warmup)
    print(f"  Average (excl. warmup): {avg_encode:.3f}s for {duration_s:.2f}s audio")
    print(f"  Realtime factor: {duration_s / avg_encode:.1f}x")

    # --- Decoding latency ---
    print("\n--- Decoding Latency ---")
    decode_times = []
    for i in range(5):
        start = time.perf_counter()
        recon = codec.decode(tokens)
        elapsed = time.perf_counter() - start
        decode_times.append(elapsed)
        print(f"  Run {i+1}: {elapsed:.3f}s")

    avg_decode = np.mean(decode_times[1:])
    print(f"  Average (excl. warmup): {avg_decode:.3f}s for {duration_s:.2f}s audio")
    print(f"  Realtime factor: {duration_s / avg_decode:.1f}x")

    # --- Bitrate comparison ---
    print("\n--- Bitrate Comparison ---")
    num_codebooks = info["num_codebooks"]
    frame_rate = info["frame_rate_hz"]
    bits_per_token = info["bits_per_token"]

    bitrates = {
        "Raw PCM 24kHz/16bit": 24000 * 16,
        "Opus (voice, 24kbps)": 24000,
        "Opus (low, 6kbps)": 6000,
        f"Mimi full ({num_codebooks} cb)": frame_rate * num_codebooks * bits_per_token,
        "Mimi 4 codebooks": frame_rate * 4 * bits_per_token,
        "Mimi 2 codebooks": frame_rate * 2 * bits_per_token,
        "Mimi semantic-only": frame_rate * 1 * bits_per_token,
    }

    print(f"\n{'Codec':<30} {'Bitrate':>12} {'Compression':>12}")
    print("-" * 58)
    raw_bps = bitrates["Raw PCM 24kHz/16bit"]
    for label, bps in bitrates.items():
        if bps >= 1000:
            rate_str = f"{bps/1000:.1f} kbps"
        else:
            rate_str = f"{bps:.1f} bps"
        ratio = f"{raw_bps / bps:.0f}x"
        print(f"{label:<30} {rate_str:>12} {ratio:>12}")

    # --- Packet loss simulation ---
    print("\n--- Packet Loss Simulation ---")
    loss_rates = [0.0, 0.05, 0.10, 0.20, 0.30]
    loss_results = []

    for loss_rate in loss_rates:
        print(f"\n  Packet loss: {loss_rate*100:.0f}%")
        semantic_tokens = tokens.clone()
        semantic_tokens[:, 1:, :] = 0  # semantic-only baseline

        num_frames = semantic_tokens.shape[2]

        if loss_rate > 0:
            # Randomly drop frames by zeroing them
            np.random.seed(42)
            drop_mask = np.random.random(num_frames) < loss_rate
            semantic_tokens[:, :, drop_mask] = 0
            dropped = int(drop_mask.sum())
            print(f"    Dropped {dropped}/{num_frames} frames")

        recon = codec.decode(semantic_tokens)
        save_audio(
            recon,
            RESULTS_DIR / f"05_loss{int(loss_rate*100)}pct_{timestamp}.wav",
            sr,
        )

        try:
            pesq_score = compute_pesq(audio, recon, sr)
        except Exception:
            pesq_score = float("nan")

        try:
            stoi_score = compute_stoi(audio, recon, sr)
        except Exception:
            stoi_score = float("nan")

        loss_results.append({
            "loss_rate": loss_rate,
            "pesq": pesq_score,
            "stoi": stoi_score,
        })
        print(f"    PESQ: {pesq_score:.3f}, STOI: {stoi_score:.3f}")

    # Summary
    print("\n" + "=" * 70)
    print("EXPERIMENT 05 SUMMARY")
    print("=" * 70)
    print(f"\nLatency budget (for {duration_s:.2f}s audio):")
    print(f"  Encoding:  {avg_encode*1000:.1f}ms ({duration_s/avg_encode:.1f}x realtime)")
    print(f"  Decoding:  {avg_decode*1000:.1f}ms ({duration_s/avg_decode:.1f}x realtime)")
    print(f"  Total:     {(avg_encode+avg_decode)*1000:.1f}ms")

    print(f"\nPacket loss resilience (semantic-only):")
    print(f"  {'Loss':>6} {'PESQ':>8} {'STOI':>8}")
    print(f"  {'-'*26}")
    for r in loss_results:
        pesq_str = f"{r['pesq']:.3f}" if not np.isnan(r["pesq"]) else "N/A"
        stoi_str = f"{r['stoi']:.3f}" if not np.isnan(r["stoi"]) else "N/A"
        print(f"  {r['loss_rate']*100:>5.0f}% {pesq_str:>8} {stoi_str:>8}")

    return {
        "avg_encode_s": avg_encode,
        "avg_decode_s": avg_decode,
        "bitrates": bitrates,
        "packet_loss": loss_results,
    }


if __name__ == "__main__":
    run_experiment()
