"""Experiment 02: Semantic-Only Reconstruction Quality Sweep.

Maps the quality/bandwidth tradeoff by varying the number of codebooks transmitted.
"""

import sys
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
    print("EXPERIMENT 02: Semantic-Only Reconstruction Quality Sweep")
    print("=" * 70)

    audio, sr = download_librispeech_sample()
    codec = MimiCodec(device="cpu")
    tokens = codec.encode(audio, sr=sr)
    info = codec.report_codebook_info(tokens)

    num_codebooks = info["num_codebooks"]
    frame_rate = info["frame_rate_hz"]
    bits_per_token = info["bits_per_token"]

    results = []

    # Sweep: use codebooks 0..k for k in [1, 2, 3, 4, 8, 16, num_codebooks]
    sweep_points = sorted(set([1, 2, 3, 4, 8, 16, num_codebooks]))
    sweep_points = [k for k in sweep_points if k <= num_codebooks]

    for k in sweep_points:
        print(f"\nReconstructing with codebooks 0..{k-1} ({k} total)...")
        modified = tokens.clone()
        modified[:, k:, :] = 0
        recon = codec.decode(modified)

        bitrate = frame_rate * k * bits_per_token

        try:
            pesq_score = compute_pesq(audio, recon, sr)
        except Exception as e:
            print(f"  PESQ failed: {e}")
            pesq_score = float("nan")

        try:
            stoi_score = compute_stoi(audio, recon, sr)
        except Exception as e:
            print(f"  STOI failed: {e}")
            stoi_score = float("nan")

        results.append({
            "codebooks": k,
            "bitrate_bps": bitrate,
            "pesq": pesq_score,
            "stoi": stoi_score,
        })

        save_audio(recon, RESULTS_DIR / f"02_cb{k}_{timestamp}.wav", sr)

    # Print results table
    print("\n" + "=" * 65)
    print(f"{'Codebooks':>10} {'Bitrate':>12} {'PESQ':>8} {'STOI':>8}")
    print("-" * 65)
    for r in results:
        bps = r["bitrate_bps"]
        rate_str = f"{bps:.1f} bps" if bps < 1000 else f"{bps/1000:.1f} kbps"
        pesq_str = f"{r['pesq']:.3f}" if not np.isnan(r["pesq"]) else "N/A"
        stoi_str = f"{r['stoi']:.3f}" if not np.isnan(r["stoi"]) else "N/A"
        print(f"{r['codebooks']:>10} {rate_str:>12} {pesq_str:>8} {stoi_str:>8}")
    print("=" * 65)

    # Plot quality vs bitrate
    try:
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        bitrates = [r["bitrate_bps"] for r in results]
        pesqs = [r["pesq"] for r in results]
        stois = [r["stoi"] for r in results]

        ax1.plot(bitrates, pesqs, "bo-")
        ax1.set_xlabel("Bitrate (bps)")
        ax1.set_ylabel("PESQ")
        ax1.set_title("PESQ vs Bitrate")
        ax1.set_xscale("log")
        ax1.grid(True)

        ax2.plot(bitrates, stois, "ro-")
        ax2.set_xlabel("Bitrate (bps)")
        ax2.set_ylabel("STOI")
        ax2.set_title("STOI vs Bitrate")
        ax2.set_xscale("log")
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig(RESULTS_DIR / f"02_quality_vs_bitrate_{timestamp}.png", dpi=150)
        plt.close()
        print(f"Saved quality vs bitrate plot.")
    except Exception as e:
        print(f"Plotting failed: {e}")

    return results


if __name__ == "__main__":
    run_experiment()
