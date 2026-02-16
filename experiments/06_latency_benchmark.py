"""Experiment 06: Latency Benchmark (Detailed).

Profiles per-component latency for real-time feasibility.
Tests with varying audio chunk sizes.
"""

import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.codec import MimiCodec
from src.utils import download_librispeech_sample

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def benchmark_chunk(codec: MimiCodec, audio: np.ndarray, sr: int, chunk_ms: int, n_runs: int = 10):
    """Benchmark encode+decode for a specific chunk size."""
    chunk_samples = int(sr * chunk_ms / 1000)

    # Use first chunk_samples of audio (pad if necessary)
    if len(audio) < chunk_samples:
        chunk = np.pad(audio, (0, chunk_samples - len(audio)))
    else:
        chunk = audio[:chunk_samples]

    encode_times = []
    decode_times = []
    token_extract_times = []

    for i in range(n_runs):
        # Encoding
        start = time.perf_counter()
        tokens = codec.encode(chunk, sr=sr)
        encode_elapsed = time.perf_counter() - start
        encode_times.append(encode_elapsed)

        # Token extraction (semantic)
        start = time.perf_counter()
        semantic = codec.extract_semantic(tokens)
        extract_elapsed = time.perf_counter() - start
        token_extract_times.append(extract_elapsed)

        # Decoding
        start = time.perf_counter()
        _ = codec.decode(tokens)
        decode_elapsed = time.perf_counter() - start
        decode_times.append(decode_elapsed)

    # Skip first run (warmup)
    return {
        "chunk_ms": chunk_ms,
        "chunk_samples": chunk_samples,
        "avg_encode_ms": np.mean(encode_times[1:]) * 1000,
        "avg_decode_ms": np.mean(decode_times[1:]) * 1000,
        "avg_extract_ms": np.mean(token_extract_times[1:]) * 1000,
        "avg_total_ms": (np.mean(encode_times[1:]) + np.mean(decode_times[1:])) * 1000,
        "realtime_ratio": chunk_ms / ((np.mean(encode_times[1:]) + np.mean(decode_times[1:])) * 1000),
    }


def run_experiment():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print("=" * 70)
    print("EXPERIMENT 06: Detailed Latency Benchmark")
    print("=" * 70)

    audio, sr = download_librispeech_sample()
    codec = MimiCodec(device="cpu")

    # Warmup
    print("\nWarming up model...")
    _ = codec.encode(audio[:sr], sr=sr)

    # Benchmark different chunk sizes
    chunk_sizes_ms = [80, 160, 320, 640, 1000, 2000]
    results = []

    for chunk_ms in chunk_sizes_ms:
        print(f"\nBenchmarking {chunk_ms}ms chunks...")
        result = benchmark_chunk(codec, audio, sr, chunk_ms)
        results.append(result)

        realtime_str = f"{result['realtime_ratio']:.1f}x" if result["realtime_ratio"] > 0 else "N/A"
        print(f"  Encode: {result['avg_encode_ms']:.1f}ms | "
              f"Decode: {result['avg_decode_ms']:.1f}ms | "
              f"Total: {result['avg_total_ms']:.1f}ms | "
              f"Realtime: {realtime_str}")

    # Summary table
    print("\n" + "=" * 80)
    print(f"{'Chunk':>8} {'Encode':>10} {'Extract':>10} {'Decode':>10} {'Total':>10} {'Realtime':>10}")
    print("-" * 80)
    for r in results:
        rt = f"{r['realtime_ratio']:.1f}x"
        print(f"{r['chunk_ms']:>6}ms {r['avg_encode_ms']:>8.1f}ms "
              f"{r['avg_extract_ms']:>8.1f}ms {r['avg_decode_ms']:>8.1f}ms "
              f"{r['avg_total_ms']:>8.1f}ms {rt:>10}")
    print("=" * 80)

    # Real-time feasibility assessment
    print("\nReal-time feasibility:")
    for r in results:
        feasible = r["avg_total_ms"] < r["chunk_ms"]
        status = "FEASIBLE" if feasible else "TOO SLOW"
        headroom = r["chunk_ms"] - r["avg_total_ms"]
        print(f"  {r['chunk_ms']}ms chunks: {status} "
              f"(headroom: {headroom:+.1f}ms)")

    # Plot
    try:
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        chunks = [r["chunk_ms"] for r in results]
        encodes = [r["avg_encode_ms"] for r in results]
        decodes = [r["avg_decode_ms"] for r in results]
        totals = [r["avg_total_ms"] for r in results]

        ax1.bar(range(len(chunks)), encodes, label="Encode", alpha=0.7)
        ax1.bar(range(len(chunks)), decodes, bottom=encodes, label="Decode", alpha=0.7)
        ax1.plot(range(len(chunks)), chunks, "r--", label="Chunk duration", linewidth=2)
        ax1.set_xticks(range(len(chunks)))
        ax1.set_xticklabels([f"{c}ms" for c in chunks])
        ax1.set_ylabel("Time (ms)")
        ax1.set_title("Processing Time vs Chunk Size")
        ax1.legend()

        ratios = [r["realtime_ratio"] for r in results]
        ax2.bar(range(len(chunks)), ratios, alpha=0.7, color="green")
        ax2.axhline(y=1.0, color="red", linestyle="--", label="Realtime threshold")
        ax2.set_xticks(range(len(chunks)))
        ax2.set_xticklabels([f"{c}ms" for c in chunks])
        ax2.set_ylabel("Realtime Ratio (higher=faster)")
        ax2.set_title("Realtime Feasibility")
        ax2.legend()

        plt.tight_layout()
        plt.savefig(RESULTS_DIR / f"06_latency_{timestamp}.png", dpi=150)
        plt.close()
        print(f"\nSaved latency plot to results/")
    except Exception as e:
        print(f"Plotting failed: {e}")

    return results


if __name__ == "__main__":
    run_experiment()
