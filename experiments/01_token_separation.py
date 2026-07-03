"""Experiment 01: Codebook Sweep.

Finds the quality/bandwidth knee by sweeping codebook count from 1 (semantic only)
through all available codebooks.  For each count, measures PESQ, STOI, and speaker
similarity against the original audio.  Averages metrics across multiple samples,
produces a dual-axis plot, and detects the sweet-spot codebook count.
"""

import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.codec import MimiCodec
from src.quality import (
    compute_pesq,
    compute_stoi,
    compute_speaker_similarity,
    optimal_codebook_count,
)
from src.results_io import save_metrics
from src.utils import (
    download_librispeech_sample,
    load_audio,
    save_audio,
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
AUDIO_DIR = Path(__file__).resolve().parent.parent / "audio"


def find_test_audio_samples(max_samples: int = 3) -> list[tuple[np.ndarray, int, str]]:
    """Collect test audio samples. Uses local audio/ files first, falls back to LibriSpeech.

    Returns:
        List of (audio_array, sample_rate, label) tuples.
    """
    samples: list[tuple[np.ndarray, int, str]] = []

    # Look for local audio files
    if AUDIO_DIR.exists():
        for ext in ("*.wav", "*.mp3", "*.flac"):
            for f in sorted(AUDIO_DIR.glob(ext)):
                if len(samples) >= max_samples:
                    break
                audio, sr = load_audio(f, target_sr=24000)
                samples.append((audio, sr, f.stem))
                print(f"Loaded local audio: {f.name}")

    # Fill remaining slots from LibriSpeech
    if len(samples) < max_samples:
        from datasets import load_dataset

        print("Loading LibriSpeech samples...")
        ds = load_dataset(
            "hf-internal-testing/librispeech_asr_dummy",
            "clean",
            split="validation",
            trust_remote_code=True,
        )
        from src.utils import resample

        for idx in range(min(max_samples - len(samples), len(ds))):
            sample = ds[idx]
            audio = np.array(sample["audio"]["array"], dtype=np.float32)
            orig_sr = sample["audio"]["sampling_rate"]
            audio_24k = resample(audio, orig_sr, 24000)
            samples.append((audio_24k, 24000, f"librispeech_{idx}"))
            print(f"Loaded LibriSpeech sample {idx}: {len(audio_24k)/24000:.2f}s")

    return samples


def choose_sweep_points(num_codebooks: int) -> list[int]:
    """Choose which codebook counts to sweep.

    Sweeps 1–8 individually; if num_codebooks > 8, adds sampled points up to the max.
    """
    points = list(range(1, min(num_codebooks, 8) + 1))
    if num_codebooks > 8:
        extra = [n for n in [12, 16, 24, 32] if 8 < n < num_codebooks]
        extra.append(num_codebooks)
        points.extend(extra)
    return sorted(set(points))


def run_experiment():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print("=" * 70)
    print("EXPERIMENT 01: Codebook Sweep")
    print("=" * 70)

    # Step 1: Load audio samples
    print("\n--- Step 1: Loading test audio ---")
    samples = find_test_audio_samples()
    if not samples:
        print("ERROR: No audio samples found.")
        return None
    print(f"Using {len(samples)} sample(s): {[s[2] for s in samples]}")

    # Step 2: Initialize codec and do a probe encode to learn codebook count
    print("\n--- Step 2: Initializing Mimi codec ---")
    codec = MimiCodec(device="cpu")
    probe_tokens = codec.encode(samples[0][0], sr=samples[0][1])
    info = codec.report_codebook_info(probe_tokens)
    num_codebooks = info["num_codebooks"]
    sweep_points = choose_sweep_points(num_codebooks)
    print(f"Sweep points: {sweep_points}")

    # Step 3: Run sweep across all samples
    print("\n--- Step 3: Running codebook sweep ---")
    # Accumulate per-n metrics across samples
    accumulated: dict[int, list[dict]] = {n: [] for n in sweep_points}

    for audio, sr, label in samples:
        print(f"\n  Sample: {label}")
        tokens = codec.encode(audio, sr=sr)

        for n in sweep_points:
            print(f"    {n} codebook(s)...", end=" ", flush=True)
            recon = codec.reconstruct_with_n_codebooks(tokens, n)

            metrics: dict[str, float] = {"codebooks": n}
            try:
                metrics["pesq"] = compute_pesq(audio, recon, sr)
            except Exception as e:
                print(f"PESQ failed: {e}", end=" ")
                metrics["pesq"] = float("nan")

            try:
                metrics["stoi"] = compute_stoi(audio, recon, sr)
            except Exception as e:
                print(f"STOI failed: {e}", end=" ")
                metrics["stoi"] = float("nan")

            try:
                metrics["speaker_similarity"] = compute_speaker_similarity(audio, recon, sr)
            except Exception:
                metrics["speaker_similarity"] = float("nan")

            accumulated[n].append(metrics)
            print(
                f"PESQ={metrics['pesq']:.3f}  STOI={metrics['stoi']:.3f}  "
                f"SpkSim={metrics['speaker_similarity']:.3f}"
                if not np.isnan(metrics["pesq"])
                else "done"
            )

        # Save per-sample reconstructions for the first sample only (for listening)
        if label == samples[0][2]:
            save_audio(audio, RESULTS_DIR / f"01_original_{timestamp}.wav", sr)
            for n in sweep_points:
                recon = codec.reconstruct_with_n_codebooks(tokens, n)
                save_audio(
                    recon,
                    RESULTS_DIR / f"reconstruction_{n}_codebooks.wav",
                    sr,
                )

    # Step 4: Average metrics across samples
    print("\n--- Step 4: Averaging metrics ---")
    sweep_results: list[dict] = []
    for n in sweep_points:
        entries = accumulated[n]
        avg: dict[str, float] = {"codebooks": n}
        for key in ("pesq", "stoi", "speaker_similarity"):
            vals = [e[key] for e in entries if not np.isnan(e.get(key, float("nan")))]
            avg[key] = float(np.mean(vals)) if vals else float("nan")
        avg["bitrate_bps"] = codec.get_bitrate(n)
        avg["vs_opus_savings"] = codec.get_bandwidth_savings_vs_opus(n)
        sweep_results.append(avg)

    # Step 5: Print summary table
    print("\n--- Step 5: Summary Table ---")
    header = (
        f"{'Codebooks':>10} {'Bitrate':>12} {'vs Opus':>10} "
        f"{'PESQ':>8} {'STOI':>8} {'SpkSim':>8}"
    )
    print(header)
    print("-" * len(header))
    for r in sweep_results:
        bps = r["bitrate_bps"]
        rate_str = f"{bps:.1f} bps" if bps < 1000 else f"{bps/1000:.1f} kbps"
        savings_str = f"{r['vs_opus_savings']*100:.1f}%"
        pesq_str = f"{r['pesq']:.3f}" if not np.isnan(r["pesq"]) else "N/A"
        stoi_str = f"{r['stoi']:.3f}" if not np.isnan(r["stoi"]) else "N/A"
        sim_str = (
            f"{r['speaker_similarity']:.3f}"
            if not np.isnan(r.get("speaker_similarity", float("nan")))
            else "N/A"
        )
        print(
            f"{r['codebooks']:>10} {rate_str:>12} {savings_str:>10} "
            f"{pesq_str:>8} {stoi_str:>8} {sim_str:>8}"
        )

    # Step 6: Sweet spot detection
    print("\n--- Step 6: Sweet Spot Detection ---")
    sweet_spot = optimal_codebook_count(sweep_results, metric="pesq")
    sweet_bitrate = codec.get_bitrate(sweet_spot)
    sweet_savings = codec.get_bandwidth_savings_vs_opus(sweet_spot)
    print(f"Sweet spot: {sweet_spot} codebook(s)")
    print(f"  Bitrate:  {sweet_bitrate:.1f} bps ({sweet_bitrate/1000:.2f} kbps)")
    print(f"  vs Opus:  {sweet_savings*100:.1f}% smaller")

    save_metrics(
        "01_codebook_sweep",
        {
            "samples": [s[2] for s in samples],
            "num_codebooks": num_codebooks,
            "sweep_points": sweep_points,
            "per_sample": {
                str(n): entries for n, entries in accumulated.items()
            },
            "averaged": sweep_results,
            "sweet_spot": {
                "codebooks": sweet_spot,
                "bitrate_bps": sweet_bitrate,
                "vs_opus_savings": sweet_savings,
            },
        },
    )

    # Step 7: Dual-axis plot
    print("\n--- Step 7: Generating plot ---")
    try:
        import matplotlib.pyplot as plt

        fig, ax1 = plt.subplots(figsize=(10, 6))

        cbs = [r["codebooks"] for r in sweep_results]
        pesqs = [r["pesq"] for r in sweep_results]
        stois = [r["stoi"] for r in sweep_results]
        bitrates = [r["bitrate_bps"] for r in sweep_results]

        # Left axis: quality metrics
        color_pesq = "#2196F3"
        color_stoi = "#4CAF50"
        ax1.plot(cbs, pesqs, "o-", color=color_pesq, linewidth=2, label="PESQ")
        ax1.plot(cbs, stois, "s-", color=color_stoi, linewidth=2, label="STOI")
        ax1.set_xlabel("Number of codebooks", fontsize=12)
        ax1.set_ylabel("Quality metric", fontsize=12)
        ax1.set_xticks(cbs)
        ax1.legend(loc="upper left")
        ax1.grid(True, alpha=0.3)

        # Right axis: bitrate
        ax2 = ax1.twinx()
        color_bps = "#FF9800"
        ax2.step(cbs, bitrates, where="mid", color=color_bps, linewidth=2,
                 linestyle="--", alpha=0.7, label="Bitrate")
        ax2.set_ylabel("Bitrate (bps)", fontsize=12, color=color_bps)
        ax2.tick_params(axis="y", labelcolor=color_bps)

        # Opus reference range (6,000–32,000 bps)
        ax2.axhspan(6000, 32000, alpha=0.08, color="red", label="Opus range")
        ax2.text(
            cbs[-1], 19000, "Opus range\n(6–32 kbps)",
            ha="right", va="center", fontsize=9, color="red", alpha=0.6,
        )
        ax2.legend(loc="lower right")

        # Sweet spot marker
        ax1.axvline(x=sweet_spot, color="#E91E63", linestyle=":", linewidth=2, alpha=0.7)
        ax1.annotate(
            f"Sweet spot\n({sweet_spot} cb, {sweet_bitrate:.0f} bps)",
            xy=(sweet_spot, max(pesqs) * 0.95),
            fontsize=9,
            color="#E91E63",
            ha="center",
        )

        plt.title("Codebook Sweep: Quality vs Bandwidth", fontsize=14)
        fig.tight_layout()
        plot_path = RESULTS_DIR / f"01_codebook_sweep_{timestamp}.png"
        plt.savefig(str(plot_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved plot: {plot_path}")
    except Exception as e:
        print(f"Plotting failed: {e}")

    # Step 8: Summary
    print("\n" + "=" * 70)
    print("EXPERIMENT 01 SUMMARY — Codebook Sweep")
    print("=" * 70)
    print(f"\nMimi codebooks: {num_codebooks}")
    print(f"Sweep points tested: {sweep_points}")
    print(f"Samples averaged: {len(samples)}")
    print(f"\nSweet spot: {sweet_spot} codebook(s) at {sweet_bitrate:.1f} bps")
    print(f"  Still {sweet_savings*100:.1f}% smaller than Opus (24 kbps)")
    print(f"\nEven ALL {num_codebooks} codebooks at "
          f"{codec.get_bitrate(num_codebooks):.0f} bps is "
          f"{codec.get_bandwidth_savings_vs_opus(num_codebooks)*100:.1f}% smaller than Opus.")
    print(f"\nOutput files saved to: {RESULTS_DIR}/")
    print("LISTEN to reconstruction_*_codebooks.wav to verify quality gradient.")
    print("=" * 70)

    return sweep_results


if __name__ == "__main__":
    run_experiment()
