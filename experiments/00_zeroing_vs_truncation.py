"""Experiment 00: Zeroing vs Truncation A/B for partial-codebook decode.

The original reconstruct_with_n_codebooks zeroed dropped codebooks
(modified[:, n:, :] = 0). Token id 0 is a valid codebook entry, so the RVQ
decoder summed in wrong embeddings for every dropped codebook instead of
omitting those quantizer layers. This experiment quantifies how much that
bug distorted quality measurements at n=4 and n=8.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.codec import MimiCodec
from src.quality import compute_pesq, compute_stoi
from src.results_io import save_metrics
from src.utils import download_librispeech_sample


def run_experiment():
    print("=" * 70)
    print("EXPERIMENT 00: Zeroing vs Truncation A/B")
    print("=" * 70)

    audio, sr = download_librispeech_sample()
    codec = MimiCodec(device="cpu")
    tokens = codec.encode(audio, sr=sr)

    results = []
    print(f"\n{'n':>4} {'Variant':<12} {'PESQ':>8} {'STOI':>8}")
    print("-" * 36)
    for n in (4, 8):
        # A: truncation (correct) — what reconstruct_with_n_codebooks now does
        truncated = codec.decode(tokens[:, :n, :])
        # B: zeroing (the bug) — decode all codebooks with dropped ones set to 0
        zeroed_tokens = tokens.clone()
        zeroed_tokens[:, n:, :] = 0
        zeroed = codec.decode(zeroed_tokens)

        for variant, recon in (("truncated", truncated), ("zeroed", zeroed)):
            min_len = min(len(audio), len(recon))
            ref, deg = audio[:min_len], recon[:min_len]
            try:
                pesq_score = compute_pesq(ref, deg, sr)
            except Exception:
                pesq_score = float("nan")
            stoi_score = compute_stoi(ref, deg, sr)
            results.append(
                {"codebooks": n, "variant": variant, "pesq": pesq_score, "stoi": stoi_score}
            )
            print(f"{n:>4} {variant:<12} {pesq_score:>8.3f} {stoi_score:>8.3f}")

    save_metrics("00_zeroing_vs_truncation", {"results": results})
    print("\nIf 'zeroed' scores differ from 'truncated', all pre-fix partial-codebook")
    print("measurements were distorted by the bug.")
    return results


if __name__ == "__main__":
    run_experiment()
