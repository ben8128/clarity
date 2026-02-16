"""Audio quality measurement functions for Clarity experiments."""

from typing import Optional

import numpy as np

from .utils import resample


def compute_pesq(
    reference: np.ndarray,
    degraded: np.ndarray,
    sr: int = 24000,
    mode: str = "wb",
) -> float:
    """Compute PESQ (Perceptual Evaluation of Speech Quality).

    Args:
        reference: Reference (original) audio array.
        degraded: Degraded (reconstructed) audio array.
        sr: Sample rate of both signals.
        mode: PESQ mode — 'wb' (wideband, 16kHz) or 'nb' (narrowband, 8kHz).

    Returns:
        PESQ score (typically -0.5 to 4.5, higher is better).
    """
    from pesq import pesq

    target_sr = 16000 if mode == "wb" else 8000

    ref = resample(reference, sr, target_sr)
    deg = resample(degraded, sr, target_sr)

    # Align lengths
    min_len = min(len(ref), len(deg))
    ref = ref[:min_len]
    deg = deg[:min_len]

    return float(pesq(target_sr, ref, deg, mode))


def compute_stoi(
    reference: np.ndarray,
    degraded: np.ndarray,
    sr: int = 24000,
) -> float:
    """Compute STOI (Short-Time Objective Intelligibility).

    Args:
        reference: Reference audio array.
        degraded: Degraded audio array.
        sr: Sample rate of both signals.

    Returns:
        STOI score (0 to 1, higher is better).
    """
    from pystoi import stoi

    # STOI works best at 10kHz, but accepts any rate; resample to 16kHz for consistency
    target_sr = 16000
    ref = resample(reference, sr, target_sr)
    deg = resample(degraded, sr, target_sr)

    min_len = min(len(ref), len(deg))
    ref = ref[:min_len]
    deg = deg[:min_len]

    return float(stoi(ref, deg, target_sr, extended=False))


def compute_speaker_similarity(
    audio_a: np.ndarray,
    audio_b: np.ndarray,
    sr: int = 24000,
) -> float:
    """Compute speaker similarity using a pretrained speaker verification model.

    Uses the SpeechBrain ECAPA-TDNN model trained on VoxCeleb.

    Args:
        audio_a: First audio array.
        audio_b: Second audio array.
        sr: Sample rate.

    Returns:
        Cosine similarity score (-1 to 1, higher means more similar speakers).
    """
    import torch
    import torchaudio

    try:
        from speechbrain.pretrained import EncoderClassifier
    except ImportError:
        print("WARNING: speechbrain not installed. Install with: pip install speechbrain")
        print("Returning NaN for speaker similarity.")
        return float("nan")

    # Resample to 16kHz for SpeechBrain
    target_sr = 16000
    a = resample(audio_a, sr, target_sr)
    b = resample(audio_b, sr, target_sr)

    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": "cpu"},
    )

    emb_a = classifier.encode_batch(torch.tensor(a).unsqueeze(0))
    emb_b = classifier.encode_batch(torch.tensor(b).unsqueeze(0))

    similarity = torch.nn.functional.cosine_similarity(
        emb_a.squeeze(), emb_b.squeeze(), dim=0
    )
    return float(similarity)


def quality_report(
    original: np.ndarray,
    reconstructions: dict[str, np.ndarray],
    sr: int = 24000,
    compute_similarity: bool = False,
) -> dict[str, dict[str, float]]:
    """Generate a quality report comparing original vs multiple reconstructions.

    Args:
        original: Original reference audio.
        reconstructions: Dict mapping label → reconstructed audio array.
        sr: Sample rate.
        compute_similarity: Whether to compute speaker similarity (slow).

    Returns:
        Dict mapping label → {pesq, stoi, speaker_similarity (if requested)}.
    """
    results = {}

    for label, recon in reconstructions.items():
        print(f"Computing metrics for: {label}...")
        metrics: dict[str, float] = {}

        try:
            metrics["pesq"] = compute_pesq(original, recon, sr)
        except Exception as e:
            print(f"  PESQ failed for {label}: {e}")
            metrics["pesq"] = float("nan")

        try:
            metrics["stoi"] = compute_stoi(original, recon, sr)
        except Exception as e:
            print(f"  STOI failed for {label}: {e}")
            metrics["stoi"] = float("nan")

        if compute_similarity:
            try:
                metrics["speaker_similarity"] = compute_speaker_similarity(
                    original, recon, sr
                )
            except Exception as e:
                print(f"  Speaker similarity failed for {label}: {e}")
                metrics["speaker_similarity"] = float("nan")

        results[label] = metrics

    # Print summary table
    print("\n" + "=" * 60)
    print(f"{'Reconstruction':<30} {'PESQ':>8} {'STOI':>8}", end="")
    if compute_similarity:
        print(f" {'SpkSim':>8}", end="")
    print()
    print("-" * 60)

    for label, m in results.items():
        pesq_str = f"{m['pesq']:.3f}" if not np.isnan(m["pesq"]) else "N/A"
        stoi_str = f"{m['stoi']:.3f}" if not np.isnan(m["stoi"]) else "N/A"
        print(f"{label:<30} {pesq_str:>8} {stoi_str:>8}", end="")
        if compute_similarity:
            sim = m.get("speaker_similarity", float("nan"))
            sim_str = f"{sim:.3f}" if not np.isnan(sim) else "N/A"
            print(f" {sim_str:>8}", end="")
        print()

    print("=" * 60 + "\n")

    return results
