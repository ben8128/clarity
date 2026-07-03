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


_speaker_classifier = None


def _get_speaker_classifier():
    """Load and cache the ECAPA-TDNN speaker verification model (slow to load)."""
    global _speaker_classifier
    if _speaker_classifier is None:
        try:
            try:
                from speechbrain.inference import EncoderClassifier  # speechbrain >= 1.0
            except ImportError:
                from speechbrain.pretrained import EncoderClassifier
        except ImportError:
            print("WARNING: speechbrain not installed. Install with: pip install speechbrain")
            print("Returning NaN for speaker similarity.")
            return None
        _speaker_classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": "cpu"},
        )
    return _speaker_classifier


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

    classifier = _get_speaker_classifier()
    if classifier is None:
        return float("nan")

    # Resample to 16kHz for SpeechBrain
    target_sr = 16000
    a = resample(audio_a, sr, target_sr)
    b = resample(audio_b, sr, target_sr)

    emb_a = classifier.encode_batch(torch.tensor(a).unsqueeze(0))
    emb_b = classifier.encode_batch(torch.tensor(b).unsqueeze(0))

    similarity = torch.nn.functional.cosine_similarity(
        emb_a.squeeze(), emb_b.squeeze(), dim=0
    )
    return float(similarity)


def extract_pitch(audio: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """Extract pitch contour (F0) using librosa's pyin.

    Returns:
        Tuple of (times, f0_values). f0_values may contain NaN for unvoiced frames.
    """
    import librosa

    f0, voiced_flag, voiced_probs = librosa.pyin(audio, fmin=50, fmax=500, sr=sr)
    times = librosa.times_like(f0, sr=sr)
    return times, f0


def pitch_correlation(f0_a: np.ndarray, f0_b: np.ndarray) -> float:
    """Compute correlation between two pitch contours, ignoring NaN frames."""
    min_len = min(len(f0_a), len(f0_b))
    a = f0_a[:min_len]
    b = f0_b[:min_len]

    # Only compare frames where both are voiced
    mask = ~(np.isnan(a) | np.isnan(b))
    if mask.sum() < 10:
        return float("nan")

    return float(np.corrcoef(a[mask], b[mask])[0, 1])


def compute_dnsmos(audio: np.ndarray, sr: int = 24000) -> float:
    """Compute DNSMOS overall quality (no-reference MOS, 1-5).

    No-reference is essential for reconstruction work: the goal is output
    BETTER than the source recording, which reference metrics can't express.

    Args:
        audio: Audio array.
        sr: Sample rate.

    Returns:
        DNSMOS OVRL score (higher is better; ~3.2+ is clean, ~3.7+ studio-like).
    """
    from speechmos import dnsmos

    audio16 = resample(audio, sr, 16000)
    result = dnsmos.run(np.clip(audio16, -1.0, 1.0), sr=16000)
    return float(result["ovrl_mos"])


_whisper_model = None


def compute_wer(audio: np.ndarray, reference_text: str, sr: int = 24000) -> float:
    """Compute word error rate of transcribed audio vs a reference transcript.

    Measures content fidelity: did the reconstruction preserve the words?

    Args:
        audio: Audio array to transcribe (faster-whisper small, cached).
        reference_text: Ground-truth transcript.
        sr: Sample rate.

    Returns:
        WER in [0, inf) — 0.0 is perfect.
    """
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel

        _whisper_model = WhisperModel("small", device="cpu", compute_type="int8")

    import soundfile as sf
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        sf.write(tmp.name, audio, sr)
        segments, _ = _whisper_model.transcribe(tmp.name, language="en")
        hypothesis = " ".join(s.text.strip() for s in segments).strip()

    return _word_error_rate(reference_text, hypothesis)


def _normalize_words(text: str) -> list[str]:
    import re

    return re.sub(r"[^a-z0-9' ]+", " ", text.lower()).split()


def _word_error_rate(reference: str, hypothesis: str) -> float:
    """Levenshtein WER between two transcripts."""
    ref = _normalize_words(reference)
    hyp = _normalize_words(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0

    d = np.zeros((len(ref) + 1, len(hyp) + 1), dtype=np.int32)
    d[:, 0] = np.arange(len(ref) + 1)
    d[0, :] = np.arange(len(hyp) + 1)
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + cost)
    return float(d[len(ref), len(hyp)] / len(ref))


def optimal_codebook_count(
    sweep_results: list[dict],
    metric: str = "pesq",
    fraction_of_range: float = 0.9,
    min_gain: Optional[float] = None,
) -> int:
    """Find the smallest codebook count achieving most of the available quality.

    Returns the smallest count whose metric reaches
    ``min + fraction_of_range * (max - min)`` across the sweep. This is robust
    to non-monotonic marginal gains, which broke the earlier
    "first gain below threshold" heuristic (the 2026-07-02 run of exp 01 has
    per-codebook gains that dip at n=3 and then rise again through n=32).

    Args:
        sweep_results: List of dicts with keys 'codebooks', 'pesq', 'stoi',
            and optionally 'speaker_similarity'.
        metric: Which metric to analyze ('pesq', 'stoi', or 'speaker_similarity').
        fraction_of_range: How much of the metric's observed range must be
            reached (0.9 = within 10% of the best observed quality).
        min_gain: Deprecated, ignored. Kept for call-site compatibility.

    Returns:
        The optimal number of codebooks (sweet spot).
    """
    sorted_results = sorted(sweep_results, key=lambda r: r["codebooks"])
    sorted_results = [r for r in sorted_results if not np.isnan(r.get(metric, float("nan")))]

    if len(sorted_results) <= 1:
        return sorted_results[0]["codebooks"] if sorted_results else 1

    values = [r[metric] for r in sorted_results]
    threshold = min(values) + fraction_of_range * (max(values) - min(values))

    for r in sorted_results:
        if r[metric] >= threshold:
            return r["codebooks"]

    return sorted_results[-1]["codebooks"]


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
