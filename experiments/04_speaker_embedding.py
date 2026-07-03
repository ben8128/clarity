"""Experiment 04: Speaker Identity vs Codebook Count.

Primary question: at which codebook count does speaker identity survive
transmission? Sweeps codebook counts for two speakers and measures ECAPA-TDNN
speaker similarity between each speaker's original audio and reconstruction.

Secondary probe: cross-combines semantic tokens from Speaker A with acoustic
tokens from Speaker B to verify acoustic tokens carry the identity.
"""

import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.codec import MimiCodec
from src.quality import compute_speaker_similarity
from src.results_io import save_metrics
from src.utils import download_librispeech_sample, load_audio, save_audio

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
AUDIO_DIR = Path(__file__).resolve().parent.parent / "audio"

SWEEP_POINTS = [1, 2, 4, 8, 16, 32]
SIMILARITY_THRESHOLD = 0.7


def find_two_speakers() -> dict[str, tuple[np.ndarray, int]]:
    """Find audio from two different speakers."""
    speakers = {}

    if AUDIO_DIR.exists():
        for prefix in ("speaker1", "speaker2"):
            matches = list(AUDIO_DIR.glob(f"{prefix}_*.*"))
            if matches:
                audio, sr = load_audio(matches[0])
                speakers[prefix] = (audio, sr)

    if len(speakers) < 2:
        # The dummy LibriSpeech set has a single speaker (1272), so fall back
        # to LibriSpeech as speaker A and the local airplane recording (a real
        # different voice, albeit noisy) as speaker B.
        print("Need 2 speakers: LibriSpeech + airplane_handheld fallback...")
        audio, sr = download_librispeech_sample()
        speakers["librispeech_1272"] = (audio, sr)

        airplane = AUDIO_DIR / "airplane_handheld.wav"
        if airplane.exists():
            noisy, sr2 = load_audio(airplane, target_sr=24000)
            speakers["airplane_handheld"] = (noisy, sr2)
            print("NOTE: speaker B is a noisy recording — similarity scores for")
            print("      B will run lower; the sweep shape is still informative.")

    return speakers


def run_experiment():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print("=" * 70)
    print("EXPERIMENT 04: Speaker Identity vs Codebook Count")
    print("=" * 70)

    codec = MimiCodec(device="cpu")
    speakers = find_two_speakers()
    speaker_names = list(speakers.keys())

    if len(speaker_names) < 2:
        print("ERROR: Need at least 2 speakers. Aborting.")
        return {}

    name_a, name_b = speaker_names[0], speaker_names[1]
    audio_a, sr_a = speakers[name_a]
    audio_b, sr_b = speakers[name_b]

    print(f"\nSpeaker A: {name_a} ({len(audio_a)/sr_a:.2f}s)")
    print(f"Speaker B: {name_b} ({len(audio_b)/sr_b:.2f}s)")

    tokens = {
        name_a: codec.encode(audio_a, sr=sr_a),
        name_b: codec.encode(audio_b, sr=sr_b),
    }
    codec.report_codebook_info(tokens[name_a])

    # === TEST 1: Speaker similarity vs codebook count ===
    print("\n--- TEST 1: Speaker similarity vs codebook count ---")
    sweep_points = [n for n in SWEEP_POINTS if n <= tokens[name_a].shape[1]]
    sweep: dict[str, list[dict]] = {}

    for name, audio, sr in ((name_a, audio_a, sr_a), (name_b, audio_b, sr_b)):
        print(f"\n  {name}:")
        sweep[name] = []
        for n in sweep_points:
            recon = codec.reconstruct_with_n_codebooks(tokens[name], n)
            sim = compute_speaker_similarity(audio, recon, sr)
            sweep[name].append({"codebooks": n, "speaker_similarity": sim})
            print(f"    {n:>2} codebook(s): similarity = {sim:.3f}")
            if name == name_a and n in (1, 4, 8, 32):
                save_audio(recon, RESULTS_DIR / f"04_{name}_cb{n:02d}_{timestamp}.wav", sr)

    # First count clearing the threshold, per speaker
    identity_emerges_at = {}
    for name, entries in sweep.items():
        identity_emerges_at[name] = next(
            (
                e["codebooks"]
                for e in entries
                if not np.isnan(e["speaker_similarity"])
                and e["speaker_similarity"] > SIMILARITY_THRESHOLD
            ),
            None,
        )

    # === TEST 2: Cross-combination probe ===
    print("\n--- TEST 2: Cross-combination (A's words, B's voice) ---")
    semantic_a = codec.extract_semantic(tokens[name_a])
    acoustic_a = codec.extract_acoustic(tokens[name_a])
    semantic_b = codec.extract_semantic(tokens[name_b])
    acoustic_b = codec.extract_acoustic(tokens[name_b])

    cross_ab = codec.reconstruct_with_modified_acoustic(semantic_a, acoustic_b)
    cross_ba = codec.reconstruct_with_modified_acoustic(semantic_b, acoustic_a)

    save_audio(cross_ab, RESULTS_DIR / f"04_cross_{name_a}sem_{name_b}aco_{timestamp}.wav")
    save_audio(cross_ba, RESULTS_DIR / f"04_cross_{name_b}sem_{name_a}aco_{timestamp}.wav")

    cross_similarities = {}
    comparisons = [
        ("A_orig vs cross_AB", audio_a, cross_ab, sr_a),
        ("B_orig vs cross_AB", audio_b, cross_ab, sr_b),
        ("B_orig vs cross_BA", audio_b, cross_ba, sr_b),
        ("A_orig vs cross_BA", audio_a, cross_ba, sr_a),
    ]
    for label, ref, test, sr in comparisons:
        try:
            sim = compute_speaker_similarity(ref, test, sr)
        except Exception as e:
            print(f"  {label}: FAILED ({e})")
            sim = float("nan")
        cross_similarities[label] = sim
        print(f"  {label}: {sim:.3f}")

    # Summary
    print("\n" + "=" * 70)
    print("EXPERIMENT 04 SUMMARY")
    print("=" * 70)
    for name, at in identity_emerges_at.items():
        at_str = f"{at} codebook(s)" if at else f"never (> {SIMILARITY_THRESHOLD})"
        print(f"  {name}: identity emerges at {at_str}")
    print("\nCross-combination interpretation:")
    print("  - A_orig vs cross_AB should be LOW (identity came from B)")
    print("  - B_orig vs cross_AB should be HIGH (acoustic tokens carry identity)")
    print(f"\nAudio saved to: {RESULTS_DIR}/")

    save_metrics(
        "04_speaker_sweep",
        {
            "similarity_threshold": SIMILARITY_THRESHOLD,
            "speakers": speaker_names,
            "sweep": sweep,
            "identity_emerges_at": identity_emerges_at,
            "cross_combination": cross_similarities,
        },
    )
    return sweep


if __name__ == "__main__":
    run_experiment()
