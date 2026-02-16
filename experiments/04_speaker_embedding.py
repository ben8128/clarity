"""Experiment 04: Speaker Embedding / Voice Cross-Combination.

Tests whether acoustic tokens carry speaker identity independently of content.
Cross-combines semantic tokens from Speaker A with acoustic tokens from Speaker B.
"""

import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.codec import MimiCodec
from src.quality import compute_speaker_similarity
from src.utils import download_librispeech_sample, load_audio, save_audio

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
AUDIO_DIR = Path(__file__).resolve().parent.parent / "audio"


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
        print("Need 2 speakers. Downloading LibriSpeech samples...")
        from datasets import load_dataset

        ds = load_dataset(
            "hf-internal-testing/librispeech_asr_dummy",
            "clean",
            split="validation",
            trust_remote_code=True,
        )

        # Use first two samples as different "speakers"
        for i, label in enumerate(["speaker_A", "speaker_B"]):
            if label not in speakers and len(speakers) < 2:
                sample = ds[i]
                audio = np.array(sample["audio"]["array"], dtype=np.float32)
                orig_sr = sample["audio"]["sampling_rate"]
                from src.utils import resample
                audio_24k = resample(audio, orig_sr, 24000)
                speakers[label] = (audio_24k, 24000)

    return speakers


def run_experiment():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print("=" * 70)
    print("EXPERIMENT 04: Speaker Embedding / Cross-Combination")
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

    # Encode both
    print("\nEncoding Speaker A...")
    tokens_a = codec.encode(audio_a, sr=sr_a)
    print("Encoding Speaker B...")
    tokens_b = codec.encode(audio_b, sr=sr_b)

    codec.report_codebook_info(tokens_a)

    # Extract components
    semantic_a = codec.extract_semantic(tokens_a)
    acoustic_a = codec.extract_acoustic(tokens_a)
    semantic_b = codec.extract_semantic(tokens_b)
    acoustic_b = codec.extract_acoustic(tokens_b)

    # Reconstructions
    print("\nCreating reconstructions...")

    full_a = codec.decode(tokens_a)
    full_b = codec.decode(tokens_b)
    semantic_only_a = codec.reconstruct_semantic_only(tokens_a)

    # Cross-combination: A's words in B's voice
    cross_ab = codec.reconstruct_with_modified_acoustic(semantic_a, acoustic_b)
    # Cross-combination: B's words in A's voice
    cross_ba = codec.reconstruct_with_modified_acoustic(semantic_b, acoustic_a)

    # Save audio
    save_audio(full_a, RESULTS_DIR / f"04_{name_a}_full_{timestamp}.wav")
    save_audio(full_b, RESULTS_DIR / f"04_{name_b}_full_{timestamp}.wav")
    save_audio(semantic_only_a, RESULTS_DIR / f"04_{name_a}_semantic_only_{timestamp}.wav")
    save_audio(cross_ab, RESULTS_DIR / f"04_cross_{name_a}sem_{name_b}aco_{timestamp}.wav")
    save_audio(cross_ba, RESULTS_DIR / f"04_cross_{name_b}sem_{name_a}aco_{timestamp}.wav")

    # Speaker similarity analysis
    print("\nComputing speaker similarity scores...")
    print("(This uses SpeechBrain ECAPA-TDNN — may need to download model)")

    similarities = {}

    comparisons = [
        (f"A_full vs A_full", audio_a, full_a, sr_a),
        (f"A_full vs A_semantic", audio_a, semantic_only_a, sr_a),
        (f"A_full vs cross_AB", audio_a, cross_ab, sr_a),
        (f"B_full vs cross_AB", audio_b, cross_ab, sr_b),
        (f"B_full vs cross_BA", audio_b, cross_ba, sr_b),
        (f"A_full vs cross_BA", audio_a, cross_ba, sr_a),
    ]

    for label, ref, test, sr in comparisons:
        try:
            sim = compute_speaker_similarity(ref, test, sr)
            similarities[label] = sim
            print(f"  {label}: {sim:.3f}")
        except Exception as e:
            print(f"  {label}: FAILED ({e})")
            similarities[label] = float("nan")

    # Summary
    print("\n" + "=" * 70)
    print("EXPERIMENT 04 SUMMARY")
    print("=" * 70)
    print(f"\n{'Comparison':<30} {'Similarity':>12}")
    print("-" * 45)
    for label, sim in similarities.items():
        sim_str = f"{sim:.3f}" if not np.isnan(sim) else "N/A"
        print(f"{label:<30} {sim_str:>12}")
    print("-" * 45)

    print("\nInterpretation:")
    print("  - A_full vs cross_AB should be LOW (different speaker identity)")
    print("  - B_full vs cross_AB should be HIGH (same speaker identity)")
    print("  - This validates that acoustic tokens carry speaker identity")
    print(f"\nAudio saved to: {RESULTS_DIR}/")

    return similarities


if __name__ == "__main__":
    run_experiment()
