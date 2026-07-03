"""Experiment 07: Export Golden Test Vectors.

Exports (wav, tokens JSON, decoded wav) triples that the iOS port (moshi-swift
or rustymimi) must reproduce. Token compatibility between the Python HF
implementation and the on-device implementation is the single biggest
cross-platform risk — these vectors are the go/no-go check for Phase 2 M2.1.

Token JSONs are small and committed to git; wav files are gitignored.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.codec import MimiCodec
from src.results_io import save_metrics
from src.utils import download_librispeech_sample, load_audio, save_audio

VECTORS_DIR = Path(__file__).resolve().parent.parent / "results" / "vectors"
AUDIO_DIR = Path(__file__).resolve().parent.parent / "audio"


def collect_clips() -> list[tuple[str, np.ndarray, int]]:
    """Two short clips: one clean LibriSpeech, one local recording if present."""
    clips = []
    audio, sr = download_librispeech_sample()
    clips.append(("librispeech_clean", audio[: 5 * sr], sr))

    airplane = AUDIO_DIR / "airplane_handheld.wav"
    if airplane.exists():
        noisy, sr2 = load_audio(airplane, target_sr=24000)
        clips.append(("airplane_handheld", noisy[: 5 * sr2], sr2))
    return clips


def run_experiment():
    print("=" * 70)
    print("EXPERIMENT 07: Export Golden Test Vectors")
    print("=" * 70)

    VECTORS_DIR.mkdir(parents=True, exist_ok=True)
    codec = MimiCodec(device="cpu")

    exported = []
    for label, audio, sr in collect_clips():
        print(f"\n--- {label} ({len(audio)/sr:.2f}s) ---")
        wav_path = VECTORS_DIR / f"{label}.wav"
        save_audio(audio, wav_path, sr)

        tokens = codec.encode(audio, sr=sr)
        decoded = codec.decode(tokens)
        save_audio(decoded, VECTORS_DIR / f"{label}.decoded.wav", sr)

        token_record = {
            "label": label,
            "sample_rate": sr,
            "num_samples": len(audio),
            "frame_rate_hz": 12.5,
            "shape": list(tokens.shape),
            "num_codebooks": tokens.shape[1],
            "num_frames": tokens.shape[2],
            "tokens": tokens.squeeze(0).cpu().numpy().tolist(),
        }
        tokens_path = VECTORS_DIR / f"{label}.tokens.json"
        tokens_path.write_text(json.dumps(token_record))
        size_kb = tokens_path.stat().st_size / 1024
        print(f"  {tokens.shape[1]} codebooks x {tokens.shape[2]} frames")
        print(f"  Saved: {wav_path.name}, {tokens_path.name} ({size_kb:.0f}KB), "
              f"{label}.decoded.wav")
        exported.append(
            {
                "label": label,
                "num_codebooks": tokens.shape[1],
                "num_frames": tokens.shape[2],
                "tokens_json": str(tokens_path),
            }
        )

    save_metrics("07_test_vectors", {"exported": exported})
    print("\nPhase 2 M2.1 check: the iOS Mimi must (a) produce these tokens from")
    print("the .wav (exact match or documented tolerance) and (b) decode these")
    print("tokens to audio perceptually identical to .decoded.wav.")
    return exported


if __name__ == "__main__":
    run_experiment()
