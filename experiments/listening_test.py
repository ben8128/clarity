"""Listening Test: Subjective evaluation of codebook sweep reconstructions.

Plays reconstruction_*_codebooks.wav files in shuffled (blind) order, collects
1–5 ratings for intelligibility, naturalness, and speaker similarity, then prints
a combined table alongside objective metrics.
"""

import json
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def find_reconstruction_files() -> list[tuple[int, Path]]:
    """Find reconstruction_*_codebooks.wav files and extract codebook counts.

    Returns:
        List of (n_codebooks, path) sorted by codebook count.
    """
    files: list[tuple[int, Path]] = []
    for f in RESULTS_DIR.glob("reconstruction_*_codebooks.wav"):
        # Parse codebook count from filename
        stem = f.stem  # e.g. "reconstruction_4_codebooks"
        parts = stem.split("_")
        try:
            n = int(parts[1])
            files.append((n, f))
        except (IndexError, ValueError):
            continue
    return sorted(files, key=lambda x: x[0])


def play_audio(path: Path) -> bool:
    """Play an audio file. Tries sounddevice first, then system player.

    Returns:
        True if playback succeeded.
    """
    # Try sounddevice
    try:
        import sounddevice as sd
        import soundfile as sf

        data, sr = sf.read(str(path))
        sd.play(data, sr)
        sd.wait()
        return True
    except (ImportError, Exception):
        pass

    # Fallback: system player
    if sys.platform == "darwin":
        cmd = ["afplay", str(path)]
    elif sys.platform.startswith("linux"):
        cmd = ["aplay", str(path)]
    else:
        print(f"  Cannot play audio on {sys.platform}. Listen manually: {path}")
        return False

    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(f"  Playback failed. Listen manually: {path}")
        return False


def get_rating(prompt: str) -> int:
    """Prompt user for a 1–5 rating."""
    while True:
        try:
            val = int(input(f"  {prompt} (1–5): "))
            if 1 <= val <= 5:
                return val
            print("  Please enter a number between 1 and 5.")
        except (ValueError, EOFError):
            print("  Please enter a number between 1 and 5.")


def run_listening_test():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print("=" * 70)
    print("LISTENING TEST: Subjective Evaluation of Codebook Reconstructions")
    print("=" * 70)

    files = find_reconstruction_files()
    if not files:
        print(f"\nNo reconstruction files found in {RESULTS_DIR}/")
        print("Run experiment 01 first: python experiments/01_token_separation.py")
        return

    print(f"\nFound {len(files)} reconstruction(s): "
          f"{[n for n, _ in files]} codebook(s)")

    # Shuffle for blind evaluation
    shuffled = list(files)
    random.shuffle(shuffled)

    print("\nYou will hear each reconstruction in random order.")
    print("Rate each on a 1–5 scale (1=poor, 5=excellent).")
    print("Press Enter to begin...\n")
    input()

    ratings: list[dict] = []

    for idx, (n_codebooks, path) in enumerate(shuffled, 1):
        print(f"--- Sample {idx}/{len(shuffled)} ---")
        print(f"  Playing... (file hidden for blind test)")
        play_audio(path)

        intel = get_rating("Intelligibility (can you understand the words?)")
        natural = get_rating("Naturalness (does it sound like a real person?)")
        spk_sim = get_rating("Speaker similarity (same voice as original?)")

        ratings.append({
            "codebooks": n_codebooks,
            "intelligibility": intel,
            "naturalness": natural,
            "speaker_similarity": spk_sim,
            "file": str(path),
        })
        print()

    # Sort back by codebook count for display
    ratings.sort(key=lambda r: r["codebooks"])

    # Print results table
    print("\n" + "=" * 70)
    print("LISTENING TEST RESULTS")
    print("=" * 70)
    header = (
        f"{'Codebooks':>10} {'Intelli.':>10} {'Natural.':>10} {'SpkSim':>10}"
    )
    print(header)
    print("-" * len(header))
    for r in ratings:
        print(
            f"{r['codebooks']:>10} {r['intelligibility']:>10} "
            f"{r['naturalness']:>10} {r['speaker_similarity']:>10}"
        )
    print("=" * 70)

    # Save results
    output_path = RESULTS_DIR / f"listening_test_{timestamp}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(ratings, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    run_listening_test()
