"""Experiment 10: Shared Reconstruction Evaluation Harness.

Every reconstruction route (experiments 11-16) is judged the same way:
given (source message, reconstruction, target-speaker reference bank), score

  - SIM      speaker similarity vs REAL recordings of the target speaker
  - DNSMOS   no-reference quality (goal: >= the degraded source = "better
             than the microphone")
  - WER      content fidelity vs the source transcript
  - F0 corr  performance preservation vs the source take

Also builds a listening page (HTML with embedded audio) so every experiment
ends with ears, not just numbers.

Import `evaluate_reconstruction` / `evaluate_arm` from other experiments;
running this module directly self-tests the harness on trivial arms
(identity, degraded, Mimi cb4/cb32) over a dataset's eval manifest.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.quality import (
    compute_dnsmos,
    compute_speaker_similarity,
    compute_wer,
    extract_pitch,
    pitch_correlation,
)
from src.results_io import save_metrics
from src.utils import load_audio

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# Success bar (plan Wave 0): the "how good could it get" question, testable
SUCCESS_BAR = {
    "sim_min": 0.80,            # vs real recordings of the speaker
    "dnsmos_beats_source": True,  # reconstruction >= degraded source
    "f0_corr_min": 0.90,        # the take is preserved
}


def f0_correlation(source: np.ndarray, recon: np.ndarray, sr: int) -> float:
    """Pitch-contour correlation between the source take and reconstruction."""
    _, f0_src = extract_pitch(source, sr)
    _, f0_rec = extract_pitch(recon, sr)
    return pitch_correlation(f0_src, f0_rec)


def evaluate_reconstruction(
    source_audio: np.ndarray,
    recon_audio: np.ndarray,
    reference_audio: np.ndarray,
    transcript: str,
    sr: int = 24000,
) -> dict:
    """Score one reconstruction. Returns {sim, dnsmos, wer, f0_corr}."""
    metrics: dict[str, float] = {}
    try:
        metrics["sim"] = compute_speaker_similarity(reference_audio, recon_audio, sr)
    except Exception as e:
        print(f"    SIM failed: {e}")
        metrics["sim"] = float("nan")
    try:
        metrics["dnsmos"] = compute_dnsmos(recon_audio, sr)
    except Exception as e:
        print(f"    DNSMOS failed: {e}")
        metrics["dnsmos"] = float("nan")
    try:
        metrics["wer"] = compute_wer(recon_audio, transcript, sr) if transcript else float("nan")
    except Exception as e:
        print(f"    WER failed: {e}")
        metrics["wer"] = float("nan")
    try:
        metrics["f0_corr"] = f0_correlation(source_audio, recon_audio, sr)
    except Exception as e:
        print(f"    F0 corr failed: {e}")
        metrics["f0_corr"] = float("nan")
    return metrics


def load_eval_set(dataset_dir: Path, degraded: bool = True) -> list[dict]:
    """Load the eval manifest of a voice_dataset build.

    Returns list of {id, clean_path, degraded_path, text}.
    """
    manifest = dataset_dir / "manifest_eval.jsonl"
    entries = []
    for line in manifest.read_text().splitlines():
        seg = json.loads(line)
        name = Path(seg["audio_path"]).name
        entries.append(
            {
                "id": Path(name).stem,
                "clean_path": dataset_dir / seg["audio_path"],
                "degraded_path": dataset_dir / "eval_degraded" / name,
                "text": seg["text"],
            }
        )
    return entries


def build_reference_bank(dataset_dir: Path, seconds: float = 60.0) -> np.ndarray:
    """Concatenate top-quality training segments into a speaker reference."""
    manifest = dataset_dir / "manifest_all.jsonl"
    chunks, total = [], 0.0
    for line in manifest.read_text().splitlines():
        seg = json.loads(line)
        audio, sr = load_audio(dataset_dir / seg["audio_path"], target_sr=24000)
        chunks.append(audio)
        total += len(audio) / sr
        if total >= seconds:
            break
    return np.concatenate(chunks)


def evaluate_arm(
    arm_name: str,
    reconstruct_fn,
    dataset_dir: Path,
    experiment_id: str,
    max_messages: int | None = None,
) -> dict:
    """Evaluate a reconstruction function over a dataset's eval set.

    Args:
        arm_name: Label for this arm.
        reconstruct_fn: (degraded_audio, sr, entry_dict) -> reconstructed audio.
        dataset_dir: A voice_dataset output dir.
        experiment_id: For results_io persistence.
        max_messages: Optionally limit eval-set size (for slow arms).

    Returns:
        {per_message: [...], means: {...}} — also persisted.
    """
    entries = load_eval_set(dataset_dir)
    if max_messages:
        entries = entries[:max_messages]
    reference = build_reference_bank(dataset_dir)

    per_message = []
    for entry in entries:
        clean, sr = load_audio(entry["clean_path"], target_sr=24000)
        degraded, _ = load_audio(entry["degraded_path"], target_sr=24000)
        print(f"  [{arm_name}] {entry['id']}")
        recon = reconstruct_fn(degraded, sr, entry)
        m = evaluate_reconstruction(degraded, recon, reference, entry["text"], sr)
        m["source_dnsmos"] = compute_dnsmos(degraded, sr)
        m["id"] = entry["id"]
        per_message.append(m)

    means = {}
    for key in ("sim", "dnsmos", "wer", "f0_corr", "source_dnsmos"):
        vals = [m[key] for m in per_message if not np.isnan(m.get(key, float("nan")))]
        means[key] = float(np.mean(vals)) if vals else float("nan")
    means["beats_microphone"] = bool(means["dnsmos"] >= means["source_dnsmos"])

    result = {"arm": arm_name, "per_message": per_message, "means": means}
    save_metrics(f"{experiment_id}_{arm_name}", result)
    print(
        f"  [{arm_name}] SIM {means['sim']:.3f} · DNSMOS {means['dnsmos']:.2f} "
        f"(src {means['source_dnsmos']:.2f}) · WER {means['wer']:.3f} · "
        f"F0 {means['f0_corr']:.3f}"
    )
    return result


def build_listening_page(
    clips: dict[str, list[tuple[str, Path]]], out_path: Path, title: str
) -> Path:
    """Minimal listening page: sections of labeled audio players, embedded audio.

    Args:
        clips: {section_title: [(label, wav_path), ...]}
        out_path: Output HTML path.
        title: Page title.
    """
    import base64

    parts = [
        f"<title>{title}</title>",
        "<style>body{font-family:system-ui;max-width:720px;margin:40px auto;"
        "padding:0 16px}h2{margin-top:32px}div.clip{margin:8px 0}"
        "span{display:inline-block;min-width:220px;font-size:14px}</style>",
        f"<h1>{title}</h1>",
    ]
    for section, items in clips.items():
        parts.append(f"<h2>{section}</h2>")
        for label, path in items:
            b64 = base64.b64encode(Path(path).read_bytes()).decode()
            parts.append(
                f'<div class="clip"><span>{label}</span>'
                f'<audio controls preload="none" src="data:audio/wav;base64,{b64}">'
                f"</audio></div>"
            )
    out_path.write_text("\n".join(parts))
    print(f"Listening page: {out_path}")
    return out_path


def run_selftest(dataset_dir: Path):
    """Harness self-test with trivial arms over a built dataset."""
    from src.codec import MimiCodec

    codec = MimiCodec(device="cpu")

    def identity(degraded, sr, entry):
        clean, _ = load_audio(entry["clean_path"], target_sr=24000)
        return clean

    def passthrough(degraded, sr, entry):
        return degraded

    def mimi_n(n):
        def fn(degraded, sr, entry):
            tokens = codec.encode(degraded, sr=sr)
            return codec.reconstruct_with_n_codebooks(tokens, n)
        return fn

    print("=" * 70)
    print("EXPERIMENT 10: Reconstruction Eval Harness — self-test")
    print("=" * 70)
    arms = {
        "upper_clean": identity,        # expect: high SIM, best DNSMOS, F0 vs degraded high
        "source_degraded": passthrough,  # expect: dnsmos == source, F0 corr 1.0
        "mimi_cb4": mimi_n(4),
        "mimi_cb32": mimi_n(32),
    }
    results = {}
    for name, fn in arms.items():
        results[name] = evaluate_arm(name, fn, dataset_dir, "10_harness_selftest", max_messages=5)

    print("\nSelf-test expectations: source_degraded F0 corr ~1.0; upper_clean")
    print("DNSMOS > source; mimi_cb32 > mimi_cb4 on SIM/DNSMOS.")
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True, help="voice_dataset output dir")
    args = parser.parse_args()
    run_selftest(args.dataset)
