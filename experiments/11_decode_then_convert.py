"""Experiment 11: Decode-then-Convert (Route C, zero-training arms).

The first real reconstruction test: a degraded "message" is Mimi-encoded,
only the coarse codebooks cross the wire, the receiver decodes and converts
the result into the target speaker's voice using their reference audio.

Arms (all pretrained, no per-speaker training):
  wire_cb4            coarse decode alone (the 550bps baseline)
  cb4_knnvc           coarse decode -> kNN-VC toward the speaker bank
  cb1_knnvc           same from semantic-only (137bps) — how low can we go?
  degraded_knnvc      kNN-VC on the uncompressed degraded audio (VC ceiling,
                      no wire loss — isolates codec cost from VC cost)
  mimi_cb32           full-Mimi anchor (4.4kbps universal decode)

Seed-VC and resemble-enhance arms land in a follow-up (isolated venvs).
Run: python experiments/11_decode_then_convert.py --dataset data/dryrun
"""

import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.codec import MimiCodec
from src.results_io import save_metrics
from src.utils import load_audio, resample, save_audio

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

KNNVC_SR = 16000
TOPK = 4


def load_harness():
    """Import the shared harness (experiments/ is not a package)."""
    import importlib.util

    path = Path(__file__).resolve().parent / "10_reconstruction_eval.py"
    spec = importlib.util.spec_from_file_location("reconstruction_eval", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_matching_set(knn_vc, dataset_dir: Path, max_seconds: float = 300.0):
    """WavLM feature bank from the speaker's best training segments."""
    import json

    import torch

    paths, total = [], 0.0
    for line in (dataset_dir / "manifest_all.jsonl").read_text().splitlines():
        seg = json.loads(line)
        paths.append(str(dataset_dir / seg["audio_path"]))
        total += seg["duration_s"]
        if total >= max_seconds:
            break
    print(f"Reference bank: {len(paths)} segments, {total/60:.1f} min")
    with torch.no_grad():
        return knn_vc.get_matching_set(paths)


def knnvc_convert(knn_vc, matching_set, audio: np.ndarray, sr: int) -> np.ndarray:
    """Run kNN-VC on an audio array; returns 24kHz float32."""
    import torch

    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        save_audio(audio, tmp.name, sr, verbose=False)
        with torch.no_grad():
            query = knn_vc.get_features(tmp.name)
            out = knn_vc.match(query, matching_set, topk=TOPK)
    out_np = out.squeeze().cpu().numpy().astype(np.float32)
    return resample(out_np, KNNVC_SR, 24000)


def _patch_torchaudio_load() -> None:
    """torchaudio 2.9's load() requires torchcodec (removed: ffmpeg dylib
    mismatch). kNN-VC calls torchaudio.load internally — swap in soundfile."""
    import soundfile as sf
    import torch
    import torchaudio

    def sf_load(path, normalize=True, **kwargs):
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        return torch.from_numpy(data.T), sr

    torchaudio.load = sf_load


def run_experiment(dataset_dir: Path, max_messages: int | None = None):
    import torch

    _patch_torchaudio_load()
    harness = load_harness()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 70)
    print("EXPERIMENT 11: Decode-then-Convert (Route C, zero-training)")
    print("=" * 70)

    codec = MimiCodec(device="cpu")
    print("Loading kNN-VC (WavLM-Large + HiFi-GAN)...")
    knn_vc = torch.hub.load("bshall/knn-vc", "knn_vc", prematched=True,
                            trust_repo=True, device="cpu")
    matching_set = build_matching_set(knn_vc, dataset_dir)

    def wire_decode(n):
        def fn(degraded, sr, entry):
            tokens = codec.encode(degraded, sr=sr)
            return codec.reconstruct_with_n_codebooks(tokens, n)
        return fn

    def wire_then_knnvc(n):
        decode = wire_decode(n)
        def fn(degraded, sr, entry):
            wire_audio = decode(degraded, sr, entry)
            return knnvc_convert(knn_vc, matching_set, wire_audio, sr)
        return fn

    def direct_knnvc(degraded, sr, entry):
        return knnvc_convert(knn_vc, matching_set, degraded, sr)

    arms = {
        "wire_cb4": wire_decode(4),
        "cb4_knnvc": wire_then_knnvc(4),
        "cb1_knnvc": wire_then_knnvc(1),
        "degraded_knnvc": direct_knnvc,
        "mimi_cb32": wire_decode(32),
    }

    results = {}
    sample_clips: dict[str, list] = {}
    entries = harness.load_eval_set(dataset_dir)
    first_id = entries[0]["id"]

    for name, fn in arms.items():
        print(f"\n--- Arm: {name} ---")
        results[name] = harness.evaluate_arm(
            name, fn, dataset_dir, "11_decode_then_convert", max_messages=max_messages
        )
        # Save the first message's reconstruction for the listening page
        degraded, sr = load_audio(entries[0]["degraded_path"], target_sr=24000)
        recon = fn(degraded, sr, entries[0])
        clip_path = RESULTS_DIR / f"11_{name}_{first_id}_{timestamp}.wav"
        save_audio(recon, clip_path, 24000, verbose=False)
        sample_clips.setdefault("Reconstructions of one message", []).append((name, clip_path))

    # Anchors on the listening page
    clean_path = entries[0]["clean_path"]
    degraded_path = entries[0]["degraded_path"]
    sample_clips["Anchors"] = [
        ("clean original (studio)", clean_path),
        ("degraded source (phone-grade)", degraded_path),
    ]
    page = RESULTS_DIR / f"11_listening_{timestamp}.html"
    harness.build_listening_page(sample_clips, page, "Exp 11 — Decode then Convert")

    # Summary table
    print("\n" + "=" * 78)
    print(f"{'arm':<18} {'SIM':>7} {'DNSMOS':>7} {'srcMOS':>7} {'WER':>7} {'F0corr':>7} {'beats mic':>10}")
    print("-" * 78)
    for name, r in results.items():
        m = r["means"]
        print(f"{name:<18} {m['sim']:>7.3f} {m['dnsmos']:>7.2f} {m['source_dnsmos']:>7.2f} "
              f"{m['wer']:>7.3f} {m['f0_corr']:>7.3f} {str(m['beats_microphone']):>10}")
    print("=" * 78)

    save_metrics(
        "11_summary",
        {"dataset": str(dataset_dir), "arms": {k: v["means"] for k, v in results.items()}},
    )
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--max-messages", type=int, default=None)
    args = parser.parse_args()
    run_experiment(args.dataset, args.max_messages)
