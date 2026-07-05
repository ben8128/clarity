"""Experiment 12: Personal Mimi Decoder (Route D, Wave 2 smoke test).

The thesis test: keep the wire format universal (frozen encoder + quantizer,
same coarse tokens as exp 11), but fine-tune the RECEIVER's decoder on this
speaker's (degraded -> clean) pairs. If the same 550 bps tokens decode into
a cleaner, more-speaker-faithful voice, per-speaker adaptation works and
scales with training data (NVIDIA nano-codec precedent).

Arms:
  wire_cb4       frozen decoder baseline (same as exp 11's wire_cb4)
  cb4_personal   fine-tuned personal decoder, same tokens

Smoke-test success (dry-run speaker, 4.5 min of data):
  - mel loss decreases over training
  - cb4_personal > wire_cb4 on SIM and DNSMOS
The real bar (SIM >= 0.80, beats-microphone, F0 >= 0.90) is for Ben's studio
hours + a cloud GPU run with more steps.

Run: python experiments/12_personal_decoder.py --dataset data/dryrun
"""

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.codec import MimiCodec
from src.personal_decoder import (
    build_training_pairs,
    finetune_decoder,
    load_decoder_checkpoint,
    save_decoder_checkpoint,
)
from src.results_io import load_latest, save_metrics
from src.utils import load_audio, save_audio

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"


def load_harness():
    """Import the shared harness (experiments/ is not a package)."""
    import importlib.util

    path = Path(__file__).resolve().parent / "10_reconstruction_eval.py"
    spec = importlib.util.spec_from_file_location("reconstruction_eval", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def run_experiment(
    dataset_dir: Path,
    n_codebooks: int = 4,
    steps: int = 200,
    batch_size: int = 4,
    window_frames: int = 24,
    lr: float = 5e-5,
    tier: str = "5min",
    max_messages: int | None = 8,
    device: str = "auto",
    checkpoint: Path | None = None,
):
    device = pick_device(device)
    harness = load_harness()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 70)
    print("EXPERIMENT 12: Personal Mimi Decoder (Route D, Wave 2)")
    print(f"  codebooks={n_codebooks} steps={steps} tier={tier} device={device}")
    print("=" * 70)

    # Two codec instances: frozen baseline + the one we fine-tune
    codec = MimiCodec(device=device)
    personal = MimiCodec(device=device)

    train_result = None
    if checkpoint is None:
        pairs = build_training_pairs(dataset_dir, codec, n_codebooks, tier=tier)
        train_result = finetune_decoder(
            personal.model,
            pairs,
            steps=steps,
            batch_size=batch_size,
            window_frames=window_frames,
            lr=lr,
            device=device,
        )
        print(f"\nMel loss: {train_result.initial_loss:.4f} -> {train_result.final_loss:.4f}")
        checkpoint = CHECKPOINT_DIR / f"12_personal_cb{n_codebooks}_{tier}_{timestamp}.pt"
        save_decoder_checkpoint(personal.model, checkpoint)
    else:
        load_decoder_checkpoint(personal.model, checkpoint)
    personal.model.eval()

    def wire_decode(c: MimiCodec):
        def fn(degraded, sr, entry):
            tokens = c.encode(degraded, sr=sr)
            with torch.no_grad():
                return c.reconstruct_with_n_codebooks(tokens, n_codebooks)
        return fn

    arms = {
        f"wire_cb{n_codebooks}": wire_decode(codec),
        f"cb{n_codebooks}_personal": wire_decode(personal),
    }

    results = {}
    sample_clips: dict[str, list] = {}
    entries = harness.load_eval_set(dataset_dir)
    first_id = entries[0]["id"]

    for name, fn in arms.items():
        print(f"\n--- Arm: {name} ---")
        results[name] = harness.evaluate_arm(
            name, fn, dataset_dir, "12_personal_decoder", max_messages=max_messages
        )
        degraded, sr = load_audio(entries[0]["degraded_path"], target_sr=24000)
        recon = fn(degraded, sr, entries[0])
        clip_path = RESULTS_DIR / f"12_{name}_{first_id}_{timestamp}.wav"
        save_audio(recon, clip_path, 24000, verbose=False)
        sample_clips.setdefault("Reconstructions of one message", []).append((name, clip_path))

    sample_clips["Anchors"] = [
        ("clean original", entries[0]["clean_path"]),
        ("degraded source (phone-grade)", entries[0]["degraded_path"]),
    ]
    page = RESULTS_DIR / f"12_listening_{timestamp}.html"
    harness.build_listening_page(sample_clips, page, "Exp 12 — Personal Mimi Decoder")

    # Context rows from exp 11 (same eval set) for the printed comparison
    context_rows = {}
    try:
        prior = load_latest("11_summary")["metrics"]["arms"]
        for k in ("cb4_knnvc", "mimi_cb32"):
            if k in prior:
                context_rows[f"[11] {k}"] = prior[k]
    except Exception as e:
        print(f"(exp 11 summary unavailable for context: {e})")

    print("\n" + "=" * 78)
    print(f"{'arm':<20} {'SIM':>7} {'DNSMOS':>7} {'srcMOS':>7} {'WER':>7} {'F0corr':>7}")
    print("-" * 78)
    for name, r in results.items():
        m = r["means"]
        print(f"{name:<20} {m['sim']:>7.3f} {m['dnsmos']:>7.2f} {m['source_dnsmos']:>7.2f} "
              f"{m['wer']:>7.3f} {m['f0_corr']:>7.3f}")
    for name, m in context_rows.items():
        print(f"{name:<20} {m['sim']:>7.3f} {m['dnsmos']:>7.2f} {m['source_dnsmos']:>7.2f} "
              f"{m['wer']:>7.3f} {m['f0_corr']:>7.3f}")
    print("=" * 78)

    summary = {
        "dataset": str(dataset_dir),
        "n_codebooks": n_codebooks,
        "tier": tier,
        "device": device,
        "checkpoint": str(checkpoint),
        "training": (
            {
                "steps": train_result.steps,
                "initial_loss": train_result.initial_loss,
                "final_loss": train_result.final_loss,
                "loss_curve": train_result.loss_curve,
                "batch_size": batch_size,
                "window_frames": window_frames,
                "lr": lr,
            }
            if train_result
            else "loaded from checkpoint"
        ),
        "arms": {k: v["means"] for k, v in results.items()},
    }
    save_metrics("12_summary", summary)
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--codebooks", type=int, default=4)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--window-frames", type=int, default=24)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--tier", default="5min")
    parser.add_argument("--max-messages", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Skip training; evaluate this checkpoint")
    args = parser.parse_args()
    run_experiment(
        args.dataset,
        n_codebooks=args.codebooks,
        steps=args.steps,
        batch_size=args.batch_size,
        window_frames=args.window_frames,
        lr=args.lr,
        tier=args.tier,
        max_messages=args.max_messages,
        device=args.device,
        checkpoint=args.checkpoint,
    )
