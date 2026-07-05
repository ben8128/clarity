"""Personal Mimi decoder fine-tuning (Route D core).

The Clarity thesis: the wire format stays universal (frozen Mimi encoder +
quantizer), but the RECEIVER's decoder is fine-tuned per speaker on
(degraded input -> clean target) pairs. The same coarse tokens then decode
into a cleaner, more-speaker-faithful voice — reconstruction denoises by
construction because the training target is always the clean recording.

Trainable modules (the entire token->audio path after the quantizer):
  upsample, decoder_transformer, decoder
Frozen: encoder, encoder_transformer, downsample, quantizer.

Loss: multi-resolution log-mel L1. No adversarial loss in the prototype —
enough to measure whether per-speaker adaptation moves SIM/DNSMOS; a GAN
polish pass is a cloud-GPU follow-up if the direction holds.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

SAMPLE_RATE = 24000
SAMPLES_PER_FRAME = 1920  # 24000 Hz / 12.5 Hz
TRIM_FRAMES = 2  # skip decoder edge effects at window boundaries in the loss

TRAINABLE_MODULES = ("upsample", "decoder_transformer", "decoder")


class MultiResolutionMelLoss(torch.nn.Module):
    """L1 distance between log-mel spectrograms at several resolutions."""

    def __init__(self, sample_rate: int = SAMPLE_RATE, device: str = "cpu") -> None:
        super().__init__()
        import torchaudio

        self.transforms = torch.nn.ModuleList(
            [
                torchaudio.transforms.MelSpectrogram(
                    sample_rate=sample_rate,
                    n_fft=n_fft,
                    hop_length=n_fft // 4,
                    n_mels=80,
                    f_max=sample_rate / 2,
                )
                for n_fft in (512, 1024, 2048)
            ]
        ).to(device)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """pred/target: (batch, samples)."""
        loss = torch.zeros((), device=pred.device)
        for mel in self.transforms:
            log_pred = torch.log(mel(pred).clamp(min=1e-5))
            log_tgt = torch.log(mel(target).clamp(min=1e-5))
            loss = loss + F.l1_loss(log_pred, log_tgt)
        return loss / len(self.transforms)


@dataclass
class TrainingPair:
    """Pre-encoded coarse tokens of a degraded segment + the clean target."""

    tokens: torch.Tensor  # (n_codebooks, frames) on CPU
    clean: torch.Tensor  # (frames * SAMPLES_PER_FRAME,) on CPU
    segment_id: str = ""


def build_training_pairs(
    dataset_dir: Path,
    codec,
    n_codebooks: int,
    tier: str = "5min",
    snr_choices: tuple[float, ...] = (12.0, 15.0, 18.0, 21.0),
) -> list[TrainingPair]:
    """Encode each training segment's DEGRADED version; pair with clean audio.

    Degradation parameters vary deterministically per segment so the decoder
    sees a range of phone-grade conditions, all mapped to the clean target.
    """
    from .degrade import DegradeConfig, degrade_phone
    from .utils import load_audio

    manifest = dataset_dir / f"manifest_{tier}.jsonl"
    if not manifest.exists():
        manifest = dataset_dir / "manifest_all.jsonl"

    pairs: list[TrainingPair] = []
    lines = manifest.read_text().splitlines()
    print(f"Building training pairs from {manifest.name} ({len(lines)} segments)...")
    for idx, line in enumerate(lines):
        seg = json.loads(line)
        clean, sr = load_audio(dataset_dir / seg["audio_path"], target_sr=SAMPLE_RATE)
        cfg = DegradeConfig(seed=idx, snr_db=snr_choices[idx % len(snr_choices)])
        degraded = degrade_phone(clean, sr, cfg)
        tokens = codec.encode(degraded, sr=sr)[0, :n_codebooks, :].cpu()  # (n, F)
        n_frames = tokens.shape[-1]
        clean_t = torch.from_numpy(clean[: n_frames * SAMPLES_PER_FRAME].copy())
        if clean_t.shape[0] < n_frames * SAMPLES_PER_FRAME:
            clean_t = F.pad(clean_t, (0, n_frames * SAMPLES_PER_FRAME - clean_t.shape[0]))
        pairs.append(TrainingPair(tokens=tokens, clean=clean_t, segment_id=Path(seg["audio_path"]).stem))
        if (idx + 1) % 10 == 0:
            print(f"  encoded {idx + 1}/{len(lines)}")
    return pairs


def set_decoder_trainable(model) -> list[torch.nn.Parameter]:
    """Freeze everything except the token->audio path; return trainable params."""
    for p in model.parameters():
        p.requires_grad_(False)
    params: list[torch.nn.Parameter] = []
    for name in TRAINABLE_MODULES:
        module = getattr(model, name, None)
        if module is None:
            raise AttributeError(
                f"MimiModel has no module '{name}' — transformers layout changed?"
            )
        for p in module.parameters():
            p.requires_grad_(True)
            params.append(p)
    n = sum(p.numel() for p in params)
    print(f"Trainable decoder-side parameters: {n / 1e6:.1f}M")
    return params


@dataclass
class FinetuneResult:
    steps: int
    initial_loss: float
    final_loss: float
    loss_curve: list[float] = field(default_factory=list)  # mean loss per 10 steps


def finetune_decoder(
    model,
    pairs: list[TrainingPair],
    steps: int = 200,
    batch_size: int = 4,
    window_frames: int = 24,
    lr: float = 5e-5,
    device: str = "cpu",
    seed: int = 0,
    log_every: int = 10,
) -> FinetuneResult:
    """Fine-tune the decoder path with mel loss on random windows.

    Windows are cut at frame boundaries so token frame f aligns with audio
    samples [f*1920, (f+1)*1920). The first/last TRIM_FRAMES of each decoded
    window are excluded from the loss (missing streaming context at edges).
    """
    rng = np.random.default_rng(seed)
    params = set_decoder_trainable(model)
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    mel_loss = MultiResolutionMelLoss(device=device)
    trim = TRIM_FRAMES * SAMPLES_PER_FRAME

    usable = [p for p in pairs if p.tokens.shape[-1] >= window_frames]
    if not usable:
        raise ValueError(f"No segment has >= {window_frames} frames")
    print(f"Fine-tuning: {steps} steps, batch {batch_size}, window {window_frames} "
          f"frames ({window_frames * 0.08:.1f}s), lr {lr}, {len(usable)} segments, {device}")

    losses: list[float] = []
    curve: list[float] = []
    initial_loss = float("nan")
    for step in range(steps):
        tok_batch, aud_batch = [], []
        for _ in range(batch_size):
            pair = usable[rng.integers(len(usable))]
            max_off = pair.tokens.shape[-1] - window_frames
            off = int(rng.integers(max_off + 1))
            tok_batch.append(pair.tokens[:, off : off + window_frames])
            a0 = off * SAMPLES_PER_FRAME
            aud_batch.append(pair.clean[a0 : a0 + window_frames * SAMPLES_PER_FRAME])
        codes = torch.stack(tok_batch).to(device)  # (B, n, W)
        target = torch.stack(aud_batch).to(device)  # (B, W*1920)

        decoded = model.decode(codes).audio_values.squeeze(1)  # (B, samples)
        decoded = decoded[:, : target.shape[-1]]
        loss = mel_loss(decoded[:, trim:-trim], target[:, trim:-trim])

        if step == 0:
            initial_loss = float(loss.item())
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()

        losses.append(float(loss.item()))
        if (step + 1) % log_every == 0:
            recent = float(np.mean(losses[-log_every:]))
            curve.append(recent)
            print(f"  step {step + 1}/{steps}  mel loss {recent:.4f}")

    return FinetuneResult(
        steps=steps,
        initial_loss=initial_loss,
        final_loss=float(np.mean(losses[-log_every:])),
        loss_curve=curve,
    )


def save_decoder_checkpoint(model, path: Path) -> None:
    """Save only the trainable (decoder-side) weights."""
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        k: v.cpu()
        for k, v in model.state_dict().items()
        if k.split(".")[0] in TRAINABLE_MODULES
    }
    torch.save(state, path)
    size_mb = path.stat().st_size / 1e6
    print(f"Checkpoint saved: {path} ({size_mb:.0f} MB — this is the per-speaker "
          f"'voice profile' a receiver would hold)")


def load_decoder_checkpoint(model, path: Path) -> None:
    """Load decoder-side weights into a (fresh) MimiModel."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    unexpected = [k for k in unexpected if k.split(".")[0] in TRAINABLE_MODULES]
    if unexpected:
        raise RuntimeError(f"Checkpoint keys not in model: {unexpected[:5]}")
    print(f"Loaded personal decoder weights from {path}")
