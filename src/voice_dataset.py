"""Voice dataset builder: long recordings → segmented, transcribed, quality-
filtered training tiers for personal voice models.

Pipeline (the passive-collection recipe a messaging app would run):
  1. Load each source recording, resample to 24kHz mono
  2. VAD segmentation (silero-vad) into utterances (merge to 3-15s chunks)
  3. Transcribe each segment (faster-whisper)
  4. Score each segment (DNSMOS via speechmos)
  5. Rank by quality, emit manifest JSONL + duration tiers + held-out eval set

Usage:
    python -m src.voice_dataset --input audio/podcast/ --output data/ben
"""

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

SAMPLE_RATE = 24000
TIERS_MINUTES = {"5min": 5, "30min": 30, "1h": 60, "3h": 180}
EVAL_SET_SIZE = 20
MIN_SEGMENT_S = 3.0
MAX_SEGMENT_S = 15.0
MIN_DNSMOS = 2.8  # drop clearly bad segments outright


@dataclass
class Segment:
    """One VAD-cut utterance with transcript and quality score."""

    audio_path: str
    source_file: str
    start_s: float
    end_s: float
    duration_s: float
    text: str = ""
    dnsmos: float = float("nan")

    def to_json(self) -> str:
        return json.dumps(asdict(self))


def _load_vad():
    from silero_vad import load_silero_vad

    return load_silero_vad()


def vad_segments(
    audio: np.ndarray, sr: int, vad_model
) -> Iterator[tuple[float, float]]:
    """Yield merged (start_s, end_s) speech spans of MIN..MAX duration."""
    import torch
    from silero_vad import get_speech_timestamps

    from .utils import resample

    audio16 = resample(audio, sr, 16000)
    stamps = get_speech_timestamps(
        torch.from_numpy(audio16), vad_model, sampling_rate=16000,
        return_seconds=True,
    )

    # Merge adjacent speech spans into chunks of MIN_SEGMENT_S..MAX_SEGMENT_S
    cur_start: Optional[float] = None
    cur_end: Optional[float] = None
    for ts in stamps:
        start, end = ts["start"], ts["end"]
        if cur_start is None:
            cur_start, cur_end = start, end
            continue
        if end - cur_start <= MAX_SEGMENT_S and start - cur_end < 1.0:
            cur_end = end
            continue
        if cur_end - cur_start >= MIN_SEGMENT_S:
            yield cur_start, cur_end
        cur_start, cur_end = start, end
    if cur_start is not None and cur_end - cur_start >= MIN_SEGMENT_S:
        yield cur_start, cur_end


def transcribe(audio_path: Path, whisper_model) -> str:
    """Transcribe one segment file with faster-whisper."""
    segments, _info = whisper_model.transcribe(str(audio_path), language="en")
    return " ".join(s.text.strip() for s in segments).strip()


def score_dnsmos(audio: np.ndarray, sr: int) -> float:
    """Overall DNSMOS (OVRL) score for one segment."""
    from .quality import compute_dnsmos

    return compute_dnsmos(audio, sr)


def build_dataset(
    input_dir: Path,
    output_dir: Path,
    max_files: Optional[int] = None,
    whisper_size: str = "small",
) -> dict:
    """Run the full pipeline; returns summary stats."""
    from faster_whisper import WhisperModel

    from .degrade import degrade_phone
    from .utils import load_audio, save_audio

    seg_dir = output_dir / "segments"
    seg_dir.mkdir(parents=True, exist_ok=True)

    print("Loading models (silero-vad, faster-whisper)...")
    vad_model = _load_vad()
    whisper_model = WhisperModel(whisper_size, device="cpu", compute_type="int8")

    source_files = sorted(
        p for ext in ("*.wav", "*.mp3", "*.flac", "*.m4a", "*.aiff")
        for p in input_dir.rglob(ext)
    )
    if max_files:
        source_files = source_files[:max_files]
    if not source_files:
        raise FileNotFoundError(f"No audio files found under {input_dir}")
    print(f"Found {len(source_files)} source file(s)")

    segments: list[Segment] = []
    for src in source_files:
        print(f"\n--- {src.name} ---")
        audio, sr = load_audio(src, target_sr=SAMPLE_RATE)
        duration = len(audio) / sr
        print(f"  {duration/60:.1f} min loaded")

        n_before = len(segments)
        for start_s, end_s in vad_segments(audio, sr, vad_model):
            clip = audio[int(start_s * sr): int(end_s * sr)]
            seg_id = f"{src.stem}_{int(start_s*1000):09d}"
            seg_path = seg_dir / f"{seg_id}.wav"
            save_audio(clip, seg_path, sr, verbose=False)

            score = score_dnsmos(clip, sr)
            if score < MIN_DNSMOS:
                seg_path.unlink()
                continue
            text = transcribe(seg_path, whisper_model)
            if not text:
                seg_path.unlink()
                continue
            segments.append(
                Segment(
                    audio_path=str(seg_path.relative_to(output_dir)),
                    source_file=src.name,
                    start_s=round(start_s, 2),
                    end_s=round(end_s, 2),
                    duration_s=round(end_s - start_s, 2),
                    text=text,
                    dnsmos=round(score, 3),
                )
            )
        kept = len(segments) - n_before
        print(f"  kept {kept} segments")

    if not segments:
        raise RuntimeError("No usable segments survived filtering")

    # Rank by quality (best first) for tiering
    segments.sort(key=lambda s: s.dnsmos, reverse=True)

    # Held-out eval set: take EVAL_SET_SIZE spread across the quality range
    # (not just the best — messages arrive at all qualities)
    idx = np.linspace(0, len(segments) - 1, min(EVAL_SET_SIZE, len(segments))).astype(int)
    eval_ids = set(idx.tolist())
    eval_set = [s for i, s in enumerate(segments) if i in eval_ids]
    train_pool = [s for i, s in enumerate(segments) if i not in eval_ids]

    # Degraded (phone-grade) copies of the eval set
    eval_deg_dir = output_dir / "eval_degraded"
    eval_deg_dir.mkdir(exist_ok=True)
    for s in eval_set:
        audio, sr = load_audio(output_dir / s.audio_path, target_sr=SAMPLE_RATE)
        deg = degrade_phone(audio, sr)
        save_audio(deg, eval_deg_dir / Path(s.audio_path).name, sr, verbose=False)

    # Manifests
    (output_dir / "manifest_all.jsonl").write_text(
        "\n".join(s.to_json() for s in train_pool) + "\n"
    )
    (output_dir / "manifest_eval.jsonl").write_text(
        "\n".join(s.to_json() for s in eval_set) + "\n"
    )

    # Tiers: best-quality-first cumulative duration cuts
    tier_stats = {}
    for tier, minutes in TIERS_MINUTES.items():
        budget = minutes * 60
        chosen, total = [], 0.0
        for s in train_pool:
            if total >= budget:
                break
            chosen.append(s)
            total += s.duration_s
        (output_dir / f"manifest_{tier}.jsonl").write_text(
            "\n".join(s.to_json() for s in chosen) + "\n"
        )
        tier_stats[tier] = {"segments": len(chosen), "minutes": round(total / 60, 1)}

    durations = [s.duration_s for s in train_pool]
    scores = [s.dnsmos for s in train_pool]
    stats = {
        "source_files": len(source_files),
        "segments_kept": len(train_pool),
        "eval_segments": len(eval_set),
        "total_hours": round(sum(durations) / 3600, 2),
        "dnsmos_mean": round(float(np.mean(scores)), 3),
        "dnsmos_p10": round(float(np.percentile(scores, 10)), 3),
        "dnsmos_p90": round(float(np.percentile(scores, 90)), 3),
        "tiers": tier_stats,
    }
    (output_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    print("\n=== Dataset summary ===")
    print(json.dumps(stats, indent=2))
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Dir of long recordings")
    parser.add_argument("--output", type=Path, required=True, help="Output dataset dir")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--whisper-size", default="small")
    args = parser.parse_args()
    build_dataset(args.input, args.output, args.max_files, args.whisper_size)


if __name__ == "__main__":
    main()
