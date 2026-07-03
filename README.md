# Clarity Prototype

Prototype validation for voice notes (and eventually calls) that transmit
Mimi neural-codec tokens instead of audio — a tunable 0.14–4.4 kbps versus
Opus's ~24 kbps.

## Quick Start

```bash
pip install -r requirements.txt
python -m pytest tests/
python experiments/01_token_separation.py
```

## What Is This?

Clarity explores transmitting [Mimi](https://huggingface.co/kyutai/mimi)
neural-codec tokens phone-to-phone. Mimi encodes 24kHz speech into 12.5Hz
frames of up to 32 residual codebooks. The number of codebooks transmitted is
a **dial**: 1 codebook = 137.5 bps, 8 = 1.1 kbps, 32 = 4.4 kbps — all far
below Opus, with quality rising smoothly along the dial.

Key measured results so far (see `docs/FINDINGS.md` for the full lab notebook):

- **Mimi at 4.4 kbps ≈ Opus at ~12 kbps quality** on clean speech
  (PESQ 3.33/STOI 0.96 vs Opus 12k's 3.89/0.97), and far above Opus 6k.
- **No quality knee below 32 codebooks** — PESQ/STOI/speaker-similarity all
  climb monotonically. "How many codebooks" is a product decision
  (bandwidth budget), not a cliff to find.
- **At 8 codebooks (1.1 kbps)**, clean speech reaches STOI ~0.90 and speaker
  similarity ~0.78 — the tier-B candidate for terrible networks.
- **Mimi does NOT denoise** — the "free noise reduction" hypothesis is dead
  (experiment 01c).

## Project Structure

- `src/` — Core library (codec wrapper, quality metrics, results persistence)
- `experiments/` — Experiment scripts (one per experiment)
- `results/metrics/` — Persisted experiment metrics (JSON, committed)
- `results/` — Audio outputs and plots (audio gitignored)
- `docs/` — Findings notebook, wire format spec
- `tests/` — Unit tests
- `audio/` — Test audio files (not committed)

## Experiments

| # | Name | Purpose | Status |
|---|------|---------|--------|
| 00 | Zeroing vs Truncation | Quantify the partial-decode bug | done |
| 01 | Codebook Sweep | Quality vs codebook count, knee detection | done |
| 01b | Mimi vs Opus | Head-to-head at multiple bitrates | done |
| 01c | Noise Isolation | Does the token bottleneck denoise? (no) | done |
| 03 | Prosody Sweep | Codebook count where pitch contour survives | done |
| 04 | Speaker Identity Sweep | Codebook count where identity survives | done |
| 05 | Packet Loss | Loss arms: zero-fill / repeat-last / redundancy | done |
| 06 | Latency | Full-context proxy + true streaming per-frame | done |
| 07 | Test Vectors | Golden vectors for the iOS port | done |
| 08 | Codec Alternatives | FocalCodec-Stream, NanoCodec | deferred (see stub) |
| — | listening_test.py | Blind subjective ratings | needs a human |

## Roadmap

Phase 1 (this repo): lock the science — done except the human listening test.
Phase 2: iPhone-to-iPhone voice note demo (`ios/ClarityLab`, moshi-swift/MLX).
Phase 3: product-shaped demo — voice+text notes, tiered quality with
progressive enhancement, Pocket-TTS floor-tier spike.
