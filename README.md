# Clarity Prototype

Prototype validation of semantic-only voice transmission using the Kyutai Mimi neural audio codec.

## Quick Start

```bash
pip install -r requirements.txt
python -m pytest tests/
python experiments/01_token_separation.py
```

## What Is This?

Clarity explores whether transmitting only the semantic tokens from a neural audio codec
(~137.5 bps) can produce intelligible voice calls — a 99%+ bandwidth reduction versus
standard codecs like Opus.

The core technology is [Mimi](https://huggingface.co/kyutai/mimi), a neural audio codec
that separates speech into semantic tokens (what is said) and acoustic tokens (how it
sounds). By transmitting only semantic tokens and reconstructing audio with a local
speaker model, we aim to enable crystal-clear calls on any network.

## Project Structure

- `src/` — Core library (codec wrapper, quality metrics, utilities)
- `experiments/` — Experiment scripts (one per experiment)
- `results/` — Output audio files, plots, and metrics
- `tests/` — Unit tests
- `audio/` — Test audio files (not committed)

## Experiments

| # | Name | Purpose |
|---|------|---------|
| 01 | Token Separation | Validate semantic-only produces intelligible speech |
| 02 | Quality Sweep | Map codebook count vs. quality tradeoff |
| 03 | Prosody Analysis | Test if pitch/emotion survives semantic-only |
| 04 | Speaker Embedding | Test voice identity cross-combination |
| 05 | Bandwidth | Measure bitrates and compression ratios |
| 06 | Latency | Profile per-component timing |
