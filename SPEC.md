# Clarity Prototype — Experiment Specifications

## Hypothesis

Transmitting a tunable subset of Mimi's codebooks (1 to N) provides a continuous
quality/bandwidth tradeoff. Even at ALL codebooks, Mimi is 82%+ smaller than Opus
voice. The codebook count is a **product dial** for bandwidth conditions; the goal
is to characterize the whole curve and pick operating tiers.

**Status 2026-07-02: Phase 1 measured.** There is no sharp knee — quality climbs
monotonically to 32 codebooks (see `docs/FINDINGS.md`). Operating tiers chosen
from the curve:

| Tier | Codebooks | Bitrate | Use |
|---|---|---|---|
| A | 32 | 4.4 kbps | good networks — PESQ 3.4 / STOI 0.96 / SpkSim 0.95 |
| B | 8 | 1.1 kbps | degraded networks — STOI 0.90 / SpkSim 0.75 (clean speech) |
| C | 0 (+ transcript) | ~bytes | floor — text now, voice later or via Pocket-TTS reconstruction (exp 09) |

Tier B → A **progressive enhancement** (backfilling codebooks 8-31 so a stored
note upgrades in place) is the headline product feature enabled by RVQ structure.
See `docs/wire_format.md`.

## Experiments

Metrics for every experiment are persisted via `src.results_io.save_metrics()`
to `results/metrics/` — never stdout-only.

### Experiment 00: Zeroing vs Truncation A/B — DONE

Quantifies the partial-decode bug (dropped codebooks were zeroed; token id 0 is
a valid entry). Truncation is correct. Zeroing cost ~0.5 PESQ / 0.08-0.13 STOI
at n=4/8 — all pre-fix partial-codebook numbers were pessimistic.

### Experiment 01: Codebook Sweep — DONE

**Goal:** Characterize quality vs codebook count from 1 to N.

**Method:** 3+ samples (local `audio/` + LibriSpeech), sweep counts 1-32,
measure PESQ/STOI/speaker-similarity per count per sample, average, detect
sweet spot via `optimal_codebook_count()` (fraction-of-range criterion — the
marginal-gain heuristic is broken for non-monotonic gain curves).

**Result:** No knee; monotonic climb. Clean speech at 8 cb: STOI 0.90,
SpkSim 0.75. See FINDINGS.

### Experiment 01b: Mimi vs Opus — DONE

**Result:** Mimi 32 cb (4.4 kbps) PESQ 3.33 / STOI 0.96 — above Opus 6k,
slightly below Opus 12k, below Opus 24k. ~Opus-12k quality at ~1/3 the bits.

### Experiment 01c: Noise Isolation — DONE (hypothesis killed)

**Result:** Mimi does NOT denoise; it reproduces noise faithfully and slightly
damages speech (ΔSTOI negative at every SNR; airplane speech-band energy −24%).
Clean-voice delivery requires an explicit denoise step.

### Experiment 03: Prosody vs Codebook Count — DONE

**Goal:** Codebook count where pitch contour survives (F0 correlation > 0.7).

**Result:** 0.92 at 2 cb, 0.99 at 8 cb — prosody lives in the first couple of
codebooks. TODO: re-run with emotionally varied local recordings
(`audio/emotional_{happy,question,frustrated}.wav`) — currently LibriSpeech only.

### Experiment 04: Speaker Identity vs Codebook Count — DONE

**Goal:** Codebook count where speaker similarity exceeds 0.7; verify acoustic
tokens carry identity via cross-combination.

**Result:** Identity emerges at 8 cb on clean speech (0.75 → 0.94 at 32).
Cross-combination confirms: A-semantic + B-acoustic sounds like B (0.83 vs 0.16).

### Experiment 05: Bandwidth & Packet Loss — DONE

**Goal:** Quality under frame loss at the tier-B codebook count, three arms:
A zero-fill baseline, B repeat-last PLC, C DRED-style redundancy (depth 1/2/4),
uniform and bursty (Gilbert) patterns, rates 0-30%.

**Result:** Gate met — at 10% uniform loss, B: ΔSTOI −0.028; C depth-1
(2.2 kbps effective): ΔSTOI 0.000. Future arm D: masked-token-prediction PLC.

### Experiment 06: Latency Benchmark — DONE

**Goal:** Per-frame latency. Full-context HF chunks are a labeled desktop proxy;
the true streaming arm (moshi package, `mimi.streaming()`) is the number that
predicts phone behavior.

**Result:** Streaming 8 cb on M-series CPU: 48.8 ms encode+decode per 80 ms
frame (1.6x realtime). Matches T-Mimi's Galaxy S22 finding (~42 ms decode).

### Experiment 07: Golden Test Vectors — DONE

Exports (wav, tokens JSON, decoded wav) triples to `results/vectors/` for the
iOS port compatibility check (Phase 2 M2.1 go/no-go). Wire format v0 in
`docs/wire_format.md`.

### Experiment 08: Codec Alternatives — DEFERRED (stub)

FocalCodec-Stream / NVIDIA NanoCodec benchmark arms. Revisit triggers
documented in the stub docstring.

### Experiment 09: Floor Tier via Pocket TTS — PHASE 3

STCTS-style: transcript (+ sparse prosody) reconstructed by Kyutai Pocket TTS
voice-cloned from a receiver-cached ~10s enrollment WAV. Compare speaker
similarity + listening scores vs Mimi 8 cb. Go/no-go on tier C having voice.

### Listening Test — PENDING (human required)

`experiments/listening_test.py` — blind 1-5 ratings of
`results/reconstruction_*_codebooks.wav`. This is a Phase-1 gate criterion.

## Phase 1 → Phase 2 gate

| criterion | status |
|---|---|
| STOI ≥ 0.90 at ≤ 2.2 kbps (clean speech) | ✅ 8 cb / 1.1 kbps |
| Speaker similarity ≥ 0.7 at tier B | ✅ 0.75 at 8 cb |
| Subjective listening ≥ 4/5 at tier B | ⏳ pending |
| A loss arm keeps ΔSTOI < 0.05 at 10% loss | ✅ arms B & C |
| Golden vectors committed | ✅ |

## Phase 2 (next): iPhone-to-iPhone voice note

`ios/ClarityLab` (SwiftUI, vendored moshi-swift MLX) + `server/relay.py`
(websocket, pairing-code rooms). Milestones: M2.1 codec-on-device vs golden
vectors (go/no-go; fallback rustymimi via C FFI), M2.2 mic→encode→decode→speaker
loopback with per-frame instrumentation, M2.3 two-phone note over the relay
under Network Link Conditioner. Details in the project plan.
