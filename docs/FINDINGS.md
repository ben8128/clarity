# Clarity Findings — Lab Notebook

Running record of headline results. Every number here has a persisted source
in `results/metrics/`. Machine: Apple Silicon Mac, CPU inference,
transformers 4.57 / torch 2.9 (exact versions embedded in each metrics JSON).

## 00 — Zeroing vs Truncation (2026-07-02)

Source: `00_zeroing_vs_truncation_latest.json`

The original partial-codebook decode zeroed dropped codebooks; token id 0 is
a valid entry, so the decoder summed wrong embeddings. Measured impact on
clean LibriSpeech:

| n | variant | PESQ | STOI |
|---|---|---|---|
| 4 | truncated (fixed) | 1.463 | 0.829 |
| 4 | zeroed (bug) | 1.215 | 0.694 |
| 8 | truncated (fixed) | 1.995 | 0.899 |
| 8 | zeroed (bug) | 1.492 | 0.821 |

**All partial-codebook results from the 2026-02-18 runs were significantly
pessimistic.** Everything below uses truncation.

## 01 — Codebook Sweep (2026-07-02)

Source: `01_codebook_sweep_latest.json` (3 samples: airplane + 2 LibriSpeech)

**There is no quality knee below 32 codebooks.** PESQ, STOI, and speaker
similarity climb monotonically along the whole dial. Averaged: PESQ 1.07 →
3.45, STOI 0.34 → 0.85, SpkSim 0.22 → 0.93 from 1 → 32 cb.

Clean speech only (per-sample data in the JSON):

| cb | bitrate | PESQ | STOI | SpkSim |
|---|---|---|---|---|
| 8 | 1.1 kbps | ~2.1 | ~0.90 | ~0.78 |
| 16 | 2.2 kbps | ~2.8 | ~0.94 | ~0.88 |
| 32 | 4.4 kbps | ~3.4 | ~0.96 | ~0.95 |

Interpretation: "how many codebooks" is a **product dial** (bandwidth
budget), not a cliff to find. 8 cb is the tier-B candidate (meets the STOI
≥ 0.90 and SpkSim ≥ ~0.7 gates on clean speech); 32 cb is tier A.
Note: averages that include the airplane clip look worse because its
*reference* is noisy — see 01c.

## 01b — Mimi vs Opus (2026-07-02)

Source: `01b_mimi_vs_opus_latest.json` (clean LibriSpeech)

| codec | bitrate | PESQ | STOI |
|---|---|---|---|
| Mimi 32 cb | 4.4 kbps | 3.333 | 0.961 |
| Opus | 6 kbps | 1.756 | 0.901 |
| Opus | 12 kbps | 3.885 | 0.969 |
| Opus | 24 kbps | 4.501 | 0.993 |

**Honest framing: Mimi at 4.4 kbps ≈ Opus at ~10-12 kbps quality** — decisively
above Opus 6k, slightly below Opus 12k, clearly below Opus 24k. (The February
"Mimi 4.4k ≈ Opus 24k" claim was optimistic.) The bandwidth story holds:
comparable-quality Opus needs ~2.5x the bits, and Opus has nothing at all
below 6 kbps while Mimi's dial goes 30x lower.

## 01c — Noise Isolation (2026-07-02) — HYPOTHESIS KILLED

Source: `01c_noise_isolation_latest.json`

**Mimi does not denoise.** Synthetic noise (white/pink/babble at 20 to -5 dB
SNR): Mimi output scores slightly *worse* than the noisy input at every
condition (ΔSTOI −0.01 to −0.04). Airplane recording: speech-band energy
−24%, noise floor −0.6%, spectral SNR −8.0 → −9.2 dB. The token bottleneck
reproduces noise faithfully and slightly damages speech.

Consequence: if Clarity wants clean-voice delivery, denoising must be an
explicit pre-encode step (or a floor-tier reconstruction property) — it is
not free.

## 03 — Prosody vs Codebook Count (2026-07-02)

Source: `03_prosody_sweep_latest.json` (LibriSpeech; no emotional recordings yet)

F0 correlation with original: unmeasurable at 1 cb (pyin finds no voiced
frames — audio too degraded), **0.924 at 2 cb**, 0.969 at 4, 0.994 at 8,
0.998 at 32. **Prosody survives almost immediately** — it lives in the first
couple of codebooks.

## 04 — Speaker Identity vs Codebook Count (2026-07-02)

Source: `04_speaker_sweep_latest.json` (LibriSpeech 1272 + airplane voice)

Clean speech similarity vs original: −0.03 at 1 cb, 0.27 at 2, 0.65 at 4,
**0.75 at 8** (threshold 0.7), 0.85 at 16, 0.94 at 32.

Cross-combination (A's semantic + B's acoustic tokens): sounds like B
(similarity 0.83 vs B, 0.16 vs A) — **acoustic tokens carry speaker
identity**, confirming the Mimi split and the premise behind receiver-side
voice profiles.

## 05 — Packet Loss (2026-07-02)

Source: `05_packet_loss_latest.json` (8 cb, clean LibriSpeech, lossless
baseline STOI 0.899)

At 10% uniform packet loss:

| arm | STOI | Δ vs lossless |
|---|---|---|
| A zero-fill (naive) | 0.846 | −0.052 |
| B repeat-last PLC | 0.871 | −0.028 |
| C redundancy d=1 (2.2 kbps) | 0.899 | **0.000** |

**Gate met.** Repeat-last PLC alone passes (Δ < 0.05); DRED-style depth-1
redundancy makes 10% loss quality-free at double the (tiny) bitrate. Even
depth-4 redundancy (5.5 kbps) stays 4x below Opus voice. Bursty-pattern and
higher-rate results in the JSON.

## 06 — Latency (2026-07-02)

Source: `06_latency_latest.json`

True streaming Mimi (moshi reference implementation, 8 cb, CPU):
**encode 24.5 ms + decode 24.3 ms = 48.8 ms per 80 ms frame → 1.6x realtime**
(p95 ~33/35 ms). Consistent with Meta's T-Mimi finding of 42 ms/frame decode
on a Galaxy S22. Full-context HF chunk numbers are recorded as a labeled
proxy only.

Mouth-to-ear budget for future calls: 80 (frame) + ~49 (codec) + network —
~200-300 ms is plausible, matching the plan's assumption.

## Open items

- **Listening test** (`experiments/listening_test.py`) — needs a human;
  reconstruction WAVs for all sweep points are in `results/`.
- Emotional prosody recordings (`audio/emotional_*.wav`) not yet recorded —
  03 currently LibriSpeech-only.
- Experiment 08 (FocalCodec-Stream / NanoCodec arms) deferred; triggers
  documented in the stub.
- Experiment 09 (Pocket TTS floor tier) is Phase 3.

## Phase 1 → 2 gate assessment (as of 2026-07-02)

| criterion | status |
|---|---|
| STOI ≥ 0.90 at ≤ 2.2 kbps (clean) | ✅ 0.90 at 8 cb / 1.1 kbps |
| Speaker similarity ≥ 0.7 | ✅ 0.75 at 8 cb |
| Listening ≥ 4/5 | ⏳ pending human listening test |
| Loss arm keeps ΔSTOI < 0.05 at 10% | ✅ arms B and C |
| Golden vectors committed | ✅ results/vectors/*.tokens.json |
