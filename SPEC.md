# Clarity Prototype — Experiment Specifications

## Hypothesis

Transmitting only Mimi's semantic tokens (codebook 0) at ~137.5 bps and reconstructing
audio with a personalized acoustic decoder on the receiver's device can produce
intelligible, recognizable voice calls — a 99%+ bandwidth reduction versus standard
audio codecs like Opus.

## Experiments

### Experiment 01: Token Separation Validation

**Goal:** Validate that Mimi's codebook 0 (semantic tokens) alone produces intelligible speech.

**Method:**
1. Load test audio (LibriSpeech sample or local recording)
2. Encode with Mimi to get full token tensor
3. Produce four reconstructions:
   - (a) Full reconstruction (all codebooks) — baseline
   - (b) Semantic-only (codebook 0, rest zeroed) — core hypothesis test
   - (c) Acoustic-only (codebook 0 zeroed, rest kept) — what we discard
   - (d) First-half semantic + second-half acoustic — boundary analysis
4. Measure PESQ, STOI for each vs. original
5. Calculate bitrate for each configuration

**Success criteria:** Semantic-only reconstruction is intelligible (STOI > 0.5, words
recognizable by human listener).

**Output:** `results/01_*.wav`, comparison plot, metrics table.

---

### Experiment 02: Semantic-Only Reconstruction Quality Sweep

**Goal:** Map the quality/bandwidth tradeoff by varying the number of codebooks transmitted.

**Method:**
1. Reconstruct using codebooks 0 only, 0-1, 0-2, ..., up to all codebooks
2. Measure PESQ and STOI at each level
3. Plot quality vs. bitrate curve

**Success criteria:** Identify the "knee" — minimum codebooks for acceptable quality.

**Output:** Quality vs. bitrate curve, audio samples at each level.

---

### Experiment 03: Prosody & Emotion Analysis

**Goal:** Determine whether semantic tokens preserve prosody (pitch, rhythm, emphasis).

**Method:**
1. Process emotionally varied speech (questions, exclamations, whispers, fast speech)
2. Compare pitch contours (F0) between original and semantic-only reconstruction
3. Measure pitch correlation

**Success criteria:** Pitch correlation > 0.7 between original and semantic-only.

**Output:** Pitch contour plots, correlation metrics.

---

### Experiment 04: Speaker Embedding / Voice Cross-Combination

**Goal:** Test whether acoustic tokens carry speaker identity independently of content.

**Method:**
1. Encode Speaker A and Speaker B audio
2. Cross-combine: Speaker A semantic + Speaker B acoustic → decode
3. Measure speaker similarity scores for all combinations

**Success criteria:** Cross-combined audio matches Speaker B's voice identity
(similarity > 0.7 with Speaker B, < 0.5 with Speaker A).

**Output:** Speaker similarity matrix, cross-combined audio samples.

---

### Experiment 05: Bandwidth & Latency Measurement

**Goal:** Quantify practical performance characteristics.

**Method:**
1. Measure actual bitrates for semantic-only vs. full Mimi vs. Opus
2. Benchmark encoding and decoding latency
3. Simulate packet loss (5%, 10%, 20%) and measure quality degradation

**Success criteria:** End-to-end latency < 200ms on CPU, graceful degradation under packet loss.

**Output:** Bitrate comparison table, latency benchmarks, packet loss curves.

---

### Experiment 06: Latency Benchmark (Detailed)

**Goal:** Profile per-component latency for real-time feasibility.

**Method:**
1. Measure time for: audio capture → preprocessing → encoding → token extraction →
   transmission simulation → decoding → audio output
2. Test with varying audio chunk sizes (80ms, 160ms, 320ms)
3. Profile CPU vs. GPU (if available)

**Success criteria:** Total pipeline < 150ms per chunk for real-time viability.

**Output:** Per-component latency breakdown, chunk size comparison.

---

## Execution Order

1. **Phase 0:** Project setup, dependency installation, basic codec test
2. **Phase 1:** Experiment 01 (token separation) — CRITICAL PATH
3. **Phase 2:** Experiment 03 (prosody) — understanding what semantic tokens preserve
4. **Phase 3:** Experiment 04 (speaker embedding) — validating personalization
5. **Phase 4:** Experiments 05-06 (bandwidth/latency) — practical feasibility

## Decision Points

- After Exp 01: If semantic-only is unintelligible → try codebooks 0-1, 0-2 (Exp 02)
- After Exp 03: If prosody is lost → design prosody sideband (pitch + energy contour)
- After Exp 04: If cross-combination fails → acoustic tokens may not cleanly separate identity
- After Exp 05: If latency > 300ms → optimize with ONNX or smaller model variants
