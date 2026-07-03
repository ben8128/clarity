# Clarity Prototype — Experiment Specifications

## Hypothesis

Transmitting a tunable subset of Mimi's codebooks (1 to N) provides a continuous
quality/bandwidth tradeoff. Even at ALL codebooks, Mimi is 96%+ smaller than Opus.
The goal is to find the **knee of the curve** — the sweet spot where adding another
codebook stops meaningfully improving quality — enabling crystal-clear calls on the
worst networks.

## Experiments

### Experiment 01: Codebook Sweep

**Goal:** Find the quality/bandwidth knee by sweeping codebook count from 1 to N.

**Method:**
1. Load 3+ audio samples (local `audio/` dir, fallback to LibriSpeech)
2. Encode each with Mimi, sweep codebook counts 1 through N
3. For each count: reconstruct, measure PESQ, STOI, and speaker similarity
4. Average metrics across samples
5. Detect sweet spot via diminishing-returns analysis (`optimal_codebook_count()`)
6. Generate dual-axis plot (quality metrics vs bitrate with Opus reference shading)

**Success criteria:** Identify the knee — codebook count where marginal quality gain
drops below threshold. All codebook counts remain 90%+ smaller than Opus.

**Output:** `results/01_codebook_sweep_*.png`, `results/reconstruction_*_codebooks.wav`,
summary table with bitrate, PESQ, STOI, speaker similarity per codebook count.

---

### Experiment 03: Prosody & Emotion Analysis

**Goal:** Find the codebook count at which prosody (pitch, rhythm, emphasis) is preserved.

**Method:**
1. Process emotionally varied speech (questions, exclamations, whispers, fast speech)
2. Sweep codebook counts (1, 2, 4, 8, all) for each sample
3. Compare pitch contours (F0) between original and each reconstruction
4. Measure pitch correlation at each codebook count to find where prosody emerges

**Success criteria:** Identify the codebook count where pitch correlation exceeds 0.7.

**Output:** Pitch contour plots per codebook count, correlation vs codebook count curve.

---

### Experiment 04: Speaker Embedding / Voice Identity vs Codebook Count

**Goal:** Identify at which codebook count speaker identity becomes recognizable.

**Method:**
1. Encode Speaker A and Speaker B audio
2. Sweep codebook counts for each speaker, measure speaker similarity vs original
3. Cross-combine at the sweet-spot codebook count: Speaker A semantic + Speaker B acoustic
4. Measure speaker similarity scores for all combinations

**Success criteria:** Find the codebook count where speaker similarity exceeds 0.7.
Cross-combined audio at that count matches Speaker B's identity.

**Output:** Speaker similarity vs codebook count curve, cross-combined audio samples,
similarity matrix at sweet-spot codebook count.

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
2. **Phase 1:** Experiment 01 (codebook sweep) — CRITICAL PATH, finds the knee
3. **Phase 2:** Experiment 03 (prosody) — find codebook count where prosody is preserved
4. **Phase 3:** Experiment 04 (speaker identity) — find codebook count for voice recognition
5. **Phase 4:** Experiments 05-06 (bandwidth/latency) — practical feasibility

## Decision Points

- After Exp 01: Sweet spot identified → use that codebook count for all subsequent experiments
- After Exp 03: If prosody requires more codebooks than sweet spot → adjust target upward
- After Exp 04: If speaker identity requires more codebooks → adjust target upward
- After Exp 05: If latency > 300ms → optimize with ONNX or smaller model variants
