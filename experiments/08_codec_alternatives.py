"""Experiment 08: Codec Alternatives Benchmark — STUB (deliberately deferred).

Candidate benchmark arms against Mimi, from the July 2026 research pass:
  - FocalCodec-Stream (Apache 2.0, 0.55-0.8 kbps, streaming, ~80ms latency)
    https://arxiv.org/abs/2509.16195
  - NVIDIA NeMo NanoCodec 1.78 kbps @ 12.5 fps (NVIDIA Open Model License)
    https://huggingface.co/nvidia/nemo-nano-codec-22khz-1.78kbps-12.5fps

Why deferred: neither has an iPhone deployment path (Mimi has moshi-swift and
rustymimi, plus Meta's T-Mimi on-device benchmarks), and Mimi meets the
Phase-1 quality gate on clean speech. Integration cost ~2-3 days for
information that doesn't change any Phase 2 decision.

Revisit triggers (implement this experiment when either fires):
  1. The knee lands above ~2.2 kbps (16 codebooks) with unsatisfying quality
     below it — then FocalCodec-Stream's sub-1kbps quality claims matter.
  2. Tier-C (floor tier) design begins and needs a sub-600bps codec arm to
     compare against text+prosody+cloned-voice reconstruction (see
     experiments/09, Phase 3).

Design when implemented: reuse the experiment 01 sweep harness — same samples,
same PESQ/STOI/speaker-similarity metrics, persisted via results_io — with one
arm per codec at its native operating points.
"""

if __name__ == "__main__":
    raise SystemExit(__doc__)
