# Clarity Wire Format v0

Payload format for transmitting a Mimi-token voice note. Designed to be an
opaque blob to the transport (websocket relay, push payload, or datagram),
and to leave room for the v1 features (codebook-range chunks for progressive
enhancement, encryption).

## Layout

All integers little-endian.

| Offset | Size | Field | Notes |
|---|---|---|---|
| 0 | 1 | `version` | `0x00` for v0 |
| 1 | 1 | `n_codebooks` | codebooks included (1–32) |
| 2 | 1 | `frame_rate` | Hz, fixed `12` (12.5 Hz truncated; readers use 12.5) |
| 3 | 1 | reserved | `0x00` |
| 4 | 4 | `num_frames` | uint32 frame count |
| 8 | ... | `tokens` | uint16 per token, codebook-major: all frames of codebook 0, then codebook 1, ... |

Total size: `8 + 2 * n_codebooks * num_frames` bytes.

## Size examples (8 codebooks, 12.5 Hz)

| Duration | Frames | Payload |
|---|---|---|
| 5 s | 63 | ~1.0 KB |
| 30 s | 375 | ~5.9 KB |
| 2 min | 1500 | ~23.4 KB |

At 32 codebooks a 30 s note is ~23.4 KB — still ~4% of the same note in
Opus at 24 kbps (~90 KB).

## Deliberate v0 simplifications

- **uint16 per token, not 11-bit packing.** Tokens fit in 11 bits
  (codebook size 2048); packing would save 31%. Skipped for debuggability;
  revisit in v1 if payload size ever matters.
- **No per-frame packetization.** Voice notes ship as one blob. Real-time
  calls will need a framed variant (sequence numbers, redundancy) — see the
  experiment 05 redundancy arms for the design direction.
- **No encryption.** The payload is opaque to the transport, so E2E
  encryption can wrap it without format changes. The version byte gates any
  future breaking change.

## v1 sketch (progressive enhancement)

v1 will add codebook-range chunks: `{start_codebook, end_codebook}` headers
so a sender can ship codebooks 0–7 first and backfill 8–31 opportunistically.
The receiver merges chunks by codebook index and re-decodes — stored notes
upgrade in place. This exploits the residual (RVQ) structure: higher
codebooks strictly refine lower ones.
