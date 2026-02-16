# Clarity Prototype

## Project Overview
Clarity is a prototype for a real-time voice communication app that transmits compressed
semantic voice tokens instead of raw audio, then reconstructs personalized voice on the
receiver's device. The goal is crystal-clear calls on poor networks.

This prototype validates the core technical hypothesis using the Kyutai Mimi neural audio
codec. We are NOT building a full app — we are running experiments to prove the architecture
works.

## Tech Stack
- Python 3.10+ (primary language for all experiments)
- PyTorch (Mimi model inference)
- HuggingFace Transformers (Mimi model loading)
- NumPy, SciPy (audio processing)
- Matplotlib (visualization of results)
- soundfile / librosa (audio I/O)
- pesq, pystoi (objective audio quality metrics)

## Key Dependencies
```
pip install torch torchaudio transformers datasets[audio] soundfile librosa
pip install matplotlib numpy scipy
pip install pesq pystoi  # audio quality metrics
```

## Project Structure
```
clarity-prototype/
├── CLAUDE.md              # This file
├── SPEC.md                # Detailed experiment specifications
├── README.md              # Project readme
├── requirements.txt       # Python dependencies
├── audio/                 # Test audio files (not committed to git)
│   ├── speaker1_*.wav
│   └── speaker2_*.wav
├── src/                   # Source code
│   ├── __init__.py
│   ├── codec.py           # Mimi codec wrapper (encode/decode/extract tokens)
│   ├── speaker.py         # Speaker embedding extraction and management
│   ├── quality.py         # Audio quality measurement (PESQ, STOI, similarity)
│   ├── pipeline.py        # End-to-end encode→transmit→decode pipeline
│   └── utils.py           # Audio I/O, resampling, visualization helpers
├── experiments/           # Experiment scripts (one per experiment)
│   ├── 01_token_separation.py
│   ├── 02_semantic_only_reconstruction.py
│   ├── 03_prosody_analysis.py
│   ├── 04_speaker_embedding.py
│   ├── 05_bandwidth_measurement.py
│   └── 06_latency_benchmark.py
├── notebooks/             # Jupyter notebooks for interactive exploration
├── results/               # Experiment outputs (plots, metrics, audio samples)
│   └── .gitkeep
└── tests/                 # Unit tests
    └── test_codec.py
```

## Commands
- `python -m pytest tests/` — Run tests
- `python experiments/01_token_separation.py` — Run experiment 1
- `pip install -r requirements.txt` — Install dependencies

## Architecture Notes

### Mimi Codec (from Kyutai)
- Model: `kyutai/mimi` on HuggingFace
- Input: 24kHz mono audio
- Output: Token tensor of shape (batch, num_codebooks, num_frames)
- Frame rate: 12.5 Hz (one frame every 80ms)
- Codebook structure:
  - Codebook 0 (index 0): SEMANTIC tokens — linguistic/phonetic content
  - Codebooks 1-7 (indices 1-7): ACOUSTIC tokens — timbre, prosody, voice characteristics
- Each codebook has 2,048 entries (11 bits per token)
- Full bitrate: 12.5 × 8 × 11 = 1,100 bps
- Semantic-only bitrate: 12.5 × 1 × 11 = 137.5 bps

### Loading the model
```python
from transformers import MimiModel, AutoFeatureExtractor
model = MimiModel.from_pretrained("kyutai/mimi")
feature_extractor = AutoFeatureExtractor.from_pretrained("kyutai/mimi")
```

### Encoding and decoding
```python
# Encode: audio → tokens
inputs = feature_extractor(raw_audio=audio, sampling_rate=24000, return_tensors="pt")
encoder_outputs = model.encode(inputs["input_values"])
audio_codes = encoder_outputs.audio_codes  # shape: (batch, num_codebooks, num_frames)

# Decode: tokens → audio
reconstructed = model.decode(audio_codes)
```

### CRITICAL: Token separation for Clarity
```python
# Extract semantic tokens only (codebook 0)
semantic_tokens = audio_codes[:, 0:1, :]  # shape: (batch, 1, num_frames)

# Extract acoustic tokens (codebooks 1-7)
acoustic_tokens = audio_codes[:, 1:, :]   # shape: (batch, 7, num_frames)

# For semantic-only reconstruction, zero out acoustic tokens
modified_codes = audio_codes.clone()
modified_codes[:, 1:, :] = 0  # zero acoustic codebooks
reconstructed = model.decode(modified_codes)
```

## Important Gotchas

- Mimi expects 24kHz mono audio. Always resample before encoding.
- The HuggingFace implementation may use 32 codebooks (not 8 as in the paper).
  Check `audio_codes.shape[1]` after encoding to confirm. If 32, semantic is still
  index 0, and acoustic is indices 1-31.
- Model downloads ~300MB on first run. This is cached by HuggingFace.
- PESQ requires 16kHz or 8kHz audio. Resample before computing.
- Always save intermediate audio outputs as .wav files in results/ for listening tests.

## Code Style

- Use type hints on all function signatures
- Write docstrings on all public functions
- Use pathlib.Path for file paths, not string concatenation
- Print progress for any operation that takes >5 seconds
- Save all experiment results (metrics + audio) to results/ with descriptive filenames
- Include timestamps in result filenames to avoid overwriting

## Git Workflow

- Commit after each working experiment
- Use descriptive commit messages: "Experiment 01: validate token separation"
- Don't commit audio files or model weights (add to .gitignore)
