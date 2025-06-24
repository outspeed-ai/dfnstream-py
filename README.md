# DeepFilterNet Streaming

A Python library for real-time audio denoising using DeepFilterNet, providing streaming audio processing capabilities through C API bindings.

## Platform Support

Currently supports the following platforms:
- **macOS ARM64** (Apple Silicon)
- **Linux x86_64** 
- **Linux ARM64** (aarch64)

## Installation

```bash
pip install dfnstream-py
```

## Quick Start

```python
from dfnstream_py import DeepFilterNetStreaming
import numpy as np

# Initialize the processor
processor = DeepFilterNetStreaming()

# Process audio chunks
audio_chunk = np.random.randn(1024).astype(np.float32)  # Your audio data
denoised_chunk = processor.process_chunk(audio_chunk)

# Don't forget to cleanup
processor.close()
```

## Examples

The `examples/` folder contains sample scripts demonstrating different use cases:

### Single File Processing
```bash
uv run python examples/enhance_wav.py input_audio.wav
```
Processes a single WAV file and saves the denoised version as `input_audio_denoised.wav`.

### Batch Directory Processing
```bash
uv run python examples/enhance_dir.py /path/to/wav/files/
```
Processes all WAV files in a directory and saves enhanced versions to an `outputs/` folder.

## API Reference

### DeepFilterNetStreaming

The main class for audio processing.

```python
DeepFilterNetStreaming(
    model_path=None,          # Optional custom model path
    atten_lim=None,           # Attenuation limit in dB (None = no limit)
    log_level="warn",         # Logging level: "error", "warn", "info", "debug", "trace"
    compensate_delay=True,    # Enable delay compensation
    post_filter_beta=0.0      # Post-filter beta (0.0 = disabled)
)
```

#### Key Methods

- `process_chunk(audio_chunk)` - Process a chunk of audio data
- `process_frame(audio_frame)` - Process a single frame
- `finalize()` - Get remaining delayed samples
- `close()` - Clean up resources