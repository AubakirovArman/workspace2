# Modern Lipsync

Modern implementation of Wav2Lip with PyTorch 2.8.0+ support

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run inference
python inference.py \
  --checkpoint_path /workspace/Wav2Lip-SD-GAN.pt \
  --face input_video.mp4 \
  --audio input_audio.wav
```

See [README.md](README.md) for full documentation.
