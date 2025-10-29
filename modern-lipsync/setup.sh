#!/bin/bash
# Quick setup script for Modern Lipsync

echo "🚀 Modern Lipsync - Quick Setup"
echo "================================"

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $python_version"

# Check CUDA
if command -v nvidia-smi &> /dev/null; then
    echo "✓ CUDA available"
    nvidia-smi --query-gpu=name --format=csv,noheader
else
    echo "⚠ CUDA not found, will use CPU"
fi

# Check ffmpeg
if command -v ffmpeg &> /dev/null; then
    echo "✓ ffmpeg installed"
else
    echo "✗ ffmpeg not found! Please install:"
    echo "  Ubuntu/Debian: sudo apt-get install ffmpeg"
    echo "  macOS: brew install ffmpeg"
    exit 1
fi

# Install dependencies
echo ""
echo "📦 Installing dependencies..."
pip install -q -r requirements.txt

echo ""
echo "✅ Setup complete!"
echo ""
echo "📝 Next steps:"
echo "1. Place model weights in /workspace/"
echo "   - Wav2Lip-SD-GAN.pt"
echo "   - Wav2Lip-SD-NOGAN.pt"
echo ""
echo "2. Run inference:"
echo "   python inference.py --checkpoint_path /workspace/Wav2Lip-SD-GAN.pt --face video.mp4 --audio audio.wav"
echo ""
