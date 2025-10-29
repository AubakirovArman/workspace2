#!/bin/bash
# Script to copy model weights to checkpoints directory

echo "📦 Copying model weights..."

# Create checkpoints directory
mkdir -p checkpoints

# Copy weights from workspace root
if [ -f "/workspace/Wav2Lip-SD-GAN.pt" ]; then
    echo "Copying Wav2Lip-SD-GAN.pt..."
    cp /workspace/Wav2Lip-SD-GAN.pt checkpoints/
    echo "✓ Wav2Lip-SD-GAN.pt copied"
else
    echo "⚠ Wav2Lip-SD-GAN.pt not found in /workspace/"
fi

if [ -f "/workspace/Wav2Lip-SD-NOGAN.pt" ]; then
    echo "Copying Wav2Lip-SD-NOGAN.pt..."
    cp /workspace/Wav2Lip-SD-NOGAN.pt checkpoints/
    echo "✓ Wav2Lip-SD-NOGAN.pt copied"
else
    echo "⚠ Wav2Lip-SD-NOGAN.pt not found in /workspace/"
fi

# Copy s3fd detector
if [ -f "/workspace/camenduru-Wav2Lip/face_detection/detection/sfd/s3fd.pth" ]; then
    echo "Face detector already present from camenduru repo"
else
    echo "⚠ Face detector not found, please download from camenduru/Wav2Lip"
fi

echo ""
echo "✅ Weight setup complete!"
