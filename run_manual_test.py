#!/usr/bin/env python3
"""Helper script to run the manual lipsync test used in the REPL snippet."""
import os
import sys
import subprocess

import torchaudio

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODERN_LIPSYNC_PATH = os.path.join(PROJECT_ROOT, "modern-lipsync")

if MODERN_LIPSYNC_PATH not in sys.path:
    sys.path.insert(0, MODERN_LIPSYNC_PATH)

from app_core.services.lipsync_initializer import init_lipsync_service  # noqa: E402
from app_core.config import AVATAR_IMAGE, OUTPUT_DIR, TEMP_DIR  # noqa: E402

AUDIO_MP3 = "/workspace/audio10.mp3"
TEMP_WAV = os.path.join(TEMP_DIR, "manual_test.wav")
OUTPUT_VIDEO = os.path.join(OUTPUT_DIR, "manual_test.mp4")


def main() -> None:
    os.makedirs(TEMP_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            AUDIO_MP3,
            "-ar",
            "16000",
            "-ac",
            "1",
            TEMP_WAV,
            "-loglevel",
            "error",
        ],
        check=True,
    )

    waveform, sample_rate = torchaudio.load(TEMP_WAV)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    print("Initializing lipsync services...")
    gan_service, nogan_service, _ = init_lipsync_service()
    service = gan_service or nogan_service
    if service is None:
        raise RuntimeError("No lipsync service loaded")

    print("Processing...")
    stats = service.process(
        face_path=AVATAR_IMAGE,
        audio_path=TEMP_WAV,
        output_path=OUTPUT_VIDEO,
        static=True,
        pads=(0, 50, 0, 0),
        fps=30.0,
        audio_waveform=waveform,
        audio_sample_rate=sample_rate,
    )

    print("Done! Stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print(f"Video saved to: {OUTPUT_VIDEO}")


if __name__ == "__main__":
    main()
