"""Application-wide configuration values."""
from __future__ import annotations

import os
from pathlib import Path


def _env_flag(name: str, default: str = "0") -> bool:
    value = os.getenv(name)
    if value is None:
        value = default
    return value.strip().lower() in {"1", "true", "yes", "on"}

BASE_DIR = Path(__file__).resolve().parent.parent
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
AVATAR_IMAGE = os.getenv(
    "AVATAR_PATH",
    str(BASE_DIR / "avatar.jpg")
)
AVATAR_STATIC_MODE = _env_flag("AVATAR_STATIC_MODE", "1")
try:
    AVATAR_FPS = float(os.getenv("AVATAR_FPS", "25.0"))
except ValueError:
    AVATAR_FPS = 25.0
# Primary realtime model (GAN version with teeth)
CHECKPOINT_PATH_GAN = os.getenv(
    "GAN_CHECKPOINT_PATH",
    str(BASE_DIR / "Wav2Lip-SD-GAN.pt")
)

# Secondary realtime model (NoGAN version, softer mouth)
CHECKPOINT_PATH_NOGAN = os.getenv(
    "NOGAN_CHECKPOINT_PATH",
    str(BASE_DIR / "Wav2Lip-SD-NOGAN.pt")
)

SEGMENTATION_PATH_HD = "/home/arman/musetalk/avatar/checkpoints/face_segmentation.pth"
SR_PATH_HD = "/home/arman/musetalk/avatar/checkpoints/esrgan_yunying.pth"
HD_MODULES_ROOT = Path("/workspace/Wav2Lip-HD")
REALESRGAN_PATH = "/home/arman/musetalk/avatar/checkpoints/RealESRGAN_x4plus.pth"

ENABLE_SEGMENTATION = _env_flag("ENABLE_SEGMENTATION", "1")
ENABLE_SUPER_RESOLUTION = _env_flag("ENABLE_SUPER_RESOLUTION", "1")
ENABLE_REALESRGAN = _env_flag("ENABLE_REALESRGAN", "0")

try:
    REALESRGAN_OUTSCALE = float(os.getenv("REALESRGAN_OUTSCALE", "4.0"))
except ValueError:
    REALESRGAN_OUTSCALE = 4.0

TTS_API_URL = os.getenv("TTS_API_URL", "https://tts.sk-ai.kz/api/tts")
OUTPUT_DIR_PATH = BASE_DIR / "outputs"
TEMP_DIR_PATH = BASE_DIR / "temp_web"
OUTPUT_DIR_PATH.mkdir(exist_ok=True)
TEMP_DIR_PATH.mkdir(exist_ok=True)
OUTPUT_DIR = str(OUTPUT_DIR_PATH)
TEMP_DIR = str(TEMP_DIR_PATH)
AVATAR_PREVIEW_PATH = str(TEMP_DIR_PATH / "avatar_preview.jpg")

HOST = os.getenv("LIPSYNC_HOST", "0.0.0.0")
PORT = int(os.getenv("LIPSYNC_PORT", "3000"))
DEBUG = bool(int(os.getenv("LIPSYNC_DEBUG", "0")))


try:
    _gan_max_env = os.getenv("GAN_MAX_MODELS", "4")
    MAX_GAN_MODELS = int(_gan_max_env)
except (TypeError, ValueError):
    MAX_GAN_MODELS = 4

if MAX_GAN_MODELS < 1:
    MAX_GAN_MODELS = 1
elif MAX_GAN_MODELS > 8:
    MAX_GAN_MODELS = 8

try:
    _gan_instances_env = os.getenv("GAN_MODEL_INSTANCES", "1")
    GAN_MODEL_INSTANCES = int(_gan_instances_env)
except (TypeError, ValueError):
    GAN_MODEL_INSTANCES = 1

if GAN_MODEL_INSTANCES < 1:
    GAN_MODEL_INSTANCES = 1
elif GAN_MODEL_INSTANCES > MAX_GAN_MODELS:
    GAN_MODEL_INSTANCES = MAX_GAN_MODELS
