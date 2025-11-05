#!/usr/bin/env python3
"""Run dynamic avatar inference and dump frames for offline experiments."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import sys

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
MODERN_ROOT = PROJECT_ROOT / "modern-lipsync"
for candidate in (PROJECT_ROOT, MODERN_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

import numpy as np
import torch
import torchaudio
import cv2

from service import LipsyncService
from app_core import config

PADS = (0, 50, 0, 0)


def _select_segmentation() -> str | None:
    if not config.ENABLE_SEGMENTATION:
        return None
    path = config.SEGMENTATION_PATH_HD
    return path if path and os.path.exists(path) else None


def _select_super_resolution() -> str | None:
    if not config.ENABLE_SUPER_RESOLUTION:
        return None
    path = config.SR_PATH_HD
    return path if path and os.path.exists(path) else None


def _select_realesrgan() -> str | None:
    if not config.ENABLE_REALESRGAN:
        return None
    path = config.REALESRGAN_PATH
    return path if path and os.path.exists(path) else None


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Unsupported type for JSON serialization: {type(value)!r}")


def _timestamp() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def _resolve_fps(avatar_path: Path, fallback: float) -> float:
    capture = cv2.VideoCapture(str(avatar_path))
    fps = capture.get(cv2.CAP_PROP_FPS)
    capture.release()
    if fps and fps > 0:
        return float(fps)
    return float(fallback)


def build_service(args: argparse.Namespace) -> tuple[LipsyncService, Dict[str, Any]]:
    segmentation_path = None if args.disable_segmentation else _select_segmentation()
    sr_path = None if args.disable_super_resolution else _select_super_resolution()
    realesrgan_path = None if args.disable_realesrgan else _select_realesrgan()

    service = LipsyncService(
        checkpoint_path=str(Path(args.checkpoint).expanduser()),
        device=args.device,
        face_det_batch_size=args.face_det_batch,
        wav2lip_batch_size=args.batch_size,
        segmentation_path=segmentation_path,
        sr_path=sr_path,
        modules_root=None,
        realesrgan_path=realesrgan_path,
        realesrgan_outscale=config.REALESRGAN_OUTSCALE,
        use_fp16=not args.disable_fp16,
        use_compile=not args.disable_compile,
        ffmpeg_threads=args.ffmpeg_threads,
        ffmpeg_filter_threads=args.ffmpeg_filter_threads,
    )
    service_info: Dict[str, Any] = {
        "segmentation_path": segmentation_path,
        "segmentation_enabled": bool(segmentation_path),
        "super_resolution_path": sr_path,
        "super_resolution_enabled": bool(sr_path),
        "realesrgan_path": realesrgan_path,
        "realesrgan_enabled": bool(realesrgan_path),
        "use_fp16": not args.disable_fp16,
        "use_compile": not args.disable_compile,
        "ffmpeg_threads": args.ffmpeg_threads,
        "ffmpeg_filter_threads": args.ffmpeg_filter_threads,
    }
    return service, service_info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dump inference frames for dynamic avatar experiments.")
    parser.add_argument("--avatar", default="/home/arman/musetalk/avatar/IMG_3899.MOV", help="Path to the avatar video source.")
    parser.add_argument("--audio", default="/home/arman/musetalk/avatar/audio_20251030_145910.wav", help="Path to the source audio file.")
    parser.add_argument("--checkpoint", default=config.CHECKPOINT_PATH_GAN, help="Path to the Wav2Lip checkpoint.")
    parser.add_argument("--device", default="cuda:0", help="Torch device for inference.")
    parser.add_argument("--batch-size", type=int, default=16, help="Override Wav2Lip batch size.")
    parser.add_argument("--face-det-batch", type=int, default=16, help="Face detector batch size.")
    parser.add_argument("--fps", type=float, default=config.AVATAR_FPS, help="Fallback FPS if the source video does not report it.")
    parser.add_argument("--output-dir", default=None, help="Directory to place the dump. Created if missing.")
    parser.add_argument("--ffmpeg-threads", type=int, default=16, help="Value for ffmpeg threads during service init.")
    parser.add_argument("--ffmpeg-filter-threads", type=int, default=0, help="Value for ffmpeg filter_threads during service init.")
    parser.add_argument("--disable-segmentation", action="store_true", help="Do not load the segmentation model even if available.")
    parser.add_argument("--disable-super-resolution", action="store_true", help="Do not load ESRGAN.")
    parser.add_argument("--disable-realesrgan", action="store_true", help="Do not load RealESRGAN.")
    parser.add_argument("--disable-fp16", action="store_true", help="Force FP32 inference.")
    parser.add_argument("--disable-compile", action="store_true", help="Disable torch.compile optimizations.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    avatar_path = Path(args.avatar).expanduser().resolve()
    audio_path = Path(args.audio).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()

    if not avatar_path.exists():
        raise FileNotFoundError(f"Avatar source not found: {avatar_path}")
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio source not found: {audio_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    dump_root = Path(args.output_dir).expanduser().resolve() if args.output_dir else Path(config.TEMP_DIR) / "dynamic_dumps" / _timestamp()
    dump_root.mkdir(parents=True, exist_ok=True)

    print(f"🗄️  Dump directory: {dump_root}")

    fps = _resolve_fps(avatar_path, args.fps)
    print(f"🎞️  Using FPS: {fps:.3f}")

    print("📦 Initializing GAN service...")
    service, service_details = build_service(args)

    print("🎬 Preloading video cache...")
    preload_start = time.perf_counter()
    service.preload_video_cache(face_path=str(avatar_path), fps=fps, pads=PADS)
    preload_time = time.perf_counter() - preload_start
    print(f"✅ Video cache ready in {preload_time:.2f}s")

    print("🔊 Loading audio...")
    waveform, sample_rate = torchaudio.load(str(audio_path))

    frames: List[np.ndarray] = []

    def _sink(frame: np.ndarray) -> None:
        frames.append(frame.copy())

    if service.is_cuda:
        torch.cuda.synchronize()
    inference_start = time.perf_counter()
    stats = service.process(
        face_path=str(avatar_path),
        audio_path="",
        output_path=None,
        static=False,
        fps=fps,
        pads=PADS,
        audio_waveform=waveform,
        audio_sample_rate=sample_rate,
        frame_sink=_sink,
        batch_size_override=args.batch_size,
        frame_offset=0,
    )
    if service.is_cuda:
        torch.cuda.synchronize()
    inference_wall = time.perf_counter() - inference_start

    frame_count = len(frames)
    if frame_count == 0:
        raise RuntimeError("Inference produced zero frames")
    frame_shape = frames[0].shape
    frame_array = np.asarray(frames, dtype=np.uint8)

    frames_path = dump_root / "frames.npz"
    print(f"💾 Saving frames to {frames_path} ({frame_array.nbytes / 1e6:.2f} MB raw)...")
    np.savez_compressed(frames_path, frames=frame_array)

    audio_dump_path = dump_root / audio_path.name
    if audio_dump_path.exists():
        audio_dump_path = dump_root / f"audio_{_timestamp()}.wav"
    shutil.copy2(audio_path, audio_dump_path)

    def _coerce(value: Any) -> Any:
        if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
            return float(value)
        return value

    metadata: Dict[str, Any] = {
        "dump_version": 1,
        "created_at_utc": datetime.utcnow().isoformat() + "Z",
        "dump_dir": str(dump_root),
        "avatar_path": str(avatar_path),
        "audio_path": str(audio_path),
        "audio_dump_path": str(audio_dump_path),
        "checkpoint_path": str(checkpoint_path),
        "device": args.device,
        "fps": fps,
        "frame_count": frame_count,
        "frame_shape": frame_shape,
        "sample_rate": sample_rate,
        "batch_size_override": args.batch_size,
        "face_det_batch": args.face_det_batch,
        "preload_time": preload_time,
        "inference_stats": {key: _coerce(value) for key, value in stats.items()},
        "inference_breakdown": {key: _coerce(val) for key, val in getattr(service, "_last_inference_breakdown", {}).items()},
        "inference_wall_time": inference_wall,
        "service_config": service_details,
    }

    metadata_path = dump_root / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2, default=_json_default)

    print(f"✅ Dump complete: {frame_count} frames @ {fps:.3f} fps")
    print(f"   Inference wall time: {inference_wall:.2f}s")
    print(f"   Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
