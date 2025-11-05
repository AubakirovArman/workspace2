#!/usr/bin/env python3
"""Encode dumped inference frames into a video without audio."""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import sys

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
MODERN_ROOT = PROJECT_ROOT / "modern-lipsync"
for candidate in (PROJECT_ROOT, MODERN_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

import numpy as np

from app_core.services.segment_lipsync import _encode_frames_sequence


def _json_dump(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _load_metadata(dump_dir: Path) -> Dict[str, Any]:
    meta_path = dump_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.json not found in {dump_dir}")
    with meta_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Encode frames produced by dynamic_inference_dump.py")
    parser.add_argument("dump_dir", help="Path to the inference dump directory")
    parser.add_argument("--output", default=None, help="Output path for the encoded video without audio")
    parser.add_argument("--fps", type=float, default=None, help="Override FPS for encoding (defaults to dump metadata)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dump_dir = Path(args.dump_dir).expanduser().resolve()
    if not dump_dir.exists():
        raise FileNotFoundError(f"Dump directory not found: {dump_dir}")

    metadata = _load_metadata(dump_dir)
    frames_path = dump_dir / "frames.npz"
    if not frames_path.exists():
        raise FileNotFoundError(f"Frames archive not found: {frames_path}")

    print(f"📥 Loading frames from {frames_path}...")
    with np.load(frames_path, allow_pickle=False) as data:
        if "frames" not in data:
            raise KeyError("frames key missing in frames.npz")
        frames_array = data["frames"]

    if frames_array.ndim != 4:
        raise ValueError(f"Unexpected frame array shape: {frames_array.shape}")
    frame_count = frames_array.shape[0]
    fps = float(args.fps if args.fps else metadata.get("fps", 25.0))

    output_path = Path(args.output).expanduser().resolve() if args.output else dump_dir / "video_no_audio.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frames: list[np.ndarray] = [frames_array[i] for i in range(frame_count)]

    print(f"🎥 Encoding {frame_count} frames @ {fps:.3f} fps → {output_path}")
    encode_start = time.perf_counter()
    return_code, encode_time, stderr_output = _encode_frames_sequence(frames, fps, output_path)
    encode_wall = time.perf_counter() - encode_start

    if return_code != 0:
        raise RuntimeError(f"FFmpeg returned {return_code}: {stderr_output}")

    print(f"✅ Encoding complete in {encode_wall:.2f}s (reported {encode_time:.2f}s)")

    metadata["encoding"] = {
        "output_path": str(output_path),
        "frame_count": frame_count,
        "fps": fps,
        "return_code": return_code,
        "encode_time_reported": encode_time,
        "encode_time_wall": encode_wall,
        "ffmpeg_stderr": stderr_output,
        "completed_at_utc": datetime.utcnow().isoformat() + "Z",
    }

    _json_dump(dump_dir / "metadata.json", metadata)
    print(f"📝 Updated metadata: {dump_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
