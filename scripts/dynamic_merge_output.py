#!/usr/bin/env python3
"""Attach audio to an encoded video produced by dynamic_encode_from_dump.py."""
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
for candidate in (PROJECT_ROOT, PROJECT_ROOT / "modern-lipsync"):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from app_core.services.segment_lipsync import _attach_audio


def _load_metadata(dump_dir: Path) -> Dict[str, Any]:
    meta_path = dump_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.json not found in {dump_dir}")
    with meta_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _save_metadata(dump_dir: Path, metadata: Dict[str, Any]) -> None:
    meta_path = dump_dir / "metadata.json"
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge encoded video with audio track")
    parser.add_argument("dump_dir", help="Path to the inference dump directory")
    parser.add_argument("--encoded", default=None, help="Path to the silent video (defaults to metadata)")
    parser.add_argument("--audio", default=None, help="Audio track to mux (defaults to dump copy)")
    parser.add_argument("--output", default=None, help="Final video output path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dump_dir = Path(args.dump_dir).expanduser().resolve()
    if not dump_dir.exists():
        raise FileNotFoundError(f"Dump directory not found: {dump_dir}")

    metadata = _load_metadata(dump_dir)

    encoded_path = Path(args.encoded).expanduser().resolve() if args.encoded else None
    if encoded_path is None:
        encoded_meta = metadata.get("encoding", {})
        encoded_path = Path(encoded_meta.get("output_path", dump_dir / "video_no_audio.mp4")).expanduser().resolve()
    if not encoded_path.exists():
        raise FileNotFoundError(f"Encoded video not found: {encoded_path}")

    audio_path = Path(args.audio).expanduser().resolve() if args.audio else None
    if audio_path is None:
        candidate = metadata.get("audio_dump_path") or metadata.get("audio_path")
        if not candidate:
            raise RuntimeError("Audio path not recorded in metadata; provide --audio")
        audio_path = Path(candidate).expanduser().resolve()
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    output_path = Path(args.output).expanduser().resolve() if args.output else dump_dir / "video_final.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"🔗 Muxing video {encoded_path} with audio {audio_path}\n   → {output_path}")
    merge_start = time.perf_counter()
    return_code, stderr_output = _attach_audio(encoded_path, audio_path, output_path)
    merge_wall = time.perf_counter() - merge_start

    if return_code != 0:
        raise RuntimeError(f"FFmpeg returned {return_code}: {stderr_output}")

    print(f"✅ Merge complete in {merge_wall:.2f}s")

    metadata["merge"] = {
        "output_path": str(output_path),
        "audio_path": str(audio_path),
        "encoded_input": str(encoded_path),
        "return_code": return_code,
        "ffmpeg_stderr": stderr_output,
        "merge_time_wall": merge_wall,
        "completed_at_utc": datetime.utcnow().isoformat() + "Z",
    }

    _save_metadata(dump_dir, metadata)
    print(f"📝 Updated metadata: {dump_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
