#!/usr/bin/env python3
"""Experiment: split frames across two FFmpeg processes to benchmark throughput."""
from __future__ import annotations

import argparse
import os
import sys
import time
import json
import subprocess
import statistics
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Tuple
import tempfile
import threading
import numpy as np
import torch
import torchaudio
import fcntl

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODERN_LIPSYNC_DIR = ROOT / "modern-lipsync"
if str(MODERN_LIPSYNC_DIR) not in sys.path:
    sys.path.insert(0, str(MODERN_LIPSYNC_DIR))

from service import LipsyncService  # type: ignore


@dataclass
class SegmentResult:
    segment_index: int
    frame_start: int
    frame_end: int
    frame_count: int
    write_time: float
    ffmpeg_time: float
    return_code: int
    stderr: str

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["stderr"] = self.stderr.strip()
        return payload


def _validate_inputs(face: Path, audio: Path, checkpoint: Path) -> None:
    if not face.exists():
        raise FileNotFoundError(f"Face source not found: {face}")
    if not audio.exists():
        raise FileNotFoundError(f"Audio file not found: {audio}")
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")


def _capture_frames(
    service: LipsyncService,
    face_path: str,
    waveform: torch.Tensor,
    sample_rate: int,
    fps: float,
    pads: Tuple[int, int, int, int],
    batch_size: int,
) -> Tuple[List[np.ndarray], float]:
    frames: List[np.ndarray] = []

    def _sink(frame: np.ndarray) -> None:
        frames.append(frame.copy())

    stats = service.process(
        face_path=face_path,
        audio_path="",
        output_path=None,
        static=True,
        fps=fps,
        pads=pads,
        audio_waveform=waveform,
        audio_sample_rate=sample_rate,
        frame_sink=_sink,
        batch_size_override=batch_size,
    )
    duration = stats.get("total_time", 0.0)
    return frames, duration


def _build_ffmpeg_command(
    codec: str,
    codec_opts: List[str],
    width: int,
    height: int,
    fps: float,
    output_path: Path,
    audio_path: Optional[Path],
) -> List[str]:
    dimension = f"{width}x{height}"
    blocksize = width * height * 3
    vf_filters: List[str] = []
    if width % 2 != 0 or height % 2 != 0:
        vf_filters.append('scale=w=trunc(iw/2)*2:h=trunc(ih/2)*2:flags=fast_bilinear')
    command: List[str] = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        dimension,
        "-r",
        str(fps),
        "-blocksize",
        str(blocksize),
        "-i",
        "pipe:0",
    ]
    if audio_path is not None:
        command += [
            "-i",
            str(audio_path),
        ]

    if vf_filters:
        command += ["-vf", ",".join(vf_filters)]

    command += ["-c:v", codec]
    command += codec_opts
    command += [
        "-pix_fmt",
        "yuv420p",
    ]
    if audio_path is not None:
        command += [
            "-c:a",
            "aac",
            "-ar",
            "16000",
            "-b:a",
            "128k",
            "-shortest",
        ]
    else:
        command += ["-an"]
    command.append(str(output_path))
    return command


def _allocate_pipe(proc: subprocess.Popen, blocksize: int) -> None:
    if proc.stdin is None:
        return
    try:
        target = max(blocksize * 4, 64 * 1024 * 1024)
        fcntl.fcntl(proc.stdin.fileno(), fcntl.F_SETPIPE_SZ, target)
    except (OSError, AttributeError):
        pass


def _write_segment(proc: subprocess.Popen, frames: List[np.ndarray]) -> float:
    assert proc.stdin is not None
    start = time.perf_counter()
    for frame in frames:
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8, copy=False)
        if not frame.flags.c_contiguous:
            frame = np.ascontiguousarray(frame)
        proc.stdin.write(memoryview(frame))
    proc.stdin.close()
    return time.perf_counter() - start


def _partition_frames(total: int, parts: int) -> List[Tuple[int, int]]:
    if parts <= 0:
        raise ValueError("segments must be positive")
    base = total // parts
    remainder = total % parts
    ranges: List[Tuple[int, int]] = []
    start = 0
    for idx in range(parts):
        extra = 1 if idx < remainder else 0
        end = start + base + extra
        ranges.append((start, end))
        start = end
    return ranges


def _run_dual_ffmpeg(
    frames: List[np.ndarray],
    fps: float,
    codec: str,
    codec_opts: List[str],
    output_dir: Path,
    audio_path: Optional[Path],
    segments_count: int,
) -> Tuple[List[SegmentResult], float, List[Path]]:
    total_frames = len(frames)
    if total_frames == 0:
        raise RuntimeError("No frames captured for encoding")
    height, width = frames[0].shape[:2]
    slice_ranges = _partition_frames(total_frames, segments_count)
    segments: List[Tuple[int, List[np.ndarray]]] = []
    for start, end in slice_ranges:
        if start >= end:
            continue
        segments.append((start, frames[start:end]))

    segment_results: List[SegmentResult] = []

    start_global = time.perf_counter()
    threads: List[threading.Thread] = []

    segment_paths: List[Path] = []

    for idx, (offset, chunk) in enumerate(segments):
        output_path = output_dir / f"segment_{idx:02d}.mp4"
        segment_paths.append(output_path)
        cmd = _build_ffmpeg_command(codec, codec_opts, width, height, fps, output_path, audio_path)
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        blocksize = width * height * 3
        _allocate_pipe(proc, blocksize)

        def _run(proc: subprocess.Popen, chunk: List[np.ndarray], idx: int, offset: int) -> None:
            write_time = _write_segment(proc, chunk)
            encode_start = time.perf_counter()
            retcode = proc.wait()
            encode_time = time.perf_counter() - encode_start
            stderr = proc.stderr.read().decode("utf-8", errors="ignore") if proc.stderr else ""
            result = SegmentResult(
                segment_index=idx,
                frame_start=offset,
                frame_end=offset + len(chunk) - 1,
                frame_count=len(chunk),
                write_time=write_time,
                ffmpeg_time=encode_time,
                return_code=retcode,
                stderr=stderr,
            )
            segment_results.append(result)

        thread = threading.Thread(target=_run, args=(proc, chunk, idx, offset), daemon=True)
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    total_time = time.perf_counter() - start_global

    segment_results.sort(key=lambda r: r.segment_index)
    return segment_results, total_time, segment_paths


def _concat_segments(segment_paths: List[Path], output_path: Path) -> Tuple[int, str]:
    if not segment_paths:
        raise ValueError("No segments to concatenate")
    concat_list = output_path.parent / "segments.txt"
    with open(concat_list, "w", encoding="utf-8") as f:
        for path in segment_paths:
            f.write(f"file '{path.name}'\n")

    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_list),
        "-c",
        "copy",
        str(output_path),
    ]
    proc = subprocess.run(command, capture_output=True, text=True)
    try:
        concat_list.unlink()
    except OSError:
        pass
    return proc.returncode, proc.stderr


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark dual FFmpeg processes for Wav2Lip output")
    parser.add_argument("--audio", type=str, default="/home/arman/workspace2/temp_web/audio_20251031_161642.wav",
                        help="Audio WAV file for inference")
    parser.add_argument("--face", type=str, default=str(ROOT / "avatar.jpg"),
                        help="Face image/video for inference")
    parser.add_argument("--checkpoint", type=str, default=str(ROOT / "Wav2Lip-SD-GAN.pt"),
                        help="Wav2Lip checkpoint path")
    parser.add_argument("--output-dir", type=str, default="outputs/dual_ffmpeg",
                        help="Directory to store encoded segments")
    parser.add_argument("--batch-size", type=int, default=512,
                        help="Batch size override for inference")
    parser.add_argument("--segments", type=int, default=2,
                        help="Number of parallel FFmpeg processes (segments)")
    parser.add_argument("--fps", type=float, default=25.0,
                        help="Frame rate for preprocessing")
    parser.add_argument("--pads", type=int, nargs=4, default=(0, 50, 0, 0),
                        help="Padding tuple for face detection")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU index to use")
    parser.add_argument("--include-audio", action="store_true",
                        help="Attach audio stream to each segment (duplicated)")
    parser.add_argument("--json", type=str, default=None,
                        help="Optional path to store JSON report")
    parser.add_argument("--output",
                        type=str,
                        default=None,
                        help="Optional final merged video path (concatenated segments)")

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    audio_path = Path(args.audio)
    face_path = Path(args.face)
    checkpoint_path = Path(args.checkpoint)
    _validate_inputs(face_path, audio_path, checkpoint_path)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    waveform, sample_rate = torchaudio.load(str(audio_path))
    if waveform.dim() == 2 and waveform.size(0) > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    service = LipsyncService(
        checkpoint_path=str(checkpoint_path),
        device="cuda",
        face_det_batch_size=16,
        wav2lip_batch_size=max(args.batch_size, 1),
        segmentation_path=None,
        sr_path=None,
        modules_root=None,
        realesrgan_path=None,
        realesrgan_outscale=1.0,
        use_fp16=True,
        use_compile=True,
    )

    try:
        service.preload_static_face(
            face_path=str(face_path),
            fps=args.fps,
            pads=tuple(args.pads),
            resize_factor=1,
            crop=(0, -1, 0, -1),
            rotate=False,
            nosmooth=False,
        )
    except Exception as preload_err:
        print(f"⚠️  Static face preload failed: {preload_err}")

    print("Capturing frames from LipsyncService...")
    frames, inference_time = _capture_frames(
        service,
        str(face_path),
        waveform,
        sample_rate,
        args.fps,
        tuple(args.pads),
        max(args.batch_size, 1),
    )
    print(f"Captured {len(frames)} frames in {inference_time:.2f}s")

    codec_name, codec_opts = service._select_ffmpeg_codec()

    if args.include_audio:
        temp_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_audio_path = Path(temp_audio.name)
        temp_audio.close()
        torchaudio.save(str(temp_audio_path), waveform, sample_rate)
    else:
        temp_audio_path = None

    print("Running dual FFmpeg processes...")
    segment_results, total_encode_time, segment_paths = _run_dual_ffmpeg(
        frames,
        args.fps,
        codec_name,
        codec_opts,
        output_dir,
        temp_audio_path,
        max(1, args.segments),
    )

    print(f"Total encoding time (two processes): {total_encode_time:.2f}s")
    for segment in segment_results:
        print(
            f" Segment {segment.segment_index}: frames {segment.frame_start}-{segment.frame_end} "
            f"({segment.frame_count} total) write={segment.write_time:.2f}s rc={segment.return_code}"
        )
        if segment.stderr:
            print(f"   stderr: {segment.stderr}")

    concat_returncode = None
    concat_stderr = ""
    merged_path: Optional[Path] = None
    if args.output:
        merged_path = Path(args.output)
        merged_path.parent.mkdir(parents=True, exist_ok=True)
        concat_returncode, concat_stderr = _concat_segments(segment_paths, merged_path)
        if concat_returncode == 0:
            print(f"Merged video saved to {merged_path}")
        else:
            print(f"⚠️  Failed to merge segments: {concat_stderr}")

    if args.json:
        payload = {
            "total_frames": len(frames),
            "inference_time": inference_time,
            "encode_time": total_encode_time,
            "segments": [s.to_dict() for s in segment_results],
            "codec": codec_name,
            "codec_opts": codec_opts,
            "segment_files": [str(path) for path in segment_paths],
        }
        if merged_path is not None:
            payload["merged_video"] = str(merged_path)
            payload["merge_status"] = concat_returncode
            payload["merge_stderr"] = concat_stderr
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"Report saved to {args.json}")

    if temp_audio_path is not None:
        try:
            temp_audio_path.unlink()
        except OSError:
            pass


if __name__ == "__main__":
    main()
