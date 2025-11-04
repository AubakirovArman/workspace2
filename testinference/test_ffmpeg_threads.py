#!/usr/bin/env python3
"""Benchmark FFmpeg thread counts by measuring writing and encoding times."""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional

DEFAULT_GPU_ID = 7
os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(DEFAULT_GPU_ID))

import torch
import torchaudio

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODERN_LIPSYNC_DIR = ROOT / "modern-lipsync"
if str(MODERN_LIPSYNC_DIR) not in sys.path:
    sys.path.insert(0, str(MODERN_LIPSYNC_DIR))

from service import LipsyncService  # type: ignore


def _validate_gpu(requested_gpu: int) -> int:
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        visible = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        try:
            requested_gpu = int(visible[0])
        except (ValueError, IndexError):
            requested_gpu = 0
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available on this system")
    torch.cuda.init()
    device_count = torch.cuda.device_count()
    if device_count == 0:
        raise RuntimeError("No CUDA devices detected after initialization")
    if requested_gpu >= device_count:
        raise RuntimeError(
            f"Requested GPU index {requested_gpu} is out of range (visible devices: {device_count})"
        )
    torch.cuda.set_device(0)
    return requested_gpu


def _format_seconds(value: float) -> str:
    return f"{value:.2f}s"


def _parse_threads(values: Optional[Iterable[int]]) -> List[int]:
    if not values:
        return [0, 4, 8, 12, 16, 24, 32]
    parsed: List[int] = []
    for entry in values:
        parsed.append(int(entry))
    return sorted(set(parsed))


@dataclass
class ThreadBenchmark:
    threads: int
    attempts: int
    totals: List[float] = field(default_factory=list)
    writing_times: List[float] = field(default_factory=list)
    ffmpeg_times: List[float] = field(default_factory=list)
    inference_times: List[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload.update(
            {
                "best_total": min(self.totals) if self.totals else None,
                "median_total": statistics.median(self.totals) if self.totals else None,
                "best_writing_time": min(self.writing_times) if self.writing_times else None,
                "median_writing_time": statistics.median(self.writing_times) if self.writing_times else None,
            }
        )
        return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark FFmpeg -threads parameter for Wav2Lip pipeline")
    parser.add_argument("--audio", type=str, default="/home/arman/workspace2/temp_web/audio_20251031_161642.wav",
                        help="Path to the audio WAV file used for the benchmark")
    parser.add_argument("--face", type=str, default=str(ROOT / "avatar.jpg"),
                        help="Path to a face image/video for inference")
    parser.add_argument("--checkpoint", type=str, default=str(ROOT / "Wav2Lip-SD-GAN.pt"),
                        help="Path to the Wav2Lip checkpoint")
    parser.add_argument("--threads", type=int, nargs="*", default=None,
                        help="List of thread counts to test (use 0 for FFmpeg auto)")
    parser.add_argument("--repeat", type=int, default=2,
                        help="Number of attempts per thread value")
    parser.add_argument("--batch-size", type=int, default=512,
                        help="Batch size override for inference")
    parser.add_argument("--fps", type=float, default=25.0,
                        help="Frame rate for preprocessing/static cache")
    parser.add_argument("--pads", type=int, nargs=4, default=(0, 50, 0, 0),
                        help="Face padding tuple applied during preprocessing")
    parser.add_argument("--gpu", type=int, default=DEFAULT_GPU_ID,
                        help="Physical GPU id to use (mapped via CUDA_VISIBLE_DEVICES)")
    parser.add_argument("--cooldown", type=float, default=0.0,
                        help="Pause between attempts to stabilize measurements (seconds)")
    parser.add_argument("--output", type=str, default=None,
                        help="Optional JSON file to store detailed results")
    parser.add_argument("--video-dir", type=str, default="outputs/ffmpeg_threads",
                        help="Directory to store rendered videos for each attempt")
    parser.add_argument("--warmup", action="store_true",
                        help="Run a warmup inference before benchmarking each thread value")

    args = parser.parse_args()

    thread_values = _parse_threads(args.threads)
    if not thread_values:
        raise ValueError("No thread values specified")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    requested_gpu = _validate_gpu(args.gpu)
    print(f"Using GPU id {requested_gpu} (CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']})")
    print(f"Thread values to evaluate: {thread_values}")

    audio_path = Path(args.audio)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    face_path = Path(args.face)
    if not face_path.exists():
        raise FileNotFoundError(f"Face image/video not found: {face_path}")
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    video_dir = Path(args.video_dir) if args.video_dir else None
    if video_dir is not None:
        video_dir.mkdir(parents=True, exist_ok=True)

    waveform, sample_rate = torchaudio.load(str(audio_path))
    if waveform.dim() == 2 and waveform.size(0) > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    print(f"Loaded audio waveform {waveform.shape} at {sample_rate} Hz")

    service = LipsyncService(
        checkpoint_path=str(checkpoint_path),
        device='cuda',
        face_det_batch_size=16,
        wav2lip_batch_size=max(args.batch_size, 1),
        segmentation_path=None,
        sr_path=None,
        modules_root=None,
        realesrgan_path=None,
        realesrgan_outscale=1.0,
        use_fp16=True,
        use_compile=True,
        ffmpeg_threads=thread_values[0],
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

    results: List[ThreadBenchmark] = []

    for idx, threads in enumerate(thread_values):
        print(f"\n▶️  Testing -threads {threads} ({idx + 1}/{len(thread_values)})")
        benchmark = ThreadBenchmark(threads=threads, attempts=args.repeat)
        service.ffmpeg_threads = threads

        if args.warmup:
            print("   🔥 Warmup run...")
            torch.cuda.empty_cache()
            service.process(
                face_path=str(face_path),
                audio_path="",
                output_path=None,
                static=True,
                fps=args.fps,
                pads=tuple(args.pads),
                audio_waveform=waveform,
                audio_sample_rate=sample_rate,
                frame_sink=None,
                batch_size_override=args.batch_size,
            )

        for attempt in range(args.repeat):
            torch.cuda.empty_cache()
            output_path: Optional[str] = None
            if video_dir is not None:
                output_name = f"threads_{threads:02d}_run{attempt + 1}.mp4"
                output_path = str(video_dir / output_name)

            start_time = time.perf_counter()
            stats = service.process(
                face_path=str(face_path),
                audio_path="",
                output_path=output_path,
                static=True,
                fps=args.fps,
                pads=tuple(args.pads),
                audio_waveform=waveform,
                audio_sample_rate=sample_rate,
                frame_sink=None,
                batch_size_override=args.batch_size,
            )
            elapsed = time.perf_counter() - start_time

            writing_time = stats.get('writing_frames_time', 0.0)
            ffmpeg_time = stats.get('ffmpeg_encoding_time', 0.0)
            inference_time = stats.get('inference_time', elapsed)

            benchmark.totals.append(elapsed)
            benchmark.writing_times.append(writing_time)
            benchmark.ffmpeg_times.append(ffmpeg_time)
            benchmark.inference_times.append(inference_time)

            print(
                f"   Attempt {attempt + 1}/{args.repeat}: total={_format_seconds(elapsed)}, "
                f"writing={_format_seconds(writing_time)}, ffmpeg={_format_seconds(ffmpeg_time)}"
            )

            if args.cooldown > 0:
                time.sleep(args.cooldown)

        if benchmark.totals:
            best_total = min(benchmark.totals)
            best_writing = min(benchmark.writing_times)
            median_total = statistics.median(benchmark.totals)
            median_writing = statistics.median(benchmark.writing_times)
            print(
                f"   ✅ Best total: {_format_seconds(best_total)} | median total: {_format_seconds(median_total)} | "
                f"best writing: {_format_seconds(best_writing)} | median writing: {_format_seconds(median_writing)}"
            )
        results.append(benchmark)

    ranked = sorted(results, key=lambda r: min(r.totals) if r.totals else float('inf'))
    print("\n🏁 Top configurations by best total time:")
    for row in ranked:
        if not row.totals:
            continue
        print(
            f"   -threads {row.threads:<3d} best total={_format_seconds(min(row.totals))} "
            f"best writing={_format_seconds(min(row.writing_times))}"
        )

    if args.output:
        payload = [entry.to_dict() for entry in results]
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\n📝 Saved results to {args.output}")


if __name__ == "__main__":
    main()
