#!/usr/bin/env python3
"""Sweep Wav2Lip batch size to evaluate throughput on a target GPU."""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from math import ceil
from pathlib import Path
from typing import Iterable, List, Optional

# Restrict execution to the requested GPU before importing torch.
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


@dataclass
class SweepResult:
    batch_size: int
    status: str
    total_time: Optional[float] = None
    inference_time: Optional[float] = None
    prep_time: Optional[float] = None
    tensor_time: Optional[float] = None
    batches: Optional[int] = None
    mel_chunks: Optional[int] = None
    throughput_chunks_per_s: Optional[float] = None
    notes: Optional[str] = None

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


def _format_seconds(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return f"{value:.2f}s"


def _sample_batches(min_batch: int, max_batch: int, step: int, explicit: Optional[Iterable[int]] = None) -> List[int]:
    if explicit:
        values = sorted({int(v) for v in explicit if int(v) > 0})
    else:
        values = list(range(min_batch, max_batch + 1, step))
        if values[-1] != max_batch:
            values.append(max_batch)
    return values


def run_sweep(
    service: LipsyncService,
    audio_waveform: torch.Tensor,
    audio_sample_rate: int,
    face_path: str,
    fps: float,
    batches: List[int],
    pads: tuple[int, int, int, int],
    repeat: int,
    cooldown: float,
    video_dir: Optional[Path],
) -> List[SweepResult]:
    results: List[SweepResult] = []
    torch.cuda.empty_cache()

    for idx, batch_size in enumerate(batches):
        print(f"\n▶️  Benchmarking batch_size={batch_size} ({idx + 1}/{len(batches)})")
        best_total: Optional[float] = None
        best_infer: Optional[float] = None
        notes: List[str] = []
        per_run_total: List[float] = []
        per_run_infer: List[float] = []

        for attempt in range(repeat):
            torch.cuda.empty_cache()
            start_time = time.perf_counter()
            output_path: Optional[str] = None
            if video_dir is not None:
                video_name = f"batch_{batch_size:05d}_run{attempt + 1}.mp4"
                output_path = str(video_dir / video_name)

            try:
                stats = service.process(
                    face_path=face_path,
                    audio_path="",
                    output_path=output_path,
                    static=True,
                    fps=fps,
                    pads=pads,
                    audio_waveform=audio_waveform,
                    audio_sample_rate=audio_sample_rate,
                    frame_sink=None,
                    batch_size_override=batch_size,
                )
                elapsed = time.perf_counter() - start_time
                per_run_total.append(elapsed)
                per_run_infer.append(stats.get("inference_time", elapsed))
                best_total = min(per_run_total) if per_run_total else elapsed
                best_infer = min(per_run_infer) if per_run_infer else stats.get("inference_time")
                mel_chunks = stats.get("num_mel_chunks")
                batches_count = ceil(mel_chunks / batch_size) if mel_chunks else None
                throughput = None
                if best_infer and mel_chunks:
                    throughput = mel_chunks / best_infer

                print(
                    f"   Attempt {attempt + 1}/{repeat}: total={elapsed:.2f}s, "
                    f"inference={stats.get('inference_time', float('nan')):.2f}s"
                )

                results.append(
                    SweepResult(
                        batch_size=batch_size,
                        status="ok",
                        total_time=elapsed,
                        inference_time=stats.get("inference_time"),
                        prep_time=stats.get("face_detection_time"),
                        tensor_time=None,
                        batches=batches_count,
                        mel_chunks=mel_chunks,
                        throughput_chunks_per_s=throughput,
                    )
                )
            except RuntimeError as err:
                message = str(err)
                notes.append(message)
                print(f"   ⚠️  RuntimeError: {message}")
                results.append(SweepResult(batch_size=batch_size, status="error", notes=message))
                break
            except torch.cuda.OutOfMemoryError as oom_err:
                message = f"CUDA OOM: {oom_err}"
                notes.append(message)
                print(f"   ❌  Out of memory at batch_size={batch_size}")
                results.append(SweepResult(batch_size=batch_size, status="oom", notes=message))
                break
            finally:
                if cooldown > 0:
                    time.sleep(cooldown)

        # Summarize best attempt if all runs succeeded
        if per_run_total:
            median = statistics.median(per_run_total)
            print(
                f"   ✅ Best total: {min(per_run_total):.2f}s | median: {median:.2f}s | "
                f"best inference: {best_infer:.2f}s"
            )

    return results


def summarize(results: List[SweepResult], top_k: int = 5) -> None:
    successful = [r for r in results if r.status == "ok" and r.total_time is not None]
    if not successful:
        print("\n❌ No successful runs recorded")
        return
    ranked = sorted(successful, key=lambda r: r.total_time or float("inf"))
    best = ranked[:top_k]

    print("\n🏁 Top configurations by total time:")
    for row in best:
        print(
            f"   batch={row.batch_size:<6d} total={_format_seconds(row.total_time)} "
            f"inference={_format_seconds(row.inference_time)} "
            f"throughput={row.throughput_chunks_per_s:.1f} chunks/s"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep wav2lip batch sizes to measure performance")
    parser.add_argument("--audio", type=str, default="/home/arman/workspace2/temp_web/audio_20251031_161642.wav",
                        help="Path to the audio WAV file used for the benchmark")
    parser.add_argument("--face", type=str, default=str(ROOT / "avatar.jpg"),
                        help="Path to a face image/video for inference")
    parser.add_argument("--checkpoint", type=str, default=str(ROOT / "Wav2Lip-SD-GAN.pt"),
                        help="Path to the Wav2Lip checkpoint")
    parser.add_argument("--min", dest="min_batch", type=int, default=1,
                        help="Minimum batch size to test")
    parser.add_argument("--max", dest="max_batch", type=int, default=1000,
                        help="Maximum batch size to test")
    parser.add_argument("--step", type=int, default=256,
                        help="Step size between batch sizes (use 1 for exhaustive sweep)")
    parser.add_argument("--values", type=int, nargs="*", default=None,
                        help="Explicit batch sizes to test (overrides min/max/step)")
    parser.add_argument("--repeat", type=int, default=1,
                        help="Number of attempts per batch size")
    parser.add_argument("--cooldown", type=float, default=0.0,
                        help="Pause between runs to let GPU cool down (seconds)")
    parser.add_argument("--pads", type=int, nargs=4, default=(0, 50, 0, 0),
                        help="Face padding tuple applied during preprocessing")
    parser.add_argument("--fps", type=float, default=25.0,
                        help="Frame rate for preprocessing/static cache")
    parser.add_argument("--output", type=str, default=None,
                        help="Optional JSON file to store full results")
    parser.add_argument("--video-dir", type=str, default=None,
                        help="Directory to store rendered videos for each batch size")
    parser.add_argument("--gpu", type=int, default=DEFAULT_GPU_ID,
                        help="Physical GPU id to use (mapped via CUDA_VISIBLE_DEVICES)")
    parser.add_argument("--warmup", action="store_true",
                        help="Run a warmup inference before measurements")

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    requested_gpu = _validate_gpu(args.gpu)
    print(f"Using GPU id {requested_gpu} (CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']})")

    audio_path = Path(args.audio)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    face_path = Path(args.face)
    if not face_path.exists():
        raise FileNotFoundError(f"Face image/video not found: {face_path}")

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    video_dir: Optional[Path] = None
    if args.video_dir:
        video_dir = Path(args.video_dir)
        video_dir.mkdir(parents=True, exist_ok=True)

    waveform, sample_rate = torchaudio.load(str(audio_path))
    if waveform.dim() == 2 and waveform.size(0) > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    print(f"Loaded audio waveform {waveform.shape} at {sample_rate} Hz")

    batches = _sample_batches(args.min_batch, args.max_batch, args.step, args.values)
    print(f"Batch sizes to evaluate: {batches}")

    service = LipsyncService(
        checkpoint_path=str(checkpoint_path),
        device='cuda',
        face_det_batch_size=16,
        wav2lip_batch_size=max(batches),
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

    if args.warmup:
        print("\n🔥 Running warmup inference...")
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
            batch_size_override=min(batches),
        )

    results = run_sweep(
        service=service,
        audio_waveform=waveform,
        audio_sample_rate=sample_rate,
        face_path=str(face_path),
        fps=args.fps,
        batches=batches,
        pads=tuple(args.pads),
        repeat=args.repeat,
        cooldown=args.cooldown,
        video_dir=video_dir,
    )

    summarize(results)

    if args.output:
        payload = [entry.to_dict() for entry in results]
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\n📝 Saved results to {args.output}")


if __name__ == "__main__":
    main()
