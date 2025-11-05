#!/usr/bin/env python3
"""Unified dynamic pipeline: inference → live encoding → audio mux."""
from __future__ import annotations

import argparse
import itertools
import json
import os
import queue
import subprocess
import sys
import threading
import time
import statistics
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, UTC
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torchaudio

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
MODERN_ROOT = PROJECT_ROOT / "modern-lipsync"
for candidate in (PROJECT_ROOT, MODERN_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from service import LipsyncService  # noqa: E402
from app_core import config  # noqa: E402
from app_core.services.segment_lipsync import _attach_audio  # noqa: E402

PADS = (0, 50, 0, 0)
DEFAULT_QUEUE_SIZE = 8
DEFAULT_PRESET = "veryfast"
DEFAULT_CRF = 18


def _ensure_list(value: Any) -> list[Any]:
    if isinstance(value, str):
        if "," in value:
            parts = [item.strip() for item in value.split(",")]
            return [part for part in parts if part]
        return [value]
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _slugify_params(params: Dict[str, Any]) -> str:
    if not params:
        return "default"
    parts = []
    for key in sorted(params):
        raw_val = params[key]
        if isinstance(raw_val, float):
            rounded = f"{raw_val:.4f}".rstrip("0").rstrip(".")
            normalized = rounded or "0"
            value_str = normalized.replace(".", "p")
        else:
            value_str = str(raw_val)
        value_str = value_str.replace("/", "_").replace(" ", "")
        if not value_str:
            value_str = "none"
        parts.append(f"{key}-{value_str}")
    slug = "_".join(parts)
    return slug[:160] if len(slug) > 160 else slug


def _load_sweep_config(config_path: Path) -> list[Dict[str, Any]]:
    with config_path.open("r", encoding="utf-8") as handle:
        raw = handle.read()
    suffix = config_path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "PyYAML is required to parse YAML sweep configs. Install via `pip install pyyaml`."
            ) from exc
        payload = yaml.safe_load(raw)
    else:
        payload = json.loads(raw)

    if payload is None:
        return []
    if not isinstance(payload, dict):
        raise ValueError("Sweep config must be a mapping at the top level")

    if "runs" in payload:
        runs = payload["runs"]
        if not isinstance(runs, list):
            raise ValueError("`runs` entry must be a list of mappings")
        combos: list[Dict[str, Any]] = []
        for entry in runs:
            if not isinstance(entry, dict):
                raise ValueError("Entries under `runs` must be mappings")
            combos.append(entry)
        return combos

    grid_payload = payload.get("grid", payload)
    if not isinstance(grid_payload, dict):
        raise ValueError("`grid` entry must be a mapping when present")
    if not grid_payload:
        return [{}]

    keys = list(grid_payload.keys())
    value_lists = []
    for key in keys:
        options = _ensure_list(grid_payload[key])
        if not options:
            raise ValueError(f"Parameter `{key}` has an empty option list")
        value_lists.append(options)

    combos = []
    for values in itertools.product(*value_lists):
        combos.append(dict(zip(keys, values)))
    return combos


def _coerce_override_value(value: Any, template: Any) -> Any:
    if template is None:
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "1", "yes", "on"}:
                return True
            if lowered in {"false", "0", "no", "off"}:
                return False
            try:
                if "." in value:
                    return float(value)
                return int(value)
            except ValueError:
                return value
        return value
    if isinstance(template, bool):
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "1", "yes", "on"}:
                return True
            if lowered in {"false", "0", "no", "off"}:
                return False
        return bool(value)
    if isinstance(template, int):
        if isinstance(value, bool):
            return int(value)
        return int(value)
    if isinstance(template, float):
        return float(value)
    return value


def _service_signature(args: argparse.Namespace) -> tuple[Any, ...]:
    checkpoint = str(Path(args.checkpoint).expanduser())
    return (
        checkpoint,
        args.device,
        bool(args.disable_segmentation),
        bool(args.disable_super_resolution),
        bool(args.disable_realesrgan),
        bool(args.disable_fp16),
        bool(args.disable_compile),
    )


class ServiceManager:
    def __init__(self, allow_reuse: bool):
        self.allow_reuse = allow_reuse
        self._service: Optional[LipsyncService] = None
        self._details: Dict[str, Any] | None = None
        self._signature: tuple[Any, ...] | None = None
        self._preload_keys: set[tuple[str, float, tuple[int, ...]]] = set()
        self._warmup_keys: set[tuple[Any, ...]] = set()

    def has_service(self) -> bool:
        return self._service is not None

    def expect_service_reuse(self, args: argparse.Namespace) -> bool:
        if not self.allow_reuse or self._service is None or self._signature is None:
            return False
        return _service_signature(args) == self._signature

    def inject_prepared_service(
        self,
        service: LipsyncService,
        details: Dict[str, Any],
        signature: tuple[Any, ...],
        preload_keys: Optional[set[tuple[str, float, tuple[int, ...]]]] = None,
        warmup_keys: Optional[set[tuple[Any, ...]]] = None,
    ) -> None:
        if not self.allow_reuse:
            raise RuntimeError("ServiceManager configured without reuse support cannot accept injected services")
        self._service = service
        self._details = dict(details)
        self._signature = signature
        self._preload_keys = set(preload_keys or set())
        self._warmup_keys = set(warmup_keys or set())

    def obtain(self, args: argparse.Namespace, avatar_path: Path, fps: float) -> Dict[str, Any]:
        signature = _service_signature(args)
        reuse_possible = self.allow_reuse and self._service is not None and signature == self._signature
        if not reuse_possible:
            service, details = build_service(args)
            self._service = service
            self._details = details
            self._signature = signature
            self._preload_keys = set()
            self._warmup_keys = set()
        assert self._service is not None  # for type checkers
        assert self._details is not None

        service = self._service

        # Update mutable runtime parameters even when reusing.
        try:
            service.face_det_batch_size = int(args.face_det_batch)
        except Exception:
            pass
        try:
            service.wav2lip_batch_size = int(args.batch_size)
        except Exception:
            pass
        try:
            service.ffmpeg_threads = int(args.ffmpeg_threads)
        except Exception:
            pass
        try:
            service.ffmpeg_filter_threads = max(0, int(args.ffmpeg_filter_threads))
        except Exception:
            pass

        current_details = dict(self._details)
        current_details.update({
            "face_det_batch_size": getattr(service, "face_det_batch_size", args.face_det_batch),
            "wav2lip_batch_size": getattr(service, "wav2lip_batch_size", args.batch_size),
            "ffmpeg_threads": getattr(service, "ffmpeg_threads", args.ffmpeg_threads),
            "ffmpeg_filter_threads": getattr(service, "ffmpeg_filter_threads", args.ffmpeg_filter_threads),
        })
        if self.allow_reuse:
            self._details = dict(current_details)

        preload_key = (str(avatar_path), float(fps), PADS)
        need_preload = True
        if self.allow_reuse and preload_key in self._preload_keys:
            need_preload = False

        warmup_key = None
        need_warmup = False
        if args.warmup_duration > 0:
            warmup_key = (preload_key, int(args.batch_size), float(args.warmup_duration))
            need_warmup = True
            if self.allow_reuse and warmup_key in self._warmup_keys:
                need_warmup = False

        return {
            "service": service,
            "details": current_details,
            "service_reused": reuse_possible,
            "preload_key": preload_key,
            "need_preload": need_preload,
            "warmup_key": warmup_key,
            "need_warmup": need_warmup,
        }

    def perform_preload(self, preload_key: tuple[str, float, tuple[int, ...]], avatar_path: Path, fps: float) -> float:
        if self._service is None:
            raise RuntimeError("Service is not initialized for preload")
        start = time.perf_counter()
        self._service.preload_video_cache(face_path=str(avatar_path), fps=fps, pads=PADS)
        duration = time.perf_counter() - start
        if self.allow_reuse:
            self._preload_keys.add(preload_key)
        return duration

    def register_warmup(self, warmup_key: Optional[tuple[Any, ...]]) -> None:
        if not self.allow_reuse or warmup_key is None:
            return
        self._warmup_keys.add(warmup_key)


def _coerce_value(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (int, float, bool)):
        return value
    return value


def build_service(args: argparse.Namespace) -> tuple[LipsyncService, Dict[str, Any]]:
    segmentation_path = config.SEGMENTATION_PATH_HD if (config.ENABLE_SEGMENTATION and os.path.exists(config.SEGMENTATION_PATH_HD) and not args.disable_segmentation) else None
    sr_path = config.SR_PATH_HD if (config.ENABLE_SUPER_RESOLUTION and os.path.exists(config.SR_PATH_HD) and not args.disable_super_resolution) else None
    realesrgan_path = config.REALESRGAN_PATH if (config.ENABLE_REALESRGAN and os.path.exists(config.REALESRGAN_PATH) and not args.disable_realesrgan) else None

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

    details = {
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
    return service, details


def warmup_service(service: LipsyncService, avatar_path: Path, fps: float, duration: float, batch_size: int) -> None:
    if duration <= 0:
        return
    print(f"🔥 Warmup: running {duration:.1f}s dummy inference to trigger compilation...")
    samples = int(16000 * duration)
    if samples < service.mel_step_size:
        samples = service.mel_step_size
    dummy_audio = torch.zeros(1, samples, dtype=torch.float32)

    def _sink(_frame: np.ndarray) -> None:
        return

    service.process(
        face_path=str(avatar_path),
        audio_path="",
        output_path=None,
        static=False,
        fps=fps,
        pads=PADS,
        audio_waveform=dummy_audio,
        audio_sample_rate=16000,
        frame_sink=_sink,
        batch_size_override=min(batch_size, service.wav2lip_batch_size),
        frame_offset=0,
    )
    print("🔥 Warmup complete")


def spawn_encoder(
    fps: float,
    temp_video_path: Path,
    queue_size: int,
    preset: str,
    crf: int,
    threads: int,
) -> tuple[queue.Queue, threading.Thread, Dict[str, Any]]:
    frame_queue: "queue.Queue[Optional[np.ndarray]]" = queue.Queue(maxsize=max(1, queue_size))
    encode_report: Dict[str, Any] = {"frame_count": 0}
    encode_error: Dict[str, BaseException] = {}

    def _encode_worker() -> None:
        proc: Optional[subprocess.Popen] = None
        try:
            first_item = frame_queue.get()
            if first_item is None:
                encode_report.update({
                    "return_code": 0,
                    "encode_wall_time": 0.0,
                    "stderr": "",
                })
                return
            frame0 = first_item
            if not isinstance(frame0, np.ndarray):
                raise TypeError("Frame payload must be numpy.ndarray")
            height, width = frame0.shape[:2]

            command = [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "-s",
                f"{width}x{height}",
                "-r",
                str(fps),
                "-i",
                "pipe:0",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                preset,
                "-crf",
                str(crf),
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
            ]
            if threads >= 0:
                command.extend(["-threads", str(threads)])
            command.append(str(temp_video_path))

            encode_start = time.perf_counter()
            proc = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

            def _write_frame(frame: np.ndarray) -> None:
                if frame.dtype != np.uint8:
                    frame = frame.astype(np.uint8, copy=False)
                if not frame.flags.c_contiguous:
                    frame = np.ascontiguousarray(frame)
                if proc.stdin is None:
                    raise RuntimeError("FFmpeg stdin is not available")
                proc.stdin.write(memoryview(frame))

            _write_frame(frame0)
            encode_report["frame_count"] = 1

            while True:
                item = frame_queue.get()
                if item is None:
                    break
                _write_frame(item)
                encode_report["frame_count"] += 1

            if proc.stdin is not None:
                proc.stdin.close()
            return_code = proc.wait()
            stderr_output = ""
            if proc.stderr is not None:
                stderr_output = proc.stderr.read().decode("utf-8", errors="ignore")
                proc.stderr.close()

            encode_report.update({
                "return_code": return_code,
                "stderr": stderr_output,
                "encode_wall_time": time.perf_counter() - encode_start,
                "width": width,
                "height": height,
            })
            if return_code != 0:
                raise RuntimeError(f"FFmpeg exited with code {return_code}: {stderr_output}")
        except BaseException as exc:  # noqa: BLE001
            encode_error["error"] = exc
        finally:
            try:
                if proc and proc.stderr is not None:
                    proc.stderr.close()
            except Exception:
                pass
            try:
                if proc and proc.stdin is not None and not proc.stdin.closed:
                    proc.stdin.close()
            except Exception:
                pass

    encoder_thread = threading.Thread(target=_encode_worker, name="encoder-thread", daemon=True)
    encoder_thread.start()
    encode_report["_error_holder"] = encode_error
    return frame_queue, encoder_thread, encode_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="End-to-end dynamic avatar pipeline")
    parser.add_argument("--avatar", default=config.AVATAR_VIDEO_PATH, help="Path to the avatar video source")
    parser.add_argument("--audio", default="/home/arman/musetalk/avatar/audio_20251030_145910.wav", help="Path to the audio track")
    parser.add_argument("--output", default="/home/arman/musetalk/avatar/outputs/dynamic_final.mp4", help="Final video path")
    parser.add_argument("--checkpoint", default=config.CHECKPOINT_PATH_GAN, help="Path to Wav2Lip checkpoint")
    parser.add_argument("--device", default="cuda:0", help="Torch device")
    parser.add_argument("--batch-size", type=int, default=4096, help="Batch size override for Wav2Lip")
    parser.add_argument("--face-det-batch", type=int, default=16, help="Face detector batch size")
    parser.add_argument("--fps", type=float, default=config.AVATAR_FPS, help="Target FPS for encoding")
    parser.add_argument("--temp-video", default=None, help="Optional path for intermediate silent video")
    parser.add_argument("--queue-size", type=int, default=DEFAULT_QUEUE_SIZE, help="Frame queue size between inference and encoder")
    parser.add_argument("--ffmpeg-preset", default=DEFAULT_PRESET, help="FFmpeg preset")
    parser.add_argument("--ffmpeg-crf", type=int, default=DEFAULT_CRF, help="FFmpeg CRF value")
    parser.add_argument("--ffmpeg-threads", type=int, default=16, help="Threads for FFmpeg")
    parser.add_argument("--ffmpeg-filter-threads", type=int, default=0, help="FFmpeg filter threads (passed to LipsyncService)")
    parser.add_argument("--warmup-duration", type=float, default=0.0, help="Seconds of dummy audio for warmup inference")
    parser.add_argument("--meta-out", default=None, help="Optional metadata JSON path")
    parser.add_argument("--disable-segmentation", action="store_true", help="Force disable segmentation model")
    parser.add_argument("--disable-super-resolution", action="store_true", help="Force disable ESRGAN")
    parser.add_argument("--disable-realesrgan", action="store_true", help="Force disable RealESRGAN")
    parser.add_argument("--disable-fp16", action="store_true", help="Disable FP16 inference")
    parser.add_argument("--disable-compile", action="store_true", help="Disable torch.compile optimizations")
    parser.add_argument("--sweep-config", default=None, help="Path to JSON/YAML config describing parameter sweep")
    parser.add_argument("--sweep-output-dir", default=None, help="Directory to store sweep run artifacts")
    parser.add_argument("--sweep-repeats", type=int, default=1, help="Repeat each sweep combination this many times")
    parser.add_argument("--sweep-limit", type=int, default=None, help="Optional cap on the number of combinations to execute")
    parser.add_argument("--sweep-cleanup", action="store_true", help="Remove rendered videos after each sweep run to save space")
    parser.add_argument("--sweep-reuse-service", action="store_true", help="Reuse LipsyncService and preloaded video cache across sweep runs when safe")
    return parser.parse_args()


def run_pipeline(args: argparse.Namespace, service_manager: Optional[ServiceManager] = None) -> Dict[str, Any]:
    avatar_path = Path(args.avatar).expanduser().resolve()
    audio_path = Path(args.audio).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    temp_video_path = Path(args.temp_video).expanduser().resolve() if args.temp_video else (output_path.parent / (output_path.stem + "_silent.mp4"))
    temp_video_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not avatar_path.exists():
        raise FileNotFoundError(f"Avatar source not found: {avatar_path}")
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio source not found: {audio_path}")

    print(f"🎯 Output video: {output_path}")

    fps = args.fps if args.fps > 0 else config.AVATAR_FPS

    manager = service_manager or ServiceManager(allow_reuse=False)
    if manager.expect_service_reuse(args):
        print("📦 Reusing GAN service...")
    else:
        print("📦 Loading GAN service...")
    service_context = manager.obtain(args, avatar_path, fps)
    service: LipsyncService = service_context["service"]
    service_details: Dict[str, Any] = service_context["details"]
    service_reused = bool(service_context["service_reused"])

    preload_time = 0.0
    if service_context["need_preload"]:
        print("🎬 Preloading video cache (dynamic mode)...")
        preload_time = manager.perform_preload(service_context["preload_key"], avatar_path, fps)
        print(f"✅ Video cache ready in {preload_time:.2f}s")
    else:
        print("🎬 Video cache already warm (reuse).")

    warmup_executed = False
    if service_context["need_warmup"]:
        warmup_service(service, avatar_path, fps, args.warmup_duration, args.batch_size)
        warmup_executed = args.warmup_duration > 0
        manager.register_warmup(service_context["warmup_key"])
    elif args.warmup_duration > 0:
        print("🔥 Warmup skipped (reuse).")

    if args.warmup_duration > 0:
        warmup_flag = "executed" if warmup_executed else "reused"
    else:
        warmup_flag = "disabled"

    print("🔊 Loading audio waveform...")
    waveform, sample_rate = torchaudio.load(str(audio_path))

    frame_queue, encoder_thread, encode_report = spawn_encoder(
        fps=fps,
        temp_video_path=temp_video_path,
        queue_size=args.queue_size,
        preset=args.ffmpeg_preset,
        crf=args.ffmpeg_crf,
        threads=args.ffmpeg_threads,
    )

    inference_frames = {"count": 0}

    def _frame_sink(frame: np.ndarray) -> None:
        inference_frames["count"] += 1
        while True:
            try:
                frame_queue.put(frame.copy(), timeout=5.0)
                break
            except queue.Full:
                continue

    if service.is_cuda:
        torch.cuda.synchronize()
    inference_start = time.perf_counter()
    stats: Dict[str, Any] = {}
    try:
        stats = service.process(
            face_path=str(avatar_path),
            audio_path="",
            output_path=None,
            static=False,
            fps=fps,
            pads=PADS,
            audio_waveform=waveform,
            audio_sample_rate=sample_rate,
            frame_sink=_frame_sink,
            batch_size_override=args.batch_size,
            frame_offset=0,
        )
        if service.is_cuda:
            torch.cuda.synchronize()
    finally:
        inference_wall = time.perf_counter() - inference_start
        while True:
            try:
                frame_queue.put(None, timeout=5.0)
                break
            except queue.Full:
                continue
        encoder_thread.join()

    encode_error_holder = encode_report.pop("_error_holder", {})
    if encode_error_holder.get("error"):
        raise encode_error_holder["error"]

    if encode_report.get("return_code") != 0:
        raise RuntimeError(f"FFmpeg encoding failed: {encode_report}")

    print("🔗 Attaching audio...")
    merge_start = time.perf_counter()
    return_code, stderr_output = _attach_audio(temp_video_path, audio_path, output_path)
    merge_time = time.perf_counter() - merge_start
    if return_code != 0:
        raise RuntimeError(f"Audio mux failed: {stderr_output}")

    summary = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "avatar_path": str(avatar_path),
        "audio_path": str(audio_path),
        "output_video": str(output_path),
        "temp_video": str(temp_video_path),
        "fps": fps,
        "preload_time": preload_time,
        "warmup_duration": args.warmup_duration,
        "inference_frames": inference_frames["count"],
        "inference_wall_time": inference_wall,
        "inference_stats": {k: _coerce_value(v) for k, v in stats.items()},
        "encode": encode_report,
        "merge_time": merge_time,
        "service_config": service_details,
        "reuse": {
            "service": service_reused,
            "preload": not service_context["need_preload"],
            "warmup": warmup_flag,
        },
    }

    print("\n✅ Pipeline complete")
    print(f"   Frames: {inference_frames['count']} @ {fps:.3f} fps")
    print(f"   Inference wall time: {inference_wall:.2f}s")
    print(f"   Encoding wall time: {encode_report.get('encode_wall_time', 0.0):.2f}s")
    print(f"   Merge time: {merge_time:.2f}s")
    print(f"   Final video: {output_path}")

    if args.meta_out:
        meta_path = Path(args.meta_out).expanduser().resolve()
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        with meta_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)
        print(f"📝 Metadata saved to {meta_path}")
        summary["meta_out"] = str(meta_path)

    return summary


def run_sweep(args: argparse.Namespace) -> None:
    config_path = Path(args.sweep_config).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Sweep config not found: {config_path}")

    combinations = _load_sweep_config(config_path)
    if args.sweep_limit is not None:
        combinations = combinations[: max(0, args.sweep_limit)]

    if not combinations:
        print("⚠️ No sweep combinations defined; exiting.")
        return

    repeats = max(1, args.sweep_repeats)
    total_runs = len(combinations) * repeats
    print(f"🧪 Sweep: {len(combinations)} combinations × {repeats} repeats (total {total_runs} runs)")

    sweep_root = Path(args.sweep_output_dir).expanduser().resolve() if args.sweep_output_dir else (PROJECT_ROOT / "temp_web" / "sweeps" / datetime.now(UTC).strftime("%Y%m%d_%H%M%S"))
    sweep_root.mkdir(parents=True, exist_ok=True)
    results_manifest = sweep_root / "results_summary.json"

    base_args = deepcopy(vars(args))
    for cleanup_key in ("sweep_config", "sweep_output_dir", "sweep_repeats", "sweep_limit", "sweep_cleanup", "sweep_reuse_service"):
        base_args.pop(cleanup_key, None)
    valid_override_keys = set(base_args.keys())

    run_records: list[Dict[str, Any]] = []
    grouped_records: defaultdict[str, list[Dict[str, Any]]] = defaultdict(list)
    service_manager: Optional[ServiceManager] = ServiceManager(allow_reuse=args.sweep_reuse_service) if args.sweep_reuse_service else None

    for combo_index, overrides in enumerate(combinations):
        if not isinstance(overrides, dict):
            raise ValueError(f"Invalid sweep entry at index {combo_index}: expected mapping, got {type(overrides)}")
        invalid_keys = [key for key in overrides if key not in valid_override_keys]
        if invalid_keys:
            raise ValueError(f"Unsupported sweep parameter(s) in combination {combo_index}: {invalid_keys}")
        coerced_overrides = {key: _coerce_override_value(value, base_args.get(key)) for key, value in overrides.items()}
        combo_slug = _slugify_params(coerced_overrides)
        for repeat_index in range(repeats):
            run_number = len(run_records) + 1
            run_dir = sweep_root / f"{combo_index:03d}_{combo_slug}_r{repeat_index}"
            run_dir.mkdir(parents=True, exist_ok=True)

            run_args_data = deepcopy(base_args)
            run_args_data.update(coerced_overrides)
            run_args_data["output"] = str(run_dir / "dynamic_final.mp4")
            run_args_data["temp_video"] = str(run_dir / "dynamic_final_silent.mp4")
            run_args_data["meta_out"] = str(run_dir / "summary.json")

            run_args = argparse.Namespace(**run_args_data)
            print(f"\n=== Sweep run {run_number}/{total_runs} :: {combo_slug} (repeat {repeat_index + 1}/{repeats}) ===")
            summary: Optional[Dict[str, Any]] = None
            success = True
            error_message = None
            try:
                summary = run_pipeline(run_args, service_manager=service_manager)
            except Exception as exc:  # noqa: BLE001
                success = False
                error_message = f"{type(exc).__name__}: {exc}"
                print(f"❌ Sweep run failed: {error_message}")

            encode_stats = summary.get("encode", {}) if success and summary else {}
            encode_time = encode_stats.get("encode_wall_time") if encode_stats else None
            record = {
                "run_number": run_number,
                "combination_index": combo_index,
                "repeat_index": repeat_index,
                "slug": combo_slug,
                "params": coerced_overrides,
                "encode_time": encode_time,
                "frames": summary.get("inference_frames") if summary else None,
                "output_video": summary.get("output_video") if summary else run_args_data["output"],
                "meta_path": (summary.get("meta_out") if summary else run_args_data["meta_out"]),
                "success": success,
                "reuse": summary.get("reuse") if summary else None,
            }
            if error_message:
                record["error"] = error_message
            run_records.append(record)
            if success:
                grouped_records[combo_slug].append(record)

            if args.sweep_cleanup:
                for path_str in (run_args_data["output"], run_args_data["temp_video"]):
                    path = Path(path_str)
                    try:
                        path.unlink(missing_ok=True)
                    except Exception as exc:  # pragma: no cover - best effort
                        print(f"⚠️ Cleanup failed for {path}: {exc}")

    aggregate_entries: list[Dict[str, Any]] = []
    for slug, records in grouped_records.items():
        times = [record["encode_time"] for record in records if isinstance(record.get("encode_time"), (int, float))]
        if not times:
            continue
        aggregate_entries.append({
            "slug": slug,
            "params": records[0]["params"],
            "runs": len(records),
            "time_avg": statistics.mean(times),
            "time_median": statistics.median(times),
            "time_min": min(times),
            "time_max": max(times),
        })
    aggregate_entries.sort(key=lambda entry: entry["time_avg"])

    summary_payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "config_path": str(config_path),
        "total_runs": total_runs,
        "runs": run_records,
        "aggregates": aggregate_entries,
    }
    with results_manifest.open("w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, ensure_ascii=False, indent=2)

    print("\n🏁 Sweep complete")
    if aggregate_entries:
        print("⭐ Top configurations by average encode time:")
        for idx, entry in enumerate(aggregate_entries[:5], start=1):
            print(
                f"  {idx}. {entry['slug']}: avg={entry['time_avg']:.2f}s "
                f"(min={entry['time_min']:.2f}s, max={entry['time_max']:.2f}s) -> {entry['params']}"
            )
    else:
        print("⚠️ No successful runs to summarize.")


def main() -> None:
    args = parse_args()
    if args.sweep_config:
        run_sweep(args)
        return
    run_pipeline(args)


if __name__ == "__main__":
    main()
