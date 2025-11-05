"""Segmented lipsync generation helpers."""
from __future__ import annotations

import json
import numbers
import queue
import subprocess
import threading
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torchaudio
import wave

from .. import state
from ..config import AVATAR_FPS, AVATAR_IMAGE, OUTPUT_DIR, TEMP_DIR
from .tts import convert_to_wav, generate_tts

FACE_PADS = (0, 50, 0, 0)
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
SEGMENT_META_FILENAME = "meta.json"


def _ensure_service_cuda_device(service) -> None:
    if not torch.cuda.is_available():
        return
    target_device = getattr(service, "device", None)
    if target_device is None:
        return
    try:
        torch.cuda.set_device(torch.device(str(target_device)))
    except Exception:
        pass


@dataclass
class SegmentEncodingResult:
    """Metadata about a single encoded video segment."""

    index: int
    frame_start: int
    frame_end: int
    frame_count: int
    write_time: float
    ffmpeg_time: float
    return_code: int
    stderr: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {
            "index": self.index,
            "frame_start": self.frame_start,
            "frame_end": self.frame_end,
            "frame_count": self.frame_count,
            "write_time": self.write_time,
            "ffmpeg_time": self.ffmpeg_time,
            "return_code": self.return_code,
            "stderr": self.stderr,
        }


@dataclass
class SegmentTiming:
    """Timing breakdown for segmented generation."""

    tts_time: float
    audio_convert_time: float
    capture_time: float
    inference_time: float
    encode_time: float
    merge_time: float
    audio_mux_time: float
    total_time: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class SegmentJobResult:
    """Final result of segmented lip-sync generation."""

    job_id: str
    video_path: Path
    temp_dir: Path
    segments: int
    requested_segments: Optional[int]
    total_frames: int
    resolution: Tuple[int, int]
    timings: SegmentTiming
    segment_results: List[SegmentEncodingResult]
    inference_stats: Dict[str, float]
    capture_workers: int
    capture_chunks: int

    def dump_metadata(self) -> None:
        payload = {
            "job_id": self.job_id,
            "video_path": str(self.video_path),
            "segments": self.segments,
            "requested_segments": self.requested_segments,
            "total_frames": self.total_frames,
            "resolution": {
                "height": int(self.resolution[0]),
                "width": int(self.resolution[1]),
            },
            "timings": self.timings.to_dict(),
            "segment_results": [segment.to_dict() for segment in self.segment_results],
            "inference_stats": self.inference_stats,
            "capture_workers": self.capture_workers,
            "capture_chunks": self.capture_chunks,
            "created_at": datetime.utcnow().isoformat() + "Z",
        }
        meta_path = self.temp_dir / SEGMENT_META_FILENAME
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        with open(meta_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)


def run_segmented_lipsync(
    text: str,
    language: str = "ru",
    segments: int = 16,
    batch_size: int = 1024,
) -> SegmentJobResult:
    text = text.strip()
    if not text:
        raise ValueError("Текст не может быть пустым")

    gan_services = [svc for svc in state.get_all_gan_services() if svc]
    service_pool = list(gan_services)
    if not service_pool and state.lipsync_service_nogan is not None:
        service_pool = [state.lipsync_service_nogan]

    if not service_pool:
        raise RuntimeError("Модель lipsync не загружена")

    primary_service = service_pool[0]

    if len(service_pool) > 1:
        device_labels = ", ".join(str(getattr(svc, "device", "cuda")) for svc in service_pool)
        print(f"🧠 Сегментный пайплайн использует {len(service_pool)} GAN сервисов: {device_labels}")

    static_mode = bool(state.avatar_static_mode or not _avatar_supports_video())
    encode_fps = 25.0

    start_total = time.perf_counter()

    start_tts = time.perf_counter()
    audio_data = generate_tts(text, language)
    tts_time = time.perf_counter() - start_tts

    start_convert = time.perf_counter()
    audio_waveform, audio_sample_rate = convert_to_wav(audio_data, None)
    audio_convert_time = time.perf_counter() - start_convert

    audio_samples = int(audio_waveform.shape[1])
    audio_duration_seconds = audio_samples / float(audio_sample_rate)

    job_id = datetime.now().strftime("segment_%Y%m%d_%H%M%S_%f")
    temp_root = Path(TEMP_DIR) / "segments" / job_id
    segments_dir = temp_root / "chunks"
    segments_dir.mkdir(parents=True, exist_ok=True)

    audio_path = temp_root / "audio.wav"
    _save_waveform(audio_path, audio_waveform, audio_sample_rate)

    for index, svc in enumerate(service_pool, start=1):
        try:
            _ensure_service_cuda_device(svc)
            if static_mode:
                svc.preload_static_face(
                    face_path=AVATAR_IMAGE,
                    fps=encode_fps,
                    pads=FACE_PADS,
                )
            else:
                svc.preload_video_cache(
                    face_path=AVATAR_IMAGE,
                    fps=encode_fps,
                    pads=FACE_PADS,
                )
        except Exception as preload_error:
            suffix = f" #{index}" if len(service_pool) > 1 else ""
            print(f"⚠️ Не удалось подготовить аватар для сервиса{suffix}: {preload_error}")

    requested_segments = segments if segments > 0 else None
    target_segments = segments if segments > 0 else 16
    segments_count = max(1, target_segments)

    if static_mode:
        start_capture = time.perf_counter()
        all_frames, primary_stats = _run_full_inference_capture(
            primary_service,
            static_mode=static_mode,
            audio_waveform=audio_waveform,
            audio_sample_rate=audio_sample_rate,
            batch_size=batch_size,
            fps=encode_fps,
        )
        inference_phase_time = time.perf_counter() - start_capture

        total_frames = len(all_frames)
        if total_frames == 0:
            raise RuntimeError("Не удалось получить кадры для кодирования")

        _sample_ranges, expected_counts, frame_offsets = _prepare_segment_layout(
            audio_samples, segments_count, audio_sample_rate, encode_fps
        )

        encode_start = time.perf_counter()
        (
            segment_results,
            segment_paths,
            total_encode_time,
            active_workers,
            segment_stats,
        ) = _encode_segments_from_frames(
            frames=all_frames,
            expected_counts=expected_counts,
            frame_offsets=frame_offsets,
            fps=encode_fps,
            segments_dir=segments_dir,
        )
        encode_phase_time = time.perf_counter() - encode_start
        capture_time = inference_phase_time + encode_phase_time

        if not segment_results:
            raise RuntimeError("Не удалось получить сегменты для кодирования")

        encoded_frames = sum(result.frame_count for result in segment_results)
        if encoded_frames != total_frames:
            raise RuntimeError(
                f"Несовпадение количества кадров: инференс дал {total_frames}, кодирование вернуло {encoded_frames}"
            )

        raw_stats = dict(primary_stats or {})
    else:
        start_capture = time.perf_counter()
        segment_results, raw_stats, segment_paths, active_workers, segment_stats = _process_segments_streaming(
            service_pool,
            static_mode=static_mode,
            audio_waveform=audio_waveform,
            audio_sample_rate=audio_sample_rate,
            batch_size=batch_size,
            segments_count=segments_count,
            fps=encode_fps,
            segments_dir=segments_dir,
        )
        capture_time = time.perf_counter() - start_capture

        if not segment_results:
            raise RuntimeError("Не удалось получить сегменты для кодирования")

        total_frames = sum(result.frame_count for result in segment_results)
        if total_frames == 0:
            raise RuntimeError("Не удалось получить кадры для кодирования")

        total_encode_time = sum(result.ffmpeg_time for result in segment_results)

    safe_segments = max(1, min(total_frames, target_segments))

    if requested_segments is None:
        print(
            f"🎯 Авторазбиение: длина аудио {audio_duration_seconds:.2f}s, кадров {total_frames}, сегменты {safe_segments}"
        )
    else:
        print(
            f"🎯 Разбиение аудио: запрос {requested_segments}, использовано {safe_segments} сегментов (кадров {total_frames})"
        )

    effective_fps = encode_fps
    encoded_duration = (total_frames / effective_fps) if effective_fps > 0 else 0.0
    sync_delta = abs(encoded_duration - audio_duration_seconds)
    if sync_delta > 0.05:  # ~3 frames @ 60fps, warns about noticeable drift
        print(
            f"⚠️ Предупреждение синхронизации: видео длительность {encoded_duration:.3f}s против аудио {audio_duration_seconds:.3f}s"
        )
    print(f"🎚️ Частота кадров для кодирования: {effective_fps:.3f} fps")

    merged_path = temp_root / "merged_no_audio.mp4"
    start_merge = time.perf_counter()
    merge_rc, merge_stderr = _concat_segments(segment_paths, merged_path)
    merge_time = time.perf_counter() - start_merge
    if merge_rc != 0:
        raise RuntimeError(f"Ошибка склейки сегментов: {merge_stderr}")

    final_output = Path(OUTPUT_DIR) / f"{job_id}.mp4"
    start_mux = time.perf_counter()
    mux_rc, mux_stderr = _attach_audio(merged_path, audio_path, final_output)
    audio_mux_time = time.perf_counter() - start_mux
    if mux_rc != 0:
        raise RuntimeError(f"Ошибка добавления аудио: {mux_stderr}")

    normalized_stats = {k: _normalize_stat(v) for k, v in (raw_stats or {}).items()}
    normalized_stats["effective_fps"] = float(effective_fps)
    normalized_stats["segment_encode_time_total"] = float(total_encode_time)
    normalized_stats["segment_encode_workers"] = float(active_workers or 0)
    normalized_stats["segment_frames_encoded"] = float(total_frames)
    inference_time_wall = float(
        normalized_stats.get("inference_time_max")
        or normalized_stats.get("inference_time")
        or normalized_stats.get("process_time")
        or 0.0
    )

    resolution: Optional[Tuple[int, int]] = None
    for stats_entry in segment_stats:
        if not stats_entry:
            continue
        height = int(_normalize_stat(stats_entry.get("frame_height", 0)))
        width = int(_normalize_stat(stats_entry.get("frame_width", 0)))
        if height > 0 and width > 0:
            resolution = (height, width)
            break

    if (resolution is None or resolution[0] <= 0 or resolution[1] <= 0) and state.avatar_preloaded is not None:
        height, width = state.avatar_preloaded.shape[:2]
        resolution = (int(height), int(width))

    if (resolution is None or resolution[0] <= 0 or resolution[1] <= 0) and segment_paths:
        probed_height, probed_width = _probe_video_resolution(segment_paths[0])
        if probed_height > 0 and probed_width > 0:
            resolution = (probed_height, probed_width)

    if resolution is None:
        resolution = (0, 0)

    timings = SegmentTiming(
        tts_time=tts_time,
        audio_convert_time=audio_convert_time,
        capture_time=capture_time,
        inference_time=inference_time_wall,
        encode_time=total_encode_time,
        merge_time=merge_time,
        audio_mux_time=audio_mux_time,
        total_time=time.perf_counter() - start_total,
    )

    result = SegmentJobResult(
        job_id=job_id,
        video_path=final_output,
        temp_dir=temp_root,
        segments=safe_segments,
        requested_segments=requested_segments,
        total_frames=total_frames,
        resolution=resolution,
        timings=timings,
        segment_results=segment_results,
        inference_stats=normalized_stats,
        capture_workers=active_workers or min(len(service_pool), len(segment_results)) or 1,
        capture_chunks=len(segment_results) if segment_results else 0,
    )
    result.dump_metadata()
    return result


def load_segment_metadata(job_id: str) -> Optional[Dict[str, object]]:
    meta_path = Path(TEMP_DIR) / "segments" / job_id / SEGMENT_META_FILENAME
    if not meta_path.exists():
        return None
    with open(meta_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _avatar_supports_video() -> bool:
    return Path(AVATAR_IMAGE).suffix.lower() in VIDEO_EXTENSIONS


def _save_waveform(path: Path, waveform, sample_rate: int) -> None:
    tensor = waveform.detach().cpu()
    save_kwargs = dict(
        filepath=str(path),
        src=tensor,
        sample_rate=sample_rate,
        channels_first=True,
        format="wav",
    )
    try:
        torchaudio.save(**save_kwargs, backend="sox_io")  # type: ignore[arg-type]
        return
    except TypeError:
        pass
    except RuntimeError as err:
        if "TorchCodec" not in str(err):
            raise

    # Fallback: manual WAV writer (PCM16)
    audio_np = tensor.numpy()
    if audio_np.ndim == 1:
        audio_np = audio_np[None, :]
    clipped = np.clip(audio_np, -1.0, 1.0)
    pcm16 = (clipped * 32767.0).round().astype(np.int16)
    interleaved = pcm16.T.reshape(-1)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(pcm16.shape[0])
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(interleaved.tobytes())


def _normalize_stat(value):
    if isinstance(value, numbers.Number):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def _resolve_effective_fps(
    raw_stats: Optional[Dict[str, float]],
    total_frames: int,
    audio_duration_seconds: float,
) -> float:
    derived = None
    if audio_duration_seconds > 0:
        derived = total_frames / audio_duration_seconds if total_frames > 0 else 0.0

    if raw_stats:
        candidate_keys = (
            "fps_max",
            "video_fps_max",
            "fps",
            "video_fps",
        )
        candidates: List[Tuple[str, float]] = []
        for key in candidate_keys:
            candidate = raw_stats.get(key)
            normalized = _normalize_stat(candidate)
            if isinstance(normalized, numbers.Number) and normalized > 0:
                candidates.append((key, float(normalized)))

        if candidates:
            if derived and derived > 0:
                key, best_value = min(candidates, key=lambda item: abs(item[1] - derived))
                if abs(best_value - derived) <= max(1.0, 0.15 * derived):
                    return float(best_value)
                return float(derived)
            return float(candidates[0][1])

    if derived and derived > 0:
        return float(derived)

    return float(AVATAR_FPS)


def _run_full_inference_capture(
    service,
    static_mode: bool,
    audio_waveform,
    audio_sample_rate: int,
    batch_size: int,
    fps: float,
):
    captured_frames: List[np.ndarray] = []

    def _collect_frame(frame):
        captured_frames.append(frame.copy())

    stats = service.process(
        face_path=AVATAR_IMAGE,
        audio_path="",
        output_path=None,
        static=static_mode,
        fps=fps,
        pads=FACE_PADS,
        audio_waveform=audio_waveform,
        audio_sample_rate=audio_sample_rate,
        batch_size_override=batch_size,
        frame_offset=0,
        frame_sink=_collect_frame,
    )
    return captured_frames, stats


def _encode_frames_sequence(frames_slice: List[np.ndarray], fps: float, output_path: Path):
    if not frames_slice:
        return 0, 0.0, ""

    height, width = frames_slice[0].shape[:2]
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
        "veryfast",
        "-crf",
        "18",
        "-movflags",
        "+faststart",
        str(output_path),
    ]

    encode_start = time.perf_counter()
    proc = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for frame in frames_slice:
            frame_to_write = frame
            if frame_to_write.dtype != np.uint8:
                frame_to_write = frame_to_write.astype(np.uint8, copy=False)
            if not frame_to_write.flags.c_contiguous:
                frame_to_write = np.ascontiguousarray(frame_to_write)
            if proc.stdin is not None:
                proc.stdin.write(memoryview(frame_to_write))
    except BrokenPipeError:
        stderr_data = proc.stderr.read().decode("utf-8", errors="ignore") if proc.stderr else ""
        raise RuntimeError(f"FFmpeg pipe broken during segment encode: {stderr_data}") from None
    finally:
        if proc.stdin is not None:
            proc.stdin.close()

    return_code = proc.wait()
    stderr_output = proc.stderr.read().decode("utf-8", errors="ignore") if proc.stderr else ""
    encode_time = time.perf_counter() - encode_start
    if proc.stderr is not None:
        proc.stderr.close()
    return return_code, encode_time, stderr_output


def _encode_segments_from_frames(
    frames: List[np.ndarray],
    expected_counts: List[int],
    frame_offsets: List[int],
    fps: float,
    segments_dir: Path,
    max_workers: Optional[int] = None,
):
    if not frames:
        return [], [], 0.0, 0, []

    non_empty_segments: List[Tuple[int, int, int]] = []
    total_frames = len(frames)
    for index, count in enumerate(expected_counts):
        if count <= 0:
            continue
        start = frame_offsets[index]
        end = min(start + count, total_frames)
        if end <= start:
            continue
        non_empty_segments.append((index, start, end))

    if not non_empty_segments:
        return [], [], 0.0, 0, []

    if max_workers is None or max_workers <= 0:
        cpu_workers = os.cpu_count() or 1
        max_workers = max(1, min(len(non_empty_segments), cpu_workers))

    results: Dict[int, Tuple[SegmentEncodingResult, Path]] = {}
    segment_paths: Dict[int, Path] = {}
    segment_stats: Dict[int, Dict[str, float]] = {}
    total_encode_time = 0.0

    first_frame = frames[0]
    frame_height, frame_width = first_frame.shape[:2]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {}
        for index, start, end in non_empty_segments:
            output_path = segments_dir / f"segment_{index:02d}.mp4"
            slice_frames = frames[start:end]
            future = executor.submit(_encode_frames_sequence, slice_frames, fps, output_path)
            future_map[future] = (index, start, end, output_path)

        for future in as_completed(future_map):
            index, start, end, output_path = future_map[future]
            return_code, encode_time, stderr_output = future.result()
            if return_code != 0:
                raise RuntimeError(f"Ошибка кодирования сегмента #{index + 1}: {stderr_output}")
            frame_count = end - start
            frame_start = start
            frame_end = end - 1
            segment_result = SegmentEncodingResult(
                index=index,
                frame_start=frame_start,
                frame_end=frame_end,
                frame_count=frame_count,
                write_time=encode_time,
                ffmpeg_time=encode_time,
                return_code=return_code,
                stderr=stderr_output,
            )
            results[index] = (segment_result, output_path)
            segment_paths[index] = output_path
            segment_stats[index] = {
                "num_frames": frame_count,
                "frame_height": frame_height,
                "frame_width": frame_width,
                "encode_time": encode_time,
            }
            total_encode_time += encode_time

    ordered_indices = sorted(results.keys())
    segment_results_ordered: List[SegmentEncodingResult] = []
    paths_ordered: List[Path] = []
    stats_list: List[Dict[str, float]] = []
    for idx in ordered_indices:
        segment_result, path_value = results[idx]
        segment_results_ordered.append(segment_result)
        paths_ordered.append(path_value)
        stats_list.append(segment_stats[idx])

    return (
        segment_results_ordered,
        paths_ordered,
        total_encode_time,
        max_workers if segment_results_ordered else 0,
        stats_list,
    )

def _partition_samples(total_samples: int, parts: int) -> List[Tuple[int, int]]:
    parts = max(1, parts)
    base = total_samples // parts
    remainder = total_samples % parts
    ranges: List[Tuple[int, int]] = []
    cursor = 0
    for index in range(parts):
        extra = 1 if index < remainder else 0
        start = cursor
        end = min(total_samples, cursor + base + extra)
        ranges.append((start, end))
        cursor = end
    return ranges


def _prepare_segment_layout(
    total_samples: int,
    segments_count: int,
    audio_sample_rate: int,
    fps: float,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    sample_ranges = _partition_samples(total_samples, segments_count)
    expected_total_frames = int(round((total_samples / audio_sample_rate) * fps))
    expected_counts: List[int] = []
    accumulated_expected = 0.0
    assigned_so_far = 0

    for start_sample, end_sample in sample_ranges:
        if start_sample >= end_sample:
            expected_counts.append(0)
            continue
        duration = (end_sample - start_sample) / audio_sample_rate
        accumulated_expected += duration * fps
        target_frames = int(round(accumulated_expected)) - assigned_so_far
        if target_frames < 0:
            target_frames = 0
        expected_counts.append(target_frames)
        assigned_so_far += target_frames

    if assigned_so_far < expected_total_frames:
        deficit = expected_total_frames - assigned_so_far
        for index in range(len(expected_counts) - 1, -1, -1):
            if sample_ranges[index][1] > sample_ranges[index][0]:
                expected_counts[index] += deficit
                assigned_so_far += deficit
                break

    frame_offsets: List[int] = []
    cumulative_offset = 0
    for count in expected_counts:
        frame_offsets.append(cumulative_offset)
        cumulative_offset += count

    return sample_ranges, expected_counts, frame_offsets


def _process_segments_streaming(
    services: List,
    static_mode: bool,
    audio_waveform,
    audio_sample_rate: int,
    batch_size: int,
    segments_count: int,
    fps: float,
    segments_dir: Path,
) -> Tuple[List[SegmentEncodingResult], Dict[str, float], List[Path], int, List[Dict[str, float]]]:
    if segments_count <= 0:
        raise ValueError("Количество сегментов должно быть положительным")

    total_samples = int(audio_waveform.shape[1])
    sample_ranges, expected_counts, frame_offsets = _prepare_segment_layout(
        total_samples, segments_count, audio_sample_rate, fps
    )

    jobs: queue.Queue = queue.Queue()
    for index, sample_range in enumerate(sample_ranges):
        jobs.put((index, sample_range, frame_offsets[index] if not static_mode else 0))
    # Sentinels for workers
    for _ in services:
        jobs.put(None)

    segment_infos: List[Optional[Dict[str, object]]] = [None] * len(sample_ranges)
    segment_paths: List[Optional[Path]] = [None] * len(sample_ranges)
    segment_stats: List[Optional[Dict[str, float]]] = [None] * len(sample_ranges)
    errors: List[Exception] = []
    errors_lock = threading.Lock()
    active_workers = 0
    active_workers_lock = threading.Lock()

    def _worker(worker_id: int, service) -> None:
        nonlocal active_workers
        _ensure_service_cuda_device(service)
        local_active = False
        while True:
            job = jobs.get()
            if job is None:
                jobs.task_done()
                break
            index, (start_sample, end_sample), frame_offset = job
            try:
                if start_sample >= end_sample:
                    segment_infos[index] = {
                        "frame_count": 0,
                        "write_time": 0.0,
                        "ffmpeg_time": 0.0,
                    }
                    segment_paths[index] = None
                    segment_stats[index] = None
                    continue

                waveform_slice = audio_waveform[:, start_sample:end_sample].contiguous()
                duration = (end_sample - start_sample) / float(audio_sample_rate)

                log_prefix = f"[segment {index + 1}/{len(sample_ranges)}]"
                print(
                    f"{log_prefix} ▶️ Старт инференса: {duration:.2f}s аудио, batch={batch_size}, offset={frame_offset}"
                )
                output_path = segments_dir / f"segment_{index:02d}.mp4"
                stats = service.process(
                    face_path=AVATAR_IMAGE,
                    audio_path="",
                    output_path=str(output_path),
                    static=static_mode,
                    fps=fps,
                    pads=FACE_PADS,
                    audio_waveform=waveform_slice,
                    audio_sample_rate=audio_sample_rate,
                    batch_size_override=batch_size,
                    frame_offset=frame_offset,
                )
                frame_count = int(stats.get("num_frames", 0))
                print(
                    f"{log_prefix} ✅ Инференс завершен: {frame_count} кадров, wall={stats.get('total_time', 0.0):.2f}s"
                )

                segment_infos[index] = {
                    "frame_count": frame_count,
                    "write_time": float(stats.get("writing_frames_time", 0.0)),
                    "ffmpeg_time": float(stats.get("ffmpeg_encoding_time", 0.0)),
                }
                segment_paths[index] = output_path
                segment_stats[index] = stats
                local_active = True
            except Exception as worker_error:
                with errors_lock:
                    errors.append(worker_error)
            finally:
                jobs.task_done()
        if local_active:
            with active_workers_lock:
                active_workers += 1

    threads: List[threading.Thread] = []
    for worker_index, service in enumerate(services):
        thread = threading.Thread(
            target=_worker,
            args=(worker_index, service),
            daemon=True,
        )
        threads.append(thread)
        thread.start()

    jobs.join()
    for thread in threads:
        thread.join()

    if errors:
        raise errors[0]

    segment_results: List[SegmentEncodingResult] = []
    paths_ordered: List[Path] = []
    stats_collected: List[Dict[str, float]] = []
    frame_cursor = 0

    for index, info in enumerate(segment_infos):
        if info is None:
            continue
        frame_count = int(info.get("frame_count", 0))
        if frame_count <= 0:
            continue
        frame_start = frame_cursor
        frame_end = frame_cursor + frame_count - 1
        frame_cursor += frame_count
        result = SegmentEncodingResult(
            index=index,
            frame_start=frame_start,
            frame_end=frame_end,
            frame_count=frame_count,
            write_time=float(info.get("write_time", 0.0)),
            ffmpeg_time=float(info.get("ffmpeg_time", 0.0)),
            return_code=0,
            stderr="",
        )
        segment_results.append(result)
        segment_path = segment_paths[index]
        if segment_path is not None:
            paths_ordered.append(segment_path)
        stats_entry = segment_stats[index]
        if stats_entry is not None:
            stats_entry = dict(stats_entry)
            stats_entry["num_frames"] = frame_count
            stats_collected.append(stats_entry)

    merged_stats = _merge_stats(stats_collected) if stats_collected else {}
    if not segment_results:
        active_workers = 0

    return segment_results, merged_stats, paths_ordered, active_workers, stats_collected


def _concat_segments(segment_paths: List[Path], output_path: Path) -> Tuple[int, str]:
    if not segment_paths:
        raise RuntimeError("Нет сегментов для склейки")

    concat_list = output_path.parent / "segments.txt"
    with open(concat_list, "w", encoding="utf-8") as handle:
        for path in segment_paths:
            resolved = path.resolve()
            handle.write(f"file '{resolved.as_posix()}'\n")

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


def _attach_audio(video_path: Path, audio_path: Path, output_path: Path) -> Tuple[int, str]:
    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-i",
        str(audio_path),
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-ar",
        "16000",
        "-b:a",
        "128k",
        "-movflags",
        "+faststart",
        "-shortest",
        str(output_path),
    ]
    proc = subprocess.run(command, capture_output=True, text=True)
    return proc.returncode, proc.stderr




def _merge_stats(stats_list: List[Dict[str, float]]) -> Dict[str, float]:
    numeric_totals: Dict[str, float] = {}
    numeric_max: Dict[str, float] = {}
    last_seen: Dict[str, float] = {}

    for stats in stats_list:
        for key, value in stats.items():
            normalized = _normalize_stat(value)
            if isinstance(normalized, numbers.Number):
                numeric_totals[key] = numeric_totals.get(key, 0.0) + float(normalized)
                numeric_max[key] = max(numeric_max.get(key, float(normalized)), float(normalized))
            else:
                last_seen[key] = normalized

    merged: Dict[str, float] = {key: float(val) for key, val in numeric_totals.items()}
    for key, val in last_seen.items():
        merged[key] = val
    for key, val in numeric_max.items():
        merged[f"{key}_max"] = val
    return merged


def _probe_video_resolution(path: Path) -> Tuple[int, int]:
    try:
        import cv2  # type: ignore
    except ImportError:
        return (0, 0)

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        return (0, 0)

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if height <= 0 or width <= 0:
        success, frame = capture.read()
        if success and frame is not None:
            height, width = frame.shape[:2]
    capture.release()
    return (height if height > 0 else 0, width if width > 0 else 0)
