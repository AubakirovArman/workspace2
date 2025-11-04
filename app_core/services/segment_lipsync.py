"""Segmented lipsync generation helpers."""
from __future__ import annotations

import json
import numbers
import subprocess
import threading
import time
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
                    fps=AVATAR_FPS,
                    pads=FACE_PADS,
                )
            else:
                svc.preload_video_cache(
                    face_path=AVATAR_IMAGE,
                    fps=AVATAR_FPS,
                    pads=FACE_PADS,
                )
        except Exception as preload_error:
            suffix = f" #{index}" if len(service_pool) > 1 else ""
            print(f"⚠️ Не удалось подготовить аватар для сервиса{suffix}: {preload_error}")

    start_capture = time.perf_counter()
    if len(service_pool) == 1:
        frames, raw_stats = _capture_frames_single(
            primary_service,
            static_mode=static_mode,
            audio_waveform=audio_waveform,
            audio_sample_rate=audio_sample_rate,
            batch_size=batch_size,
        )
        active_chunks = 1
    else:
        frames, raw_stats, active_chunks = _capture_frames_parallel(
            service_pool,
            static_mode=static_mode,
            audio_waveform=audio_waveform,
            audio_sample_rate=audio_sample_rate,
            batch_size=batch_size,
            chunks=len(service_pool),
        )
    capture_time = time.perf_counter() - start_capture

    total_frames = len(frames)
    if total_frames == 0:
        raise RuntimeError("Не удалось получить кадры для кодирования")

    requested_segments = segments if segments > 0 else None
    target_segments = segments if segments > 0 else 16
    safe_segments = max(1, min(total_frames, target_segments))

    if requested_segments is None:
        print(
            f"🎯 Авторазбиение: длина аудио {audio_duration_seconds:.2f}s, кадров {total_frames}, сегменты {safe_segments}"
        )
    else:
        print(
            f"🎯 Разбиение аудио: запрос {requested_segments}, использовано {safe_segments} сегментов (кадров {total_frames})"
        )

    codec_name, codec_opts = primary_service._select_ffmpeg_codec()  # type: ignore[attr-defined]

    start_encode = time.perf_counter()
    segment_results, total_encode_time, segment_paths = _encode_segments(
        frames,
        AVATAR_FPS,
        codec_name,
        codec_opts,
        segments_dir,
        safe_segments,
    )

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
    inference_time_wall = float(
        normalized_stats.get("inference_time_max")
        or normalized_stats.get("inference_time")
        or normalized_stats.get("process_time")
        or 0.0
    )

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
        resolution=frames[0].shape[:2],
        timings=timings,
        segment_results=segment_results,
        inference_stats=normalized_stats,
        capture_workers=len(service_pool),
        capture_chunks=active_chunks if len(service_pool) > 1 else 1,
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


def _partition_frames(total_frames: int, segments_count: int) -> List[Tuple[int, int]]:
    segments_count = max(1, segments_count)
    base = total_frames // segments_count
    remainder = total_frames % segments_count
    ranges: List[Tuple[int, int]] = []
    cursor = 0
    for index in range(segments_count):
        extra = 1 if index < remainder else 0
        start = cursor
        end = min(total_frames, cursor + base + extra)
        ranges.append((start, end))
        cursor = end
    return ranges


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


def _build_ffmpeg_command(
    codec: str,
    codec_opts: List[str],
    width: int,
    height: int,
    fps: float,
    output_path: Path,
) -> List[str]:
    dimension = f"{width}x{height}"
    blocksize = width * height * 3
    vf_filters: List[str] = []
    if width % 2 != 0 or height % 2 != 0:
        vf_filters.append("scale=w=trunc(iw/2)*2:h=trunc(ih/2)*2:flags=fast_bilinear")
    command: List[str] = [
        "ffmpeg",
        "-y",
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
        "-c:v",
        codec,
    ]
    command += codec_opts
    if vf_filters:
        command += ["-vf", ",".join(vf_filters)]
    command += [
        "-pix_fmt",
        "yuv420p",
        "-an",
        str(output_path),
    ]
    return command


def _write_segment(proc: subprocess.Popen, frames: List[np.ndarray]) -> float:
    if proc.stdin is None:
        raise RuntimeError("FFmpeg stdin не доступен")
    start = time.perf_counter()
    for frame in frames:
        array = np.asarray(frame, dtype=np.uint8, order="C")
        if array.ndim != 3:
            raise ValueError("Неверный формат кадра для сегментации")
        proc.stdin.write(array.tobytes())
    proc.stdin.close()
    return time.perf_counter() - start


def _encode_segments(
    frames: List[np.ndarray],
    fps: float,
    codec: str,
    codec_opts: List[str],
    output_dir: Path,
    segments_count: int,
) -> Tuple[List[SegmentEncodingResult], float, List[Path]]:
    total_frames = len(frames)
    if total_frames == 0:
        raise RuntimeError("Нет кадров для кодирования")

    height, width = frames[0].shape[:2]
    frame_ranges = _partition_frames(total_frames, segments_count)

    results: List[SegmentEncodingResult] = []
    segment_paths: List[Path] = []
    threads: List[threading.Thread] = []
    results_lock = threading.Lock()
    start_global = time.perf_counter()

    for index, (start_frame, end_frame) in enumerate(frame_ranges):
        if start_frame >= end_frame:
            continue
        segment_frames = frames[start_frame:end_frame]
        output_path = output_dir / f"segment_{index:02d}.mp4"
        cmd = _build_ffmpeg_command(codec, codec_opts, width, height, fps, output_path)
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

        def _worker(idx: int, base: int, chunk: List[np.ndarray], popen: subprocess.Popen) -> None:
            write_time = _write_segment(popen, chunk)
            wait_start = time.perf_counter()
            return_code = popen.wait()
            ffmpeg_time = time.perf_counter() - wait_start
            stderr_data = popen.stderr.read().decode("utf-8", errors="ignore") if popen.stderr else ""
            result = SegmentEncodingResult(
                index=idx,
                frame_start=base,
                frame_end=base + len(chunk) - 1,
                frame_count=len(chunk),
                write_time=write_time,
                ffmpeg_time=ffmpeg_time,
                return_code=return_code,
                stderr=stderr_data,
            )
            with results_lock:
                results.append(result)

        thread = threading.Thread(
            target=_worker,
            args=(index, start_frame, segment_frames, proc),
            daemon=True,
        )
        segment_paths.append(output_path)
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    total_encode_time = time.perf_counter() - start_global
    results.sort(key=lambda item: item.index)
    return results, total_encode_time, segment_paths


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


def _capture_frames_single(
    service,
    static_mode: bool,
    audio_waveform,
    audio_sample_rate: int,
    batch_size: int,
) -> Tuple[List[np.ndarray], Dict[str, float]]:
    _ensure_service_cuda_device(service)

    frames: List[np.ndarray] = []

    def _sink(frame: np.ndarray) -> None:
        frames.append(frame.copy())

    stats = service.process(
        face_path=AVATAR_IMAGE,
        audio_path="",
        output_path=None,
        static=static_mode,
        fps=AVATAR_FPS,
        pads=FACE_PADS,
        audio_waveform=audio_waveform,
        audio_sample_rate=audio_sample_rate,
        frame_sink=_sink,
        batch_size_override=batch_size,
    )
    return frames, stats or {}


def _capture_frames_parallel(
    services: List,
    static_mode: bool,
    audio_waveform,
    audio_sample_rate: int,
    batch_size: int,
    chunks: int,
) -> Tuple[List[np.ndarray], Dict[str, float], int]:
    total_samples = int(audio_waveform.shape[1])
    sample_ranges = _partition_samples(total_samples, chunks)

    frames_per_chunk: List[Optional[List[np.ndarray]]] = [None] * len(sample_ranges)
    stats_per_chunk: List[Optional[Dict[str, float]]] = [None] * len(sample_ranges)
    errors: List[Exception] = []
    errors_lock = threading.Lock()
    threads: List[threading.Thread] = []

    active_chunks = 0

    for index, (start_sample, end_sample) in enumerate(sample_ranges):
        if start_sample >= end_sample:
            continue
        active_chunks += 1
        service = services[index % len(services)]
        waveform_slice = audio_waveform[:, start_sample:end_sample].contiguous()

        def _worker(idx: int, svc, chunk_waveform):
            try:
                chunk_frames, chunk_stats = _capture_frames_single(
                    svc,
                    static_mode=static_mode,
                    audio_waveform=chunk_waveform,
                    audio_sample_rate=audio_sample_rate,
                    batch_size=batch_size,
                )
                frames_per_chunk[idx] = chunk_frames
                stats_per_chunk[idx] = chunk_stats
            except Exception as worker_error:  # pragma: no cover - runtime safeguard
                with errors_lock:
                    errors.append(worker_error)

        thread = threading.Thread(
            target=_worker,
            args=(index, service, waveform_slice),
            daemon=True,
        )
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    if errors:
        raise errors[0]

    ordered_frames: List[np.ndarray] = []
    collected_stats: List[Dict[str, float]] = []
    for chunk in frames_per_chunk:
        if chunk:
            ordered_frames.extend(chunk)
    for stats in stats_per_chunk:
        if stats:
            collected_stats.append(stats)

    merged_stats = _merge_stats(collected_stats) if collected_stats else {}
    return ordered_frames, merged_stats, active_chunks


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
