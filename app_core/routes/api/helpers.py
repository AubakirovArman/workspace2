"""Shared helpers for API routes."""
from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple
import threading

import numpy as np
import torchaudio
import torch

from ...config import AVATAR_FPS, AVATAR_IMAGE

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}

DEFAULT_BATCH_SIZE = 4096


def avatar_supports_dynamic() -> bool:
    return Path(AVATAR_IMAGE).suffix.lower() in VIDEO_EXTENSIONS


def coerce_optional_bool(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    raise ValueError("Не удалось разобрать значение static_mode")


def generate_frames_single(
    service,
    static_mode: bool,
    audio_waveform,
    audio_sample_rate: int,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Tuple[List, Dict]:
    """Generate frames using a single inference service."""
    frames: List = []

    def _sink(frame) -> None:
        frames.append(frame)

    samples = int(audio_waveform.shape[1])
    duration = samples / float(audio_sample_rate)
    print(f"[single] ▶️ Старт инференса: {duration:.2f}s аудио, batch={batch_size}")
    stats = service.process(
        face_path=AVATAR_IMAGE,
        audio_path="",
        output_path=None,
        static=static_mode,
        fps=AVATAR_FPS,
        pads=(0, 50, 0, 0),
        audio_waveform=audio_waveform,
        audio_sample_rate=audio_sample_rate,
        frame_sink=_sink,
        batch_size_override=batch_size,
    )
    print(f"[single] ✅ Инференс завершен: {len(frames)} кадров, wall={stats.get('total_time', 0.0):.2f}s")
    return frames, stats or {}


def generate_frames_parallel(
    service_pool: List,
    static_mode: bool,
    audio_waveform,
    audio_sample_rate: int,
    desired_chunks: int,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Tuple[List, Dict, int]:
    """Generate frames in parallel across several inference services."""
    def _ensure_cuda_device(svc) -> None:
        if not torch.cuda.is_available():
            return
        target_device = getattr(svc, "device", None)
        if target_device is None:
            return
        try:
            torch.cuda.set_device(torch.device(str(target_device)))
        except Exception:
            pass

    total_samples = int(audio_waveform.shape[1])
    chunks = max(1, int(desired_chunks))
    base = total_samples // chunks
    remainder = total_samples % chunks

    sample_ranges = []
    cursor = 0
    for index in range(chunks):
        extra = 1 if index < remainder else 0
        start = cursor
        end = min(total_samples, cursor + base + extra)
        sample_ranges.append((start, end))
        cursor = end

    expected_total_frames = int(round((total_samples / audio_sample_rate) * AVATAR_FPS))
    expected_counts: List[int] = []
    accumulated_expected = 0.0
    assigned_so_far = 0
    for start_sample, end_sample in sample_ranges:
        if start_sample >= end_sample:
            expected_counts.append(0)
            continue
        duration = (end_sample - start_sample) / audio_sample_rate
        accumulated_expected += duration * AVATAR_FPS
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

    frames_per_chunk = [None] * len(sample_ranges)
    stats_per_chunk = [None] * len(sample_ranges)
    errors = []
    errors_lock = threading.Lock()
    threads = []

    active_chunks = 0

    for index, (start_sample, end_sample) in enumerate(sample_ranges):
        if start_sample >= end_sample:
            continue
        active_chunks += 1
        service = service_pool[index % len(service_pool)]
        waveform_slice = audio_waveform[:, start_sample:end_sample].contiguous()

        def _worker(idx: int, svc, chunk_waveform, chunk_offset: int):
            try:
                _ensure_cuda_device(svc)
                chunk_frames = []

                def _sink(frame) -> None:
                    chunk_frames.append(frame)

                chunk_samples = int(chunk_waveform.shape[1])
                chunk_duration = chunk_samples / float(audio_sample_rate)
                label = f"[chunk {idx + 1}/{len(sample_ranges)}]"
                print(
                    f"{label} ▶️ Старт инференса: {chunk_duration:.2f}s аудио, batch={batch_size}, offset={chunk_offset}"
                )
                chunk_stats = svc.process(
                    face_path=AVATAR_IMAGE,
                    audio_path="",
                    output_path=None,
                    static=static_mode,
                    fps=AVATAR_FPS,
                    pads=(0, 50, 0, 0),
                    audio_waveform=chunk_waveform,
                    audio_sample_rate=audio_sample_rate,
                    frame_sink=_sink,
                    batch_size_override=batch_size,
                    frame_offset=chunk_offset,
                )
                print(
                    f"{label} ✅ Инференс завершен: {len(chunk_frames)} кадров, wall={chunk_stats.get('total_time', 0.0):.2f}s"
                )
                frames_per_chunk[idx] = chunk_frames
                stats_per_chunk[idx] = chunk_stats
            except Exception as worker_error:
                with errors_lock:
                    errors.append(worker_error)

        thread = threading.Thread(
            target=_worker,
            args=(index, service, waveform_slice, frame_offsets[index] if not static_mode else 0),
            daemon=True,
        )
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    if errors:
        raise errors[0]

    ordered_frames = []
    collected_stats = []
    dropped_total = 0

    for index, chunk in enumerate(frames_per_chunk):
        if not chunk:
            continue
        target = expected_counts[index] if index < len(expected_counts) else len(chunk)
        if target <= 0:
            trimmed_chunk = []
            target = 0
        elif len(chunk) > target:
            dropped_total += len(chunk) - target
            trimmed_chunk = chunk[:target]
        else:
            trimmed_chunk = chunk
            target = len(chunk)
        ordered_frames.extend(trimmed_chunk)
        stats_entry = stats_per_chunk[index]
        actual_count = len(trimmed_chunk)
        if stats_entry is not None:
            stats_entry['num_frames'] = actual_count
            if len(chunk) > actual_count:
                stats_entry['dropped_frames'] = len(chunk) - actual_count
        if stats_entry:
            collected_stats.append(stats_entry)

    if dropped_total > 0:
        print(f"⚙️ Коррекция синхронизации: удалено {dropped_total} лишних кадров при параллельной генерации")

    merged_stats = {}
    if collected_stats:
        for stats in collected_stats:
            for key, value in stats.items():
                if isinstance(value, (int, float)):
                    merged_stats[key] = merged_stats.get(key, 0.0) + float(value)

    return ordered_frames, merged_stats, active_chunks


def encode_video_with_audio(
    frames: List,
    output_path: str,
    audio_waveform,
    audio_sample_rate: int,
    fps: float,
    codec_service,
    segments: int = 1,
) -> None:
    """Encode frames into an MP4 container with audio."""
    if not frames:
        raise RuntimeError("Нет кадров для кодирования")

    height, width = frames[0].shape[:2]
    codec_name, codec_opts = codec_service._select_ffmpeg_codec()

    temp_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    temp_audio.close()
    try:
        torchaudio.save(temp_audio.name, audio_waveform.cpu(), audio_sample_rate, backend="sox_io")
    except Exception:
        write_waveform_to_wav(temp_audio.name, audio_waveform, audio_sample_rate)

    segments = max(1, int(segments))
    
    # Debug: проверяем количество кадров
    print(f"🔍 DEBUG encode_video_with_audio:")
    print(f"   Кадров для кодирования: {len(frames)}")
    print(f"   Аудио: {temp_audio.name}")
    print(f"   FPS: {fps}")
    print(f"   Разрешение: {width}x{height}")
    print(f"   Сегменты: {segments}")
    
    try:
        if segments > 1:
            _encode_multi_segment(
                frames,
                output_path,
                temp_audio.name,
                fps,
                codec_name,
                list(codec_opts),
                (height, width),
                segments,
            )
        else:
            _encode_single_segment(
                frames,
                output_path,
                temp_audio.name,
                fps,
                codec_name,
                list(codec_opts),
                (height, width),
            )
    finally:
        try:
            os.unlink(temp_audio.name)
        except OSError:
            pass


def write_waveform_to_wav(destination: str, audio_waveform, audio_sample_rate: int) -> None:
    """Persist waveform tensor as PCM16 WAV with manual fallback."""
    import wave

    tensor = audio_waveform.detach().cpu()
    audio_np = tensor.numpy()
    if audio_np.ndim == 1:
        audio_np = audio_np[None, :]
    clipped = np.clip(audio_np.copy(), -1.0, 1.0)
    pcm16 = (clipped * 32767.0).round().astype(np.int16)
    interleaved = pcm16.T.reshape(-1)
    with wave.open(destination, "wb") as wav_file:
        wav_file.setnchannels(pcm16.shape[0])
        wav_file.setsampwidth(2)
        wav_file.setframerate(audio_sample_rate)
        wav_file.writeframes(interleaved.tobytes())


def _encode_single_segment(
    frames: List,
    output_path: str,
    audio_path: str,
    fps: float,
    codec_name: str,
    codec_opts: List[str],
    resolution: Tuple[int, int],
) -> None:
    height, width = resolution
    dimension = f"{width}x{height}"
    blocksize = width * height * 3
    vf_filters = []
    if width % 2 != 0 or height % 2 != 0:
        vf_filters.append("scale=w=trunc(iw/2)*2:h=trunc(ih/2)*2:flags=fast_bilinear")

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
        dimension,
        "-r",
        str(fps),
        "-blocksize",
        str(blocksize),
        "-i",
        "pipe:0",
        "-i",
        audio_path,
        "-c:v",
        codec_name,
    ]
    command += codec_opts
    if vf_filters:
        command += ["-vf", ",".join(vf_filters)]
    command += [
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-ar",
        "16000",
        "-b:a",
        "128k",
        "-shortest",
        "-movflags",
        "+faststart",
        output_path,
    ]

    proc = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.stdin is None:
        raise RuntimeError("FFmpeg stdin недоступен")

    frames_written = 0
    try:
        for frame in frames:
            array = np.asarray(frame, dtype=np.uint8, order="C")
            proc.stdin.write(array.tobytes())
            frames_written += 1
    except BrokenPipeError:
        # FFmpeg закрылся раньше времени — читаем stderr для диагностики
        proc.stdin.close()
        proc.wait()
        stderr_output = proc.stderr.read().decode("utf-8", errors="ignore") if proc.stderr else ""
        print(f"🔍 DEBUG BrokenPipe:")
        print(f"   Записано кадров: {frames_written}/{len(frames)}")
        print(f"   FFmpeg stderr: {stderr_output or '(пусто)'}")
        raise RuntimeError(f"FFmpeg closed prematurely after {frames_written}/{len(frames)} frames. stderr: {stderr_output or '(empty)'}")
    finally:
        try:
            proc.stdin.close()
        except Exception:
            pass

    returncode = proc.wait()
    stderr_output = proc.stderr.read().decode("utf-8", errors="ignore") if proc.stderr else ""
    if returncode != 0:
        raise RuntimeError(f"FFmpeg encoding failed (rc={returncode}): {stderr_output}")


def _encode_multi_segment(
    frames: List,
    output_path: str,
    audio_path: str,
    fps: float,
    codec_name: str,
    codec_opts: List[str],
    resolution: Tuple[int, int],
    segments: int,
) -> None:
    total_frames = len(frames)
    if total_frames == 0:
        raise RuntimeError("Нет кадров для кодирования")

    height, width = resolution
    segments = min(max(1, segments), total_frames)
    frame_ranges = _partition_frames(total_frames, segments)

    errors: List[str] = []
    errors_lock = threading.Lock()
    segment_paths: List[Path] = []
    threads: List[threading.Thread] = []

    with tempfile.TemporaryDirectory(prefix="encode_segments_") as temp_dir:
        temp_root = Path(temp_dir)

        for index, (start_frame, end_frame) in enumerate(frame_ranges):
            if start_frame >= end_frame:
                continue
            segment_frames = frames[start_frame:end_frame]
            segment_path = temp_root / f"segment_{index:03d}.mp4"
            cmd = _build_ffmpeg_command(codec_name, codec_opts, width, height, fps, segment_path)
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

            def _worker(idx: int, chunk_frames: List, popen: subprocess.Popen) -> None:
                try:
                    _write_segment(popen, chunk_frames)
                    wait_rc = popen.wait()
                    stderr_text = popen.stderr.read().decode("utf-8", errors="ignore") if popen.stderr else ""
                    if wait_rc != 0:
                        message = f"FFmpeg segment {idx} failed (rc={wait_rc}): {stderr_text}"
                        with errors_lock:
                            errors.append(message)
                except Exception as encode_error:
                    with errors_lock:
                        errors.append(str(encode_error))

            thread = threading.Thread(
                target=_worker,
                args=(index, segment_frames, proc),
                daemon=True,
            )
            segment_paths.append(segment_path)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        if errors:
            raise RuntimeError(errors[0])

        concat_list = temp_root / "segments.txt"
        with open(concat_list, "w", encoding="utf-8") as handle:
            for path in segment_paths:
                handle.write(f"file '{path.resolve().as_posix()}'\n")

        merged_path = temp_root / "merged_video.mp4"
        concat_cmd = [
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
            str(merged_path),
        ]
        concat_proc = subprocess.run(concat_cmd, capture_output=True, text=True)
        if concat_proc.returncode != 0:
            raise RuntimeError(f"FFmpeg concat failed: {concat_proc.stderr}")

        mux_cmd = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(merged_path),
            "-i",
            audio_path,
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
            output_path,
        ]
        mux_proc = subprocess.run(mux_cmd, capture_output=True, text=True)
        if mux_proc.returncode != 0:
            raise RuntimeError(f"FFmpeg mux failed: {mux_proc.stderr}")


def _partition_frames(total_frames: int, segments: int) -> List[Tuple[int, int]]:
    segments = max(1, segments)
    base = total_frames // segments
    remainder = total_frames % segments
    ranges: List[Tuple[int, int]] = []
    cursor = 0
    for index in range(segments):
        extra = 1 if index < remainder else 0
        start = cursor
        end = min(total_frames, cursor + base + extra)
        ranges.append((start, end))
        cursor = end
    return ranges


def _build_ffmpeg_command(
    codec_name: str,
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
        dimension,
        "-r",
        str(fps),
        "-blocksize",
        str(blocksize),
        "-i",
        "pipe:0",
        "-c:v",
        codec_name,
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


def _write_segment(proc: subprocess.Popen, frames: List) -> None:
    if proc.stdin is None:
        raise RuntimeError("FFmpeg stdin недоступен")
    for frame in frames:
        array = np.asarray(frame, dtype=np.uint8, order="C")
        proc.stdin.write(array.tobytes())
    proc.stdin.close()
