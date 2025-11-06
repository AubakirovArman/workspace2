"""Low-latency HLS streaming pipeline for dynamic avatar generation."""
from __future__ import annotations

import json
import queue
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from .. import state
from ..config import AVATAR_FPS, AVATAR_IMAGE, AVATAR_VIDEO_PATH, TEMP_DIR
from .tts import convert_to_wav, generate_tts

BASE_STREAM_DIR = Path(TEMP_DIR) / "hls"
BASE_STREAM_DIR.mkdir(parents=True, exist_ok=True)

PADS = (0, 50, 0, 0)
DEFAULT_BATCH_SIZE = 32

ENCODER_THREADS = 8
SEGMENT_DURATION = 1.0  # seconds
SEGMENT_LIST_SIZE = 6
HLS_FLAGS = "independent_segments+append_list+omit_endlist+delete_segments"


def _coerce_value(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {k: _coerce_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_coerce_value(v) for v in value]
    return value


class HLSStreamEncoder:
    """Incrementally feeds frames into an ffmpeg HLS+MP4 tee pipeline."""

    def __init__(
        self,
        audio_path: Path,
        playlist_path: Path,
        segment_template: Path,
        mp4_path: Path,
        fps: float,
        crf: int = 20,
        preset: str = "veryfast",
    ) -> None:
        self.audio_path = audio_path
        self.playlist_path = playlist_path
        self.segment_template = segment_template
        self.mp4_path = mp4_path
        self.fps = fps
        self.crf = crf
        self.preset = preset

        self._queue: "queue.Queue[Optional[np.ndarray]]" = queue.Queue(maxsize=256)
        self._writer_thread = threading.Thread(target=self._write_loop, daemon=True)
        self._proc: Optional[subprocess.Popen] = None
        self._width: Optional[int] = None
        self._height: Optional[int] = None
        self._start_monotonic: Optional[float] = None
        self._frame_count = 0
        self._lock = threading.Lock()
        self._error: Optional[str] = None
        self._encode_time = 0.0

    @property
    def error(self) -> Optional[str]:
        return self._error

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def encode_time(self) -> float:
        return self._encode_time

    def start(self) -> None:
        self._writer_thread.start()

    def push_frame(self, frame: np.ndarray) -> None:
        if self._error:
            return
        try:
            while True:
                try:
                    self._queue.put(frame.copy(), timeout=0.5)
                    break
                except queue.Full:
                    try:
                        _ = self._queue.get_nowait()
                    except queue.Empty:
                        pass
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                self._error = str(exc)

    def finish(self) -> None:
        try:
            self._queue.put(None, timeout=1.0)
        except Exception:
            pass
        self._writer_thread.join()

    def _launch_encoder(self, width: int, height: int) -> None:
        playlist_parent = self.playlist_path.parent
        playlist_parent.mkdir(parents=True, exist_ok=True)

        hls_spec = (
            f"[f=hls:hls_time={SEGMENT_DURATION}:hls_segment_type=fmp4:"
            f"hls_flags={HLS_FLAGS}:hls_list_size={SEGMENT_LIST_SIZE}:"
            f"hls_segment_filename={self.segment_template}]"
            f"{self.playlist_path}"
        )
        tee_spec = f"{hls_spec}|[f=mp4]{self.mp4_path}"

        command = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-nostdin",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{width}x{height}",
            "-r",
            f"{self.fps}",
            "-i",
            "pipe:0",
            "-thread_queue_size",
            "2048",
            "-i",
            str(self.audio_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-shortest",
            "-c:v",
            "libx264",
            "-preset",
            self.preset,
            "-crf",
            str(self.crf),
            "-tune",
            "zerolatency",
            "-g",
            f"{int(self.fps)}",
            "-keyint_min",
            f"{int(self.fps)}",
            "-sc_threshold",
            "0",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-ar",
            "48000",
            "-b:a",
            "128k",
            "-threads",
            str(ENCODER_THREADS),
            "-f",
            "tee",
            tee_spec,
        ]

        self._proc = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def _write_loop(self) -> None:
        frame_interval = 1.0 / max(self.fps, 1e-3)
        next_deadline = None
        proc: Optional[subprocess.Popen] = None
        start_time = time.perf_counter()

        while True:
            item = self._queue.get()
            if item is None:
                break

            frame = item
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8, copy=False)

            height, width = frame.shape[:2]
            if proc is None:
                try:
                    self._launch_encoder(width, height)
                except Exception as exc:  # noqa: BLE001
                    with self._lock:
                        self._error = f"Failed to start encoder: {exc}"
                    return
                proc = self._proc
                self._width = width
                self._height = height
                next_deadline = time.perf_counter()
                self._start_monotonic = start_time

            if proc is None or proc.stdin is None:
                with self._lock:
                    self._error = "FFmpeg process not available"
                return

            try:
                proc.stdin.write(frame.tobytes())
            except Exception as exc:  # noqa: BLE001
                with self._lock:
                    self._error = f"Encoder write error: {exc}"
                return

            self._frame_count += 1
            if next_deadline is None:
                next_deadline = time.perf_counter()
            next_deadline += frame_interval
            sleep_time = next_deadline - time.perf_counter()
            if sleep_time > 0:
                time.sleep(min(sleep_time, frame_interval))

        if proc and proc.stdin:
            try:
                proc.stdin.close()
            except Exception:
                pass
            try:
                proc.wait(timeout=30.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        if proc and proc.stderr:
            try:
                stderr_output = proc.stderr.read().decode("utf-8", errors="ignore")
                if stderr_output.strip():
                    with self._lock:
                        self._error = stderr_output.strip()
            except Exception:
                pass
            finally:
                try:
                    proc.stderr.close()
                except Exception:
                    pass
        end_time = time.perf_counter()
        if self._start_monotonic is not None:
            self._encode_time = end_time - self._start_monotonic


@dataclass
class StreamJob:
    session_id: str
    session_dir: Path
    playlist_path: Path
    segment_template: Path
    mp4_path: Path
    meta_path: Path
    status: str = "initializing"
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    fps: float = AVATAR_FPS
    dynamic: bool = True
    summary: Dict[str, Any] = field(default_factory=dict)
    thread: Optional[threading.Thread] = None

    def playlist_url(self) -> str:
        return f"/stream/{self.session_id}/playlist.m3u8"

    def mp4_url(self) -> Optional[str]:
        if self.mp4_path.exists() and self.status == "ready":
            return f"/stream/{self.session_id}/{self.mp4_path.name}"
        return None


_STREAM_JOBS: Dict[str, StreamJob] = {}
_JOBS_LOCK = threading.Lock()


def _register_job(job: StreamJob) -> None:
    with _JOBS_LOCK:
        _STREAM_JOBS[job.session_id] = job


def get_stream_job(session_id: str) -> Optional[StreamJob]:
    with _JOBS_LOCK:
        return _STREAM_JOBS.get(session_id)


def _update_job(job: StreamJob, **kwargs: Any) -> None:
    with _JOBS_LOCK:
        for key, value in kwargs.items():
            setattr(job, key, value)


def _run_stream_job(job: StreamJob, text: str, language: str) -> None:
    try:
        job.session_dir.mkdir(parents=True, exist_ok=True)
        if not job.playlist_path.exists():
            initial_playlist = "\n".join([
                "#EXTM3U",
                "#EXT-X-VERSION:7",
                "#EXT-X-TARGETDURATION:2",
                "#EXT-X-PLAYLIST-TYPE:EVENT",
                "#EXT-X-MEDIA-SEQUENCE:0",
            ])
            job.playlist_path.write_text(initial_playlist + "\n", encoding="utf-8")
        _update_job(job, status="tts")
        tts_start = time.perf_counter()
        audio_data = generate_tts(text, language)
        tts_time = time.perf_counter() - tts_start

        _update_job(job, status="audio")
        audio_path = job.session_dir / "audio.wav"
        convert_start = time.perf_counter()
        waveform, sample_rate = convert_to_wav(audio_data, str(audio_path))
        convert_time = time.perf_counter() - convert_start

        service = state.lipsync_service_gan or state.lipsync_service_nogan
        if service is None:
            raise RuntimeError("Lipsync service is not initialized")

        fps = job.fps if job.fps > 0 else AVATAR_FPS
        encoder = HLSStreamEncoder(
            audio_path=audio_path,
            playlist_path=job.playlist_path,
            segment_template=job.segment_template,
            mp4_path=job.mp4_path,
            fps=fps,
        )
        encoder.start()

        face_path = AVATAR_VIDEO_PATH if job.dynamic else AVATAR_IMAGE
        start_total = time.perf_counter()
        _update_job(job, status="inference")

        def _frame_sink(frame: np.ndarray) -> None:
            encoder.push_frame(frame)

        try:
            stats = service.process(
                face_path=str(face_path),
                audio_path="",
                output_path=None,
                static=not job.dynamic,
                fps=fps,
                pads=PADS,
                audio_waveform=waveform,
                audio_sample_rate=sample_rate,
                frame_sink=_frame_sink,
                batch_size_override=DEFAULT_BATCH_SIZE,
                frame_offset=0,
            )
        finally:
            _update_job(job, status="finalizing")
            encoder.finish()
        encode_error = encoder.error
        if encode_error:
            raise RuntimeError(encode_error)

        total_time = time.perf_counter() - start_total

        summary = {
            "timestamp": time.time(),
            "session_id": job.session_id,
            "playlist": job.playlist_url(),
            "mp4": job.mp4_url(),
            "tts_time": tts_time,
            "audio_convert_time": convert_time,
            "fps": fps,
            "frames": stats.get("num_frames", encoder.frame_count),
            "encode_time": encoder.encode_time,
            "stats": {k: _coerce_value(v) for k, v in stats.items()} if isinstance(stats, dict) else _coerce_value(stats),
            "total_time": total_time,
        }
        job.meta_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        _update_job(job, status="ready", summary=summary, finished_at=time.time())
    except Exception as exc:  # noqa: BLE001
        _update_job(job, status="error", error=str(exc), finished_at=time.time())


def start_stream_job(text: str, language: str, dynamic: bool = True) -> StreamJob:
    text = (text or "").strip()
    if not text:
        raise ValueError("Text must not be empty")

    language = (language or "ru").strip().lower()
    if language not in {"ru", "en", "kk"}:
        raise ValueError("Unsupported language")

    session_id = uuid.uuid4().hex
    session_dir = BASE_STREAM_DIR / session_id
    playlist_path = session_dir / "playlist.m3u8"
    segment_template = session_dir / "segment_%05d.m4s"
    mp4_path = session_dir / "final.mp4"
    meta_path = session_dir / "summary.json"

    job = StreamJob(
        session_id=session_id,
        session_dir=session_dir,
        playlist_path=playlist_path,
        segment_template=segment_template,
        mp4_path=mp4_path,
        meta_path=meta_path,
        dynamic=dynamic,
    )
    _register_job(job)

    thread = threading.Thread(
        target=_run_stream_job,
        args=(job, text, language),
        name=f"hls-stream-{session_id[:8]}",
        daemon=True,
    )
    job.thread = thread
    thread.start()
    return job


__all__ = ["start_stream_job", "get_stream_job"]
