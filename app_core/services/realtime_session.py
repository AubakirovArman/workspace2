"""Support for realtime WebRTC streaming with chunked TTS audio."""
from __future__ import annotations

import asyncio
import os
import queue
import re
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from threading import Thread
from typing import Any, Coroutine, Iterable, List, Optional, Tuple, TypeVar

import numpy as np
import torch
import torchaudio
import torchaudio.functional as audio_fn
try:
    from aiortc import (
        RTCPeerConnection,
        RTCSessionDescription,
        RTCConfiguration,
        RTCIceServer,
    )
    from aiortc.mediastreams import AudioStreamTrack, MediaStreamError, VideoStreamTrack
    AIORTC_IMPORT_ERROR: Optional[ImportError] = None
except ImportError as import_error:  # pragma: no cover - optional dependency guard
    RTCPeerConnection = None  # type: ignore[assignment]
    RTCSessionDescription = None  # type: ignore[assignment]
    RTCConfiguration = None  # type: ignore[assignment]
    RTCIceServer = None  # type: ignore[assignment]
    MediaStreamError = Exception  # type: ignore[assignment]
    AudioStreamTrack = object  # type: ignore[assignment]
    VideoStreamTrack = object  # type: ignore[assignment]
    AIORTC_IMPORT_ERROR = import_error

try:
    from av import AudioFrame, VideoFrame
    PYAV_IMPORT_ERROR: Optional[ImportError] = None
except ImportError as import_error:  # pragma: no cover - optional dependency guard
    AudioFrame = None  # type: ignore[assignment]
    VideoFrame = None  # type: ignore[assignment]
    PYAV_IMPORT_ERROR = import_error

from .. import state
from ..config import AVATAR_IMAGE, AVATAR_VIDEO_PATH, TEMP_DIR
from .segment_lipsync import _attach_audio
from .tts import convert_to_wav, generate_tts


# Background asyncio loop used for all WebRTC sessions
_event_loop = asyncio.new_event_loop()


def _loop_runner(loop: asyncio.AbstractEventLoop) -> None:
    asyncio.set_event_loop(loop)
    loop.run_forever()


_thread = Thread(target=_loop_runner, args=(_event_loop,), daemon=True)
_thread.start()


T = TypeVar("T")


def _run_async(coro: Coroutine[Any, Any, T]) -> T:
    """Submit a coroutine to the background loop and wait for the result."""
    return asyncio.run_coroutine_threadsafe(coro, _event_loop).result()


_DIRECTION_LINES = {"a=sendrecv", "a=sendonly", "a=recvonly", "a=inactive"}


def _ensure_sdp_directions(sdp: str) -> str:
    """Ensure each media section has a direction attribute and validate SDP."""
    lines = [line for line in sdp.replace("\r\n", "\n").split("\n") if line != ""]
    normalized: List[str] = []
    in_media = False
    media_has_direction = False
    media_indices: List[int] = []

    for line in lines:
        if line.startswith("m="):
            if in_media and not media_has_direction:
                normalized.append("a=sendrecv")
            in_media = True
            media_has_direction = False
            media_indices.append(len(normalized))
            normalized.append(line)
            continue

        if in_media and line in _DIRECTION_LINES:
            media_has_direction = True

        normalized.append(line)

    if in_media and not media_has_direction:
        normalized.append("a=sendrecv")

    if not media_indices:
        raise RuntimeError("Invalid SDP offer: no media sections provided")

    return "\r\n".join(normalized) + "\r\n"


@dataclass
class SessionOptions:
    language: str = "ru"
    pads: Tuple[int, int, int, int] = (0, 50, 0, 0)
    fps: float = 30.0
    chunk_word_limit: int = 15
    chunk_stride: int = 12
    audio_sample_rate: int = 16000
    webrtc_sample_rate: int = 48000
    connection_timeout: float = 12.0
    low_latency: bool = False
    dynamic_mode: bool = False
    base_video_path: str = AVATAR_VIDEO_PATH
    batch_size_override: Optional[int] = None
    session_id: str = ""


DEFAULT_DYNAMIC_BATCH = 32


class WebRTCRenderSession:
    """Store intermediate artefacts for a single WebRTC generation run."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.temp_dir = Path(TEMP_DIR) / "webrtc" / session_id
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.temp_video_path = self.temp_dir / "stream_silent.mp4"
        self.audio_path = self.temp_dir / "stream_audio.wav"
        self.output_video_path = self.temp_dir / "stream_final.mp4"
        self.encoder_queue: "queue.Queue[Optional[np.ndarray]]" = queue.Queue(maxsize=32)
        self.encoder_thread: Optional[threading.Thread] = None
        self.encoder_report: dict[str, Any] = {"frame_count": 0}
        self.encoder_error: Optional[BaseException] = None
        self.encoder_error_holder: dict[str, BaseException] = {}
        self.width: Optional[int] = None
        self.height: Optional[int] = None
        self.fps: Optional[float] = None
        self.audio_segments: List[torch.Tensor] = []
        self.audio_sample_rate: Optional[int] = None
        self.result_ready = False
        self.error: Optional[str] = None
        self.created_at = time.time()
        self._encoder_started = False
        self._lock = threading.Lock()
        self.pc: Optional["RTCPeerConnection"] = None
        self.video_track: Optional["LipsyncVideoStreamTrack"] = None
        self.audio_track: Optional["ChunkedAudioStreamTrack"] = None

    def ensure_encoder(self, width: int, height: int, fps: float) -> None:
        with self._lock:
            if self._encoder_started:
                return
            self.width = width
            self.height = height
            self.fps = fps
            self.encoder_queue, self.encoder_thread, self.encoder_report, self.encoder_error_holder = _spawn_encoder(
                fps=fps,
                temp_video_path=self.temp_video_path,
                width=width,
                height=height,
            )
            self._encoder_started = True

    def push_frame(self, frame: np.ndarray) -> None:
        if not self._encoder_started:
            self.ensure_encoder(frame.shape[1], frame.shape[0], self.fps or 25.0)
        while True:
            try:
                self.encoder_queue.put(frame.copy(), timeout=5.0)
                break
            except queue.Full:
                continue

    def append_audio(self, waveform: torch.Tensor, sample_rate: int) -> None:
        if self.audio_sample_rate is None:
            self.audio_sample_rate = sample_rate
        elif self.audio_sample_rate != sample_rate:
            waveform = audio_fn.resample(waveform, sample_rate, self.audio_sample_rate)
        self.audio_segments.append(waveform.cpu())

    def finalize(self) -> None:
        if self.result_ready or self.error:
            return
        if not self._encoder_started:
            if not self.error:
                self.error = "Encoder was not started; no frames were produced"
            return
        while True:
            try:
                self.encoder_queue.put(None, timeout=5.0)
                break
            except queue.Full:
                continue
        if self.encoder_thread:
            self.encoder_thread.join()
        encode_error = self.encoder_error_holder.get("error")
        if encode_error:
            self.error = f"FFmpeg encoding failed: {encode_error}"
            return
        if not self.audio_segments:
            if not self.error:
                self.error = "No audio segments captured during stream"
            return
        audio_tensor = torch.cat(self.audio_segments, dim=1)
        self.audio_segments.clear()
        torchaudio.save(str(self.audio_path), audio_tensor, self.audio_sample_rate or 16000)
        del audio_tensor
        return_code, stderr_output = _attach_audio(self.temp_video_path, self.audio_path, self.output_video_path)
        if return_code != 0:
            self.error = f"Audio mux failed: {stderr_output}"
            return
        self.result_ready = True


def _spawn_encoder(
    fps: float,
    temp_video_path: Path,
    width: int,
    height: int,
    preset: str = "veryfast",
    crf: int = 18,
    threads: int = 8,
) -> tuple["queue.Queue[Optional[np.ndarray]]", threading.Thread, dict[str, Any], dict[str, BaseException]]:
    frame_queue: "queue.Queue[Optional[np.ndarray]]" = queue.Queue(maxsize=32)
    report: dict[str, Any] = {"frame_count": 0, "encode_wall_time": 0.0}
    error_holder: dict[str, BaseException] = {}

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
        f"{fps}",
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
        str(temp_video_path),
    ]
    if threads >= 0:
        command.extend(["-threads", str(threads)])

    def _encode_worker() -> None:
        proc: Optional[subprocess.Popen] = None
        encode_start = time.perf_counter()
        try:
            proc = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
            while True:
                item = frame_queue.get()
                if item is None:
                    break
                frame = item
                if frame.dtype != np.uint8:
                    frame = frame.astype(np.uint8, copy=False)
                if not frame.flags.c_contiguous:
                    frame = np.ascontiguousarray(frame)
                if proc.stdin is None:
                    raise RuntimeError("FFmpeg stdin is not available")
                proc.stdin.write(memoryview(frame))
                report["frame_count"] += 1
            if proc.stdin is not None:
                proc.stdin.close()
            return_code = proc.wait()
            stderr_output = ""
            if proc.stderr is not None:
                stderr_output = proc.stderr.read().decode("utf-8", errors="ignore")
                proc.stderr.close()
            report.update({
                "return_code": return_code,
                "stderr": stderr_output,
                "encode_wall_time": time.perf_counter() - encode_start,
                "width": width,
                "height": height,
            })
            if return_code != 0:
                raise RuntimeError(f"FFmpeg exited with code {return_code}: {stderr_output}")
        except BaseException as exc:  # noqa: BLE001
            error_holder["error"] = exc
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

    thread = threading.Thread(target=_encode_worker, daemon=True)
    thread.start()
    return frame_queue, thread, report, error_holder


_STREAM_SESSIONS: dict[str, WebRTCRenderSession] = {}
_STREAM_SESSIONS_LOCK = threading.Lock()


def register_stream_session(session: WebRTCRenderSession) -> None:
    with _STREAM_SESSIONS_LOCK:
        _STREAM_SESSIONS[session.session_id] = session


def get_stream_session(session_id: str) -> Optional[WebRTCRenderSession]:
    with _STREAM_SESSIONS_LOCK:
        return _STREAM_SESSIONS.get(session_id)


def remove_stream_session(session_id: str) -> None:
    with _STREAM_SESSIONS_LOCK:
        _STREAM_SESSIONS.pop(session_id, None)


class LipsyncVideoStreamTrack(VideoStreamTrack):
    """Video track fed by lipsync frames coming from the inference service."""

    def __init__(self, fps: float, loop: asyncio.AbstractEventLoop) -> None:
        super().__init__()
        self._fps = max(1.0, fps)
        self._loop = loop
        self._queue: asyncio.Queue[Optional[np.ndarray]] = asyncio.Queue()
        self._pts: int = 0
        self._ended = False

    async def recv(self) -> VideoFrame:
        frame = await self._queue.get()
        if frame is None:
            self._ended = True
            raise MediaStreamError("Video stream finished")

        video_frame = VideoFrame.from_ndarray(frame, format="bgr24")
        video_frame.pts = self._pts
        video_frame.time_base = Fraction(1, int(self._fps))
        self._pts += 1
        return video_frame

    async def _enqueue(self, frame: Optional[np.ndarray]) -> None:
        await self._queue.put(frame)

    def push(self, frame: np.ndarray) -> None:
        if self._ended:
            return
        asyncio.run_coroutine_threadsafe(self._enqueue(frame.copy()), self._loop)

    def finish(self) -> None:
        if self._ended:
            return
        self._ended = True
        asyncio.run_coroutine_threadsafe(self._enqueue(None), self._loop)
        super().stop()


class ChunkedAudioStreamTrack(AudioStreamTrack):
    """Audio track streaming PCM chunks generated by TTS."""

    def __init__(self, sample_rate: int, loop: asyncio.AbstractEventLoop) -> None:
        super().__init__()
        self._sample_rate = sample_rate
        self._loop = loop
        self._queue: asyncio.Queue[Optional[np.ndarray]] = asyncio.Queue()
        self._pts: int = 0
        # 20 ms frames by default
        self._frame_size = max(1, int(sample_rate * 0.02))
        self._ended = False

    async def recv(self) -> AudioFrame:
        samples = await self._queue.get()
        if samples is None:
            self._ended = True
            raise MediaStreamError("Audio stream finished")

        frame = AudioFrame.from_ndarray(samples[np.newaxis, :], format="s16", layout="mono")
        frame.sample_rate = self._sample_rate
        frame.time_base = Fraction(1, self._sample_rate)
        frame.pts = self._pts
        self._pts += samples.shape[0]
        return frame

    async def _enqueue(self, samples: Optional[np.ndarray]) -> None:
        await self._queue.put(samples)

    def push(self, waveform: torch.Tensor, sample_rate: int) -> None:
        if self._ended:
            return
        if waveform.dim() == 2:
            waveform = waveform.squeeze(0)
        if waveform.dim() != 1:
            raise ValueError("Audio waveform must be 1D after squeeze")

        torch_wave = waveform.detach().cpu().float()
        if sample_rate != self._sample_rate:
            torch_wave = audio_fn.resample(torch_wave.unsqueeze(0), sample_rate, self._sample_rate).squeeze(0)

        np_wave = torch_wave.numpy()
        np_wave = np.clip(np_wave, -1.0, 1.0)
        int_wave = (np_wave * 32767.0).astype(np.int16)

        for start in range(0, len(int_wave), self._frame_size):
            chunk = int_wave[start:start + self._frame_size]
            if chunk.size == 0:
                continue
            asyncio.run_coroutine_threadsafe(self._enqueue(chunk.copy()), self._loop)

    def finish(self) -> None:
        if self._ended:
            return
        self._ended = True
        asyncio.run_coroutine_threadsafe(self._enqueue(None), self._loop)
        super().stop()


def _split_text(text: str, max_words: int, stride: int) -> List[str]:
    """Split text into short chunks suitable for low-latency TTS."""
    text = text.strip()
    if not text:
        return []

    sentence_pattern = re.compile(r"[^.!?]+[.!?]|[^.!?]+$", re.MULTILINE)
    sentences = sentence_pattern.findall(text)
    if not sentences:
        sentences = [text]

    chunks: List[str] = []
    for sentence in sentences:
        words = sentence.strip().split()
        if not words:
            continue
        if len(words) <= max_words:
            chunks.append(" ".join(words))
            continue

        step = max(1, stride)
        for i in range(0, len(words), step):
            slice_words = words[i:i + max_words]
            if not slice_words:
                continue
            chunks.append(" ".join(slice_words))

    return chunks


async def _run_stream_pipeline(
    pc: RTCPeerConnection,
    text_chunks: Iterable[str],
    options: SessionOptions,
    service,
    video_track: LipsyncVideoStreamTrack,
    audio_track: ChunkedAudioStreamTrack,
) -> None:
    loop = asyncio.get_running_loop()
    pads = options.pads
    face_path = options.base_video_path if options.dynamic_mode else AVATAR_IMAGE
    static_mode = not options.dynamic_mode
    frame_state = {"count": 0}
    session = get_stream_session(options.session_id) if options.session_id else None
    if session:
        session.fps = options.fps

    def frame_sink(frame: np.ndarray) -> None:
        video_track.push(frame)
        frame_state["count"] += 1
        if session:
            session.ensure_encoder(frame.shape[1], frame.shape[0], options.fps)
            session.push_frame(frame)

    async def _wait_for_connection_ready(timeout: float) -> bool:
        start = loop.time()
        while True:
            state = pc.connectionState
            if state == "connected":
                print("✅ WebRTC connection established; starting media stream")
                return True
            if state in {"failed", "closed", "disconnected"}:
                print(f"⚠️ WebRTC connection state {state}; aborting stream pipeline")
                return False
            if loop.time() - start >= timeout:
                print(f"⚠️ Timeout waiting for WebRTC connection ({timeout:.1f}s); aborting")
                return False
            await asyncio.sleep(0.1)

    try:
        if static_mode:
            service.preload_static_face(
                face_path,
                fps=options.fps,
                pads=pads,
                resize_factor=1,
                crop=(0, -1, 0, -1),
                rotate=False,
                nosmooth=False,
            )
        else:
            service.preload_video_cache(
                face_path=face_path,
                fps=options.fps,
                pads=pads,
            )
    except Exception:
        # preload is an optimization; failures should not abort the session
        pass

    if not await _wait_for_connection_ready(options.connection_timeout):
        video_track.finish()
        audio_track.finish()
        await pc.close()
        if session:
            session.error = session.error or "WebRTC connection not established"
            session.finalize()
        print("⚠️ WebRTC connection not established; session terminated")
        return

    try:
        for idx, chunk in enumerate(text_chunks):
            if pc.connectionState in {"closed", "failed", "disconnected"}:
                break
            if not chunk:
                continue

            print(f"🎧 WebRTC TTS chunk #{idx + 1}: {chunk[:60]}{'...' if len(chunk) > 60 else ''}")

            # TTS generation and decoding are blocking; run in executor
            audio_bytes = await loop.run_in_executor(None, generate_tts, chunk, options.language)
            waveform, sample_rate = await loop.run_in_executor(
                None,
                lambda: convert_to_wav(audio_bytes, None),
            )

            try:
                audio_track.push(waveform, sample_rate)
                if session:
                    session.append_audio(waveform, sample_rate)

                if options.dynamic_mode:
                    batch_override = options.batch_size_override or DEFAULT_DYNAMIC_BATCH
                    frame_offset = frame_state["count"]
                else:
                    batch_override = options.batch_size_override
                    if batch_override is None:
                        batch_override = 32 if options.low_latency else None
                    frame_offset = 0

                def _run_process() -> None:
                    service.process(
                        face_path=face_path,
                        audio_path="",
                        output_path=None,
                        static=static_mode,
                        fps=options.fps,
                        pads=pads,
                        audio_waveform=waveform,
                        audio_sample_rate=sample_rate,
                        frame_sink=frame_sink,
                        batch_size_override=batch_override,
                        frame_offset=frame_offset,
                    )

                await loop.run_in_executor(None, _run_process)
            finally:
                if not options.dynamic_mode:
                    frame_state["count"] = 0

    except Exception as exc:
        print(f"❌ Realtime pipeline error: {exc}")
        import traceback

        traceback.print_exc()
        if session:
            session.error = str(exc)
    finally:
        video_track.finish()
        audio_track.finish()
        await pc.close()
        if session:
            session.finalize()
            if session.error:
                print(f"⚠️ WebRTC session {session.session_id} failed during finalize: {session.error}")
        print("✅ WebRTC session finished")


async def _create_webrtc_session(
    offer: RTCSessionDescription,
    text: str,
    options: SessionOptions,
    session: WebRTCRenderSession,
) -> RTCSessionDescription:
    service = state.lipsync_service_gan or state.lipsync_service_nogan
    if service is None:
        raise RuntimeError("Lipsync service is not initialized")

    ice_servers = [
        RTCIceServer(urls="stun:stun.l.google.com:19302"),
        RTCIceServer(urls="stun:stun1.l.google.com:19302"),
    ]
    pc = RTCPeerConnection(configuration=RTCConfiguration(iceServers=ice_servers))
    loop = asyncio.get_running_loop()

    video_track = LipsyncVideoStreamTrack(options.fps, loop)
    audio_track = ChunkedAudioStreamTrack(sample_rate=options.webrtc_sample_rate, loop=loop)
    pc.addTrack(video_track)
    pc.addTrack(audio_track)

    session.pc = pc
    session.video_track = video_track
    session.audio_track = audio_track

    # aiortc requires explicit transceiver directions; some browsers omit them in offers.
    for transceiver in pc.getTransceivers():
        if transceiver.direction is None:
            if transceiver.kind == "video":
                transceiver.direction = "sendonly"
            elif transceiver.kind == "audio":
                transceiver.direction = "sendonly"

    @pc.on("connectionstatechange")
    async def _on_state_change() -> None:
        state = pc.connectionState
        print(f"🔌 WebRTC connection state: {state}")
        if state in {"failed", "closed", "disconnected"}:
            video_track.finish()
            audio_track.finish()
            if state in {"failed", "disconnected"} and not session.error:
                session.error = f"WebRTC connection {state}"

    await pc.setRemoteDescription(offer)

    # Some browsers omit a=direction values in their offer which leaves aiortc
    # transceivers without a preferred direction. Default to sendrecv so answer
    # generation doesn't fail.
    for transceiver in pc.getTransceivers():
        if transceiver.direction is None:
            transceiver.direction = "sendrecv"
        if getattr(transceiver, "_offerDirection", None) is None:
            transceiver._offerDirection = transceiver.direction
    answer = await pc.createAnswer()
    print("===== Generated local SDP =====")
    print(answer.sdp)
    print("===== End local SDP =====")
    await pc.setLocalDescription(answer)

    async def _wait_for_ice_gathering_complete(timeout: float = 5.0) -> None:
        start = loop.time()
        while pc.iceGatheringState not in {"complete", "closed"}:
            if loop.time() - start > timeout:
                print(f"⚠️ ICE gathering timed out after {timeout:.1f}s; returning partial candidates")
                break
            await asyncio.sleep(0.1)

    await _wait_for_ice_gathering_complete()

    chunks = _split_text(text, options.chunk_word_limit, options.chunk_stride)
    if not chunks:
        raise RuntimeError("Text is empty after preprocessing")

    asyncio.create_task(
        _run_stream_pipeline(pc, chunks, options, service, video_track, audio_track)
    )

    return pc.localDescription  # type: ignore[return-value]


def start_webrtc_session(
    sdp: str,
    offer_type: str,
    text: str,
    language: str = "ru",
    pads: Tuple[int, int, int, int] = (0, 50, 0, 0),
    fps: float = 25.0,
    low_latency: bool = False,
    dynamic_mode: bool = False,
    base_video_path: Optional[str] = None,
    batch_size: Optional[int] = None,
) -> Tuple[RTCSessionDescription, str]:
    text = text.strip()
    if not text:
        raise ValueError("Text must not be empty")
    if fps <= 0:
        raise ValueError("FPS must be positive")

    if AIORTC_IMPORT_ERROR is not None:
        raise RuntimeError(
            "aiortc не установлен. Установите зависимости из requirements_web.txt"
        )
    if PYAV_IMPORT_ERROR is not None:
        raise RuntimeError(
            "PyAV не установлен. Установите зависимости из requirements_web.txt"
        )

    try:
        normalized_sdp = _ensure_sdp_directions(sdp)
    except RuntimeError as err:
        raise RuntimeError(f"Invalid WebRTC offer: {err}") from err
    print("===== Incoming offer SDP =====")
    print(normalized_sdp)
    print("===== End offer SDP =====")
    offer = RTCSessionDescription(sdp=normalized_sdp, type=offer_type)
    session = WebRTCRenderSession(uuid.uuid4().hex)
    register_stream_session(session)
    video_path = base_video_path or AVATAR_VIDEO_PATH
    if dynamic_mode and not os.path.exists(video_path):
        session.error = f"Видео-аватар не найден: {video_path}"
        remove_stream_session(session.session_id)
        raise FileNotFoundError(f"Видео-аватар не найден: {video_path}")

    options = SessionOptions(
        language=language,
        pads=pads,
        fps=fps,
        low_latency=low_latency,
        dynamic_mode=dynamic_mode,
        base_video_path=video_path,
        batch_size_override=batch_size if batch_size is not None else (DEFAULT_DYNAMIC_BATCH if dynamic_mode else None),
        session_id=session.session_id,
    )
    if dynamic_mode:
        options.chunk_word_limit = max(6, min(options.chunk_word_limit, 14))
        options.chunk_stride = max(4, min(options.chunk_stride, 12))

    try:
        answer = _run_async(_create_webrtc_session(offer, text, options, session))
        return answer, session.session_id
    except Exception as exc:
        session.error = str(exc)
        session.finalize()
        remove_stream_session(session.session_id)
        raise
