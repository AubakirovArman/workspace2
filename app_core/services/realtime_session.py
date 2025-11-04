"""Support for realtime WebRTC streaming with chunked TTS audio."""
from __future__ import annotations

import asyncio
import os
import re
import uuid
from dataclasses import dataclass
from fractions import Fraction
from threading import Thread
from typing import Any, Coroutine, Iterable, List, Optional, Tuple, TypeVar

import numpy as np
import torch
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
from ..config import AVATAR_IMAGE, TEMP_DIR
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
    fps: float = 25.0
    chunk_word_limit: int = 15
    chunk_stride: int = 12
    audio_sample_rate: int = 16000
    webrtc_sample_rate: int = 48000
    connection_timeout: float = 12.0
    low_latency: bool = False


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

    def frame_sink(frame: np.ndarray) -> None:
        video_track.push(frame)

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
        service.preload_static_face(
            AVATAR_IMAGE,
            fps=options.fps,
            pads=pads,
            resize_factor=1,
            crop=(0, -1, 0, -1),
            rotate=False,
            nosmooth=False,
        )
    except Exception:
        # preload is an optimization; failures should not abort the session
        pass

    if not await _wait_for_connection_ready(options.connection_timeout):
        video_track.finish()
        audio_track.finish()
        await pc.close()
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

                def _run_process() -> None:
                    service.process(
                        face_path=AVATAR_IMAGE,
                        audio_path="",
                        output_path=None,
                        static=True,
                        fps=options.fps,
                        pads=pads,
                        audio_waveform=waveform,
                        audio_sample_rate=sample_rate,
                        frame_sink=frame_sink,
                        batch_size_override=32 if options.low_latency else None,
                    )

                await loop.run_in_executor(None, _run_process)
            finally:
                pass

    except Exception as exc:
        print(f"❌ Realtime pipeline error: {exc}")
        import traceback

        traceback.print_exc()
    finally:
        video_track.finish()
        audio_track.finish()
        await pc.close()
        print("✅ WebRTC session finished")


async def _create_webrtc_session(
    offer: RTCSessionDescription,
    text: str,
    options: SessionOptions,
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
) -> RTCSessionDescription:
    text = text.strip()
    if not text:
        raise ValueError("Text must not be empty")

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
    options = SessionOptions(language=language, pads=pads, fps=fps, low_latency=low_latency)
    return _run_async(_create_webrtc_session(offer, text, options))
