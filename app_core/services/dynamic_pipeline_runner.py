"""Helper for running the dynamic full pipeline inside the web app."""
from __future__ import annotations

import argparse
import shutil
import time
from datetime import datetime, UTC
from pathlib import Path
from typing import Dict, Tuple

from flask import current_app

from scripts.dynamic_full_pipeline import (
    PADS,
    ServiceManager,
    run_pipeline,
    _service_signature,
)

from .. import state
from ..config import CHECKPOINT_PATH_GAN, TEMP_DIR_PATH, AVATAR_FPS, AVATAR_VIDEO_PATH as CONFIG_AVATAR_VIDEO_PATH
from .tts import convert_to_wav, generate_tts

# Оптимальные параметры из последнего бенчмарка
BEST_BATCH_SIZE = 32
BEST_FFMPEG_PRESET = "veryfast"
BEST_FFMPEG_CRF = 16
BEST_FFMPEG_THREADS = 8
BEST_QUEUE_SIZE = 8
FACE_DET_BATCH = 16
DEFAULT_FPS = AVATAR_FPS
AVATAR_VIDEO_PATH = Path(CONFIG_AVATAR_VIDEO_PATH).expanduser()

# Директория для временных артефактов
TEST_RUN_ROOT = TEMP_DIR_PATH / "test_runs"
TEST_RUN_ROOT.mkdir(parents=True, exist_ok=True)

_PIPELINE_MANAGER = ServiceManager(allow_reuse=True)
_SERVICE_INJECTED = False


def _extract_service_details(service) -> Dict[str, object]:
    return {
        "segmentation_path": getattr(service, "_segmentation_path", None),
        "segmentation_enabled": bool(getattr(service, "segmentation_enabled", False)),
        "super_resolution_path": getattr(service, "_sr_path", None),
        "super_resolution_enabled": bool(getattr(service, "sr_enabled", False)),
        "realesrgan_path": getattr(service, "_realesrgan_path", None),
        "realesrgan_enabled": bool(getattr(service, "realesrgan_enabled", False)),
        "use_fp16": bool(getattr(service, "use_fp16", False)),
        "use_compile": bool(getattr(service, "use_compile", False)),
        "ffmpeg_threads": getattr(service, "ffmpeg_threads", BEST_FFMPEG_THREADS),
        "ffmpeg_filter_threads": getattr(service, "ffmpeg_filter_threads", 0),
        "face_det_batch_size": getattr(service, "face_det_batch_size", FACE_DET_BATCH),
        "wav2lip_batch_size": getattr(service, "wav2lip_batch_size", BEST_BATCH_SIZE),
    }


def _ensure_service_registered(args: argparse.Namespace) -> None:
    global _SERVICE_INJECTED
    if _SERVICE_INJECTED or not _PIPELINE_MANAGER.allow_reuse:
        return

    service = state.lipsync_service_gan or state.lipsync_service_nogan
    if service is None:
        return

    # Убеждаемся, что динамическое видео закэшировано
    try:
        cache_key = service._video_cache_key(  # type: ignore[attr-defined]
            str(AVATAR_VIDEO_PATH),
            float(args.fps),
            PADS,
            1,
            (0, -1, 0, -1),
            False,
            False,
        )
        video_cache = getattr(service, "_video_cache", {})
        if cache_key not in video_cache:
            current_app.logger.info("🎬 Предзагрузка динамического видео через основной сервис...")
            service.preload_video_cache(
                face_path=str(AVATAR_VIDEO_PATH),
                fps=float(args.fps),
                pads=PADS,
            )
    except Exception as preload_error:  # noqa: BLE001
        current_app.logger.warning("⚠️ Не удалось предварительно прогреть кеш динамического видео: %s", preload_error)

    details = _extract_service_details(service)
    signature = _service_signature(args)
    preload_key = (str(AVATAR_VIDEO_PATH), float(args.fps), PADS)
    _PIPELINE_MANAGER.inject_prepared_service(
        service,
        details,
        signature,
        preload_keys={preload_key},
        warmup_keys=set(),
    )
    _SERVICE_INJECTED = True


def _build_args(
    audio_path: Path,
    output_path: Path,
    meta_path: Path,
    temp_video_path: Path,
) -> argparse.Namespace:
    args_dict = {
        "avatar": str(AVATAR_VIDEO_PATH),
        "audio": str(audio_path),
        "output": str(output_path),
        "checkpoint": str(CHECKPOINT_PATH_GAN),
        "device": "cuda:0",
        "batch_size": BEST_BATCH_SIZE,
        "face_det_batch": FACE_DET_BATCH,
        "fps": DEFAULT_FPS,
        "temp_video": str(temp_video_path),
        "queue_size": BEST_QUEUE_SIZE,
        "ffmpeg_preset": BEST_FFMPEG_PRESET,
        "ffmpeg_crf": BEST_FFMPEG_CRF,
        "ffmpeg_threads": BEST_FFMPEG_THREADS,
        "ffmpeg_filter_threads": 0,
        "warmup_duration": 0.0,
        "meta_out": str(meta_path),
        "disable_segmentation": False,
        "disable_super_resolution": False,
        "disable_realesrgan": False,
        "disable_fp16": False,
        "disable_compile": False,
        "sweep_config": None,
        "sweep_output_dir": None,
        "sweep_repeats": 1,
        "sweep_limit": None,
        "sweep_cleanup": False,
        "sweep_reuse_service": False,
    }
    return argparse.Namespace(**args_dict)


def _cleanup_temp_files(run_dir: Path, keep_video: bool = True) -> None:
    for candidate in run_dir.glob("dynamic_final_silent*.mp4"):
        try:
            candidate.unlink()
        except OSError:
            pass
    if not keep_video:
        try:
            shutil.rmtree(run_dir, ignore_errors=True)
        except Exception:
            pass


def generate_dynamic_test_video(text: str, language: str = "ru") -> Tuple[Path, Dict[str, object]]:
    """Сгенерировать видео через dynamic_full_pipeline и вернуть путь + метаданные."""
    if not text or not text.strip():
        raise ValueError("Текст не может быть пустым")
    if not AVATAR_VIDEO_PATH.exists():
        raise FileNotFoundError(f"Аватар-видео не найдено: {AVATAR_VIDEO_PATH}")

    timestamp = datetime.now(UTC).strftime("test_%Y%m%d_%H%M%S_%f")
    run_dir = TEST_RUN_ROOT / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    audio_path = run_dir / "input.wav"
    meta_path = run_dir / "summary.json"
    output_path = run_dir / "dynamic_final.mp4"
    temp_video_path = run_dir / "dynamic_final_silent.mp4"

    current_app.logger.info("🎤 Генерация TTS для test-пайплайна (%s)", language)
    start_tts = time.perf_counter()
    audio_data = generate_tts(text.strip(), language)
    tts_time = time.perf_counter() - start_tts
    current_app.logger.info("✅ TTS готов за %.2fs", tts_time)

    start_convert = time.perf_counter()
    _, _ = convert_to_wav(audio_data, str(audio_path))
    convert_time = time.perf_counter() - start_convert
    current_app.logger.info("🔄 Конвертация аудио → WAV заняла %.2fs", convert_time)

    args = _build_args(
        audio_path=audio_path,
        output_path=output_path,
        meta_path=meta_path,
        temp_video_path=temp_video_path,
    )

    _ensure_service_registered(args)

    current_app.logger.info("🚀 Запуск dynamic_full_pipeline (batch=%s, preset=%s)",
                            BEST_BATCH_SIZE, BEST_FFMPEG_PRESET)
    summary = run_pipeline(args, service_manager=_PIPELINE_MANAGER)

    _cleanup_temp_files(run_dir, keep_video=True)

    summary_payload: Dict[str, object] = dict(summary)
    summary_payload["tts_time"] = tts_time
    summary_payload["audio_convert_time"] = convert_time
    summary_payload["run_dir"] = str(run_dir)
    summary_payload["timestamp"] = timestamp

    return output_path, summary_payload
