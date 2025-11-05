"""Initialization logic for the preloaded lipsync service."""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional, Tuple

import torch

from ..config import (
    AVATAR_FPS,
    AVATAR_IMAGE,
    AVATAR_VIDEO_PATH,
    AVATAR_PREVIEW_PATH,
    AVATAR_STATIC_MODE,
    CHECKPOINT_PATH_GAN,
    CHECKPOINT_PATH_NOGAN,
    MAX_GAN_MODELS,
    GAN_MODEL_INSTANCES,
    ENABLE_REALESRGAN,
    ENABLE_SEGMENTATION,
    ENABLE_SUPER_RESOLUTION,
    HD_MODULES_ROOT,
    REALESRGAN_OUTSCALE,
    REALESRGAN_PATH,
    SEGMENTATION_PATH_HD,
    SR_PATH_HD,
)
from ..state import set_state

from service import LipsyncService  # noqa: E402


_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
_VIDEO_EXTENSIONS = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v'}


def _is_video(path: str) -> bool:
    return Path(path).suffix.lower() in _VIDEO_EXTENSIONS


def _is_image(path: str) -> bool:
    return Path(path).suffix.lower() in _IMAGE_EXTENSIONS


def _resolve_realesrgan_path() -> Optional[str]:
    if not ENABLE_REALESRGAN:
        print("ℹ️ Real-ESRGAN отключен (ENABLE_REALESRGAN=0).")
        return None
    if os.path.exists(REALESRGAN_PATH):
        print(f"✅ Real-ESRGAN веса найдены: {REALESRGAN_PATH} (outscale={REALESRGAN_OUTSCALE})")
        return REALESRGAN_PATH
    print(f"⚠️ Предупреждение: веса Real-ESRGAN не найдены ({REALESRGAN_PATH}). Улучшение будет пропущено.")
    return None


def init_lipsync_service() -> Tuple[LipsyncService, Optional[LipsyncService], Optional[object]]:
    print("\n" + "=" * 60)
    print("🚀 Инициализация Avatar Lipsync Service")
    print("=" * 60)

    if not os.path.exists(AVATAR_IMAGE):
        raise FileNotFoundError(f"Аватар не найден: {AVATAR_IMAGE}")
    if not os.path.exists(CHECKPOINT_PATH_GAN):
        raise FileNotFoundError(f"GAN модель не найдена: {CHECKPOINT_PATH_GAN}")
    if not os.path.exists(CHECKPOINT_PATH_NOGAN):
        print(f"⚠️ Предупреждение: NoGAN модель не найдена ({CHECKPOINT_PATH_NOGAN}). Страница realtime2 будет недоступна.")
    realesrgan_available = _resolve_realesrgan_path()

    print(f"✅ Аватар найден: {AVATAR_IMAGE}")
    print(f"✅ GAN модель найдена: {CHECKPOINT_PATH_GAN}")
    if os.path.exists(CHECKPOINT_PATH_NOGAN):
        print(f"✅ NoGAN модель найдена: {CHECKPOINT_PATH_NOGAN}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🔧 Устройство: {device}")
    if device == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
        if hasattr(torch, 'set_float32_matmul_precision'):
            torch.set_float32_matmul_precision('high')

    use_hd_modules = ENABLE_SEGMENTATION or ENABLE_SUPER_RESOLUTION or ENABLE_REALESRGAN

    hd_modules_root: Optional[str] = None
    if use_hd_modules:
        if HD_MODULES_ROOT.exists():
            hd_modules_root = str(HD_MODULES_ROOT)
            print(f"✅ Найдены модули Wav2Lip-HD: {hd_modules_root}")
        else:
            print(f"⚠️ Папка с кодом Wav2Lip-HD не найдена: {HD_MODULES_ROOT}")
    else:
        print("ℹ️ Дополнительные HD модули отключены (ENABLE_* флаги = 0).")

    segmentation_path: Optional[str] = None
    if ENABLE_SEGMENTATION:
        if os.path.exists(SEGMENTATION_PATH_HD):
            segmentation_path = SEGMENTATION_PATH_HD
            print(f"✅ Модель сегментации включена: {SEGMENTATION_PATH_HD}")
        else:
            print(f"⚠️ Предупреждение: файл сегментации не найден ({SEGMENTATION_PATH_HD}). Будет использована только Wav2Lip без сегментации.")
    else:
        print("ℹ️ Сегментация отключена (ENABLE_SEGMENTATION=0).")

    sr_path: Optional[str] = None
    if ENABLE_SUPER_RESOLUTION:
        if os.path.exists(SR_PATH_HD):
            sr_path = SR_PATH_HD
            print(f"✅ Суперразрешение ESRGAN включено: {SR_PATH_HD}")
        else:
            print(f"⚠️ Предупреждение: файл суперразрешения не найден ({SR_PATH_HD}). Будет использована базовая модель без ESRGAN.")
    else:
        print("ℹ️ Суперразрешение отключено (ENABLE_SUPER_RESOLUTION=0).")

    common_kwargs = dict(
        face_det_batch_size=16,
        wav2lip_batch_size=16,
        segmentation_path=segmentation_path,
        sr_path=sr_path,
        modules_root=hd_modules_root,
        realesrgan_path=realesrgan_available,
        realesrgan_outscale=REALESRGAN_OUTSCALE,
        use_fp16=True,
        use_compile=True,
    )

    total_instances = GAN_MODEL_INSTANCES
    if total_instances > MAX_GAN_MODELS:
        total_instances = MAX_GAN_MODELS

    if device == 'cuda':
        logical_gpu_count = torch.cuda.device_count()
        visible_devices = [f'cuda:{idx}' for idx in range(logical_gpu_count)]
        if not visible_devices:
            visible_devices = ['cuda:0']

        if len(visible_devices) >= total_instances:
            if len(visible_devices) > total_instances:
                print(
                    f"ℹ️ Будут использованы первые {total_instances} из {len(visible_devices)} доступных GPU для GAN моделей."
                )
            gan_devices = visible_devices[:total_instances]
        else:
            print(
                f"⚠️ Запрошено {total_instances} GAN моделей, но доступно только {len(visible_devices)} GPU. Некоторые устройства будут использованы повторно."
            )
            gan_devices = [visible_devices[idx % len(visible_devices)] for idx in range(total_instances)]
    else:
        gan_devices = [device] * total_instances

    if not gan_devices:
        gan_devices = [device]
        total_instances = 1

    unique_devices = sorted(set(gan_devices), key=gan_devices.index)
    reuse_notice = " (повторное использование GPU)" if len(unique_devices) < len(gan_devices) else ""
    print(
        f"🧠 Планируется загрузка {len(gan_devices)} GAN моделей на устройства: {', '.join(gan_devices)}{reuse_notice}"
    )

    is_video_source = _is_video(AVATAR_IMAGE)
    use_static_cache = AVATAR_STATIC_MODE or not is_video_source
    if not AVATAR_STATIC_MODE and not is_video_source:
        print("⚠️ Запрошен динамический режим, но источник не видео. Используется статичный режим.")

    if use_static_cache:
        print("🎯 Режим аватара: статичный (предкэшированное лицо)")
    else:
        print("🎞️ Режим аватара: динамический (используется всё видео)")

    primary_device = gan_devices[0]
    print(f"\n📦 Загрузка GAN модели в память (device={primary_device})...")
    start = time.time()
    gan_service = LipsyncService(
        checkpoint_path=CHECKPOINT_PATH_GAN,
        device=primary_device,
        **common_kwargs
    )
    model_ready_time = time.time()
    print(f"✅ GAN модель загружена за {model_ready_time - start:.2f}s")

    if use_static_cache:
        preload_start = time.time()
        gan_service.preload_static_face(
            face_path=AVATAR_IMAGE,
            fps=AVATAR_FPS,
            pads=(0, 50, 0, 0)
        )
        print(f"⚡ Предобработка аватара (GAN) завершена за {time.time() - preload_start:.2f}s")
    else:
        preload_start = time.time()
        gan_service.preload_video_cache(
            face_path=AVATAR_IMAGE,
            fps=AVATAR_FPS,
            pads=(0, 50, 0, 0)
        )
        print(f"⚡ Предобработка детекции (GAN) завершена за {time.time() - preload_start:.2f}s")

    if _is_video(AVATAR_VIDEO_PATH) and os.path.exists(AVATAR_VIDEO_PATH):
        need_dynamic_preload = use_static_cache or AVATAR_VIDEO_PATH != AVATAR_IMAGE
        if need_dynamic_preload:
            try:
                dynamic_start = time.time()
                gan_service.preload_video_cache(
                    face_path=AVATAR_VIDEO_PATH,
                    fps=AVATAR_FPS,
                    pads=(0, 50, 0, 0)
                )
                print(f"⚡ Предзагрузка динамического видео ({AVATAR_VIDEO_PATH}) завершена за {time.time() - dynamic_start:.2f}s")
            except Exception as dynamic_error:
                print(f"⚠️ Не удалось предзагрузить динамический аватар {AVATAR_VIDEO_PATH}: {dynamic_error}")

    nogan_service: Optional[LipsyncService] = None

    additional_gan_services = []
    for idx, device_name in enumerate(gan_devices[1:], start=2):
        print(f"\n📦 Загрузка GAN модели #{idx} (device={device_name})...")
        start = time.time()
        gan_extra = LipsyncService(
            checkpoint_path=CHECKPOINT_PATH_GAN,
            device=device_name,
            **common_kwargs
        )
        model_ready_time = time.time()
        print(f"✅ GAN-{idx} модель загружена за {model_ready_time - start:.2f}s")

        if use_static_cache:
            preload_start = time.time()
            gan_extra.preload_static_face(
                face_path=AVATAR_IMAGE,
                fps=AVATAR_FPS,
                pads=(0, 50, 0, 0)
            )
            print(f"⚡ Предобработка аватара (GAN-{idx}) завершена за {time.time() - preload_start:.2f}s")
        else:
            preload_start = time.time()
            gan_extra.preload_video_cache(
                face_path=AVATAR_IMAGE,
                fps=AVATAR_FPS,
                pads=(0, 50, 0, 0)
            )
            print(f"⚡ Предобработка детекции (GAN-{idx}) завершена за {time.time() - preload_start:.2f}s")

        if _is_video(AVATAR_VIDEO_PATH) and os.path.exists(AVATAR_VIDEO_PATH):
            need_dynamic_preload_extra = use_static_cache or AVATAR_VIDEO_PATH != AVATAR_IMAGE
            if need_dynamic_preload_extra:
                try:
                    dynamic_start = time.time()
                    gan_extra.preload_video_cache(
                        face_path=AVATAR_VIDEO_PATH,
                        fps=AVATAR_FPS,
                        pads=(0, 50, 0, 0)
                    )
                    print(f"⚡ Предзагрузка динамического видео (GAN-{idx}) завершена за {time.time() - dynamic_start:.2f}s")
                except Exception as dynamic_error:
                    print(f"⚠️ Не удалось предзагрузить динамический аватар {AVATAR_VIDEO_PATH} для GAN-{idx}: {dynamic_error}")

        additional_gan_services.append(gan_extra)

    print("\n🖼️  Предзагрузка аватара...")
    avatar_preloaded = None
    try:
        import cv2  # Local import to avoid unnecessary import during unit tests

        preview_saved = False
        if _is_video(AVATAR_IMAGE):
            capture = cv2.VideoCapture(AVATAR_IMAGE)
            success, frame = capture.read()
            capture.release()
            if success and frame is not None:
                avatar_preloaded = frame
                print(f"✅ Первый кадр видео-аватара: {avatar_preloaded.shape}")
                preview_saved = cv2.imwrite(AVATAR_PREVIEW_PATH, frame)
            else:
                print("⚠️ Не удалось считать первый кадр видео-аватара")
        else:
            avatar_preloaded = cv2.imread(AVATAR_IMAGE)
            if avatar_preloaded is not None:
                print(f"✅ Аватар загружен: {avatar_preloaded.shape}")
                preview_saved = cv2.imwrite(AVATAR_PREVIEW_PATH, avatar_preloaded)
            else:
                print("⚠️ Не удалось предзагрузить аватар в память")

        if preview_saved:
            print(f"🖼️  Превью аватара сохранено: {AVATAR_PREVIEW_PATH}")
        elif avatar_preloaded is not None:
            print("⚠️ Не удалось сохранить превью аватара")
    except ImportError:
        print("⚠️ OpenCV не установлен. Предзагрузка аватара пропущена.")

    total_gan_models = 1 + len(additional_gan_services)

    print("\n" + "=" * 60)
    print("✅ Сервис полностью готов к работе!")
    print(f"   🚀 Загружено моделей для параллельной обработки: {total_gan_models}x GAN" + (" + 1x NoGAN" if nogan_service else ""))
    print("=" * 60 + "\n")

    set_state(gan_service, nogan_service, avatar_preloaded, use_static_cache, *additional_gan_services)
    return gan_service, nogan_service, avatar_preloaded
