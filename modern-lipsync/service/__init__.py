"""
Service package: разбивка большого класса на миксины и модули.
Сохраняет совместимость импорта: `from service import LipsyncService`.
"""
from __future__ import annotations

from typing import Optional, Tuple, List, Callable
import os
import time
import torch

# Миксины с реализацией методов
from .models import ModelsMixin
from .video import VideoMixin
from .audio import AudioMixin
from .ffmpeg_codec import FfmpegCodecMixin
from .ffmpeg_io import FfmpegIOMixin
from .ffmpeg_infer import FfmpegInferMixin
from .cache_basic import CacheBasicMixin
from .cache_video import CacheVideoMixin
from .process import ProcessMixin


class LipsyncService(
    ModelsMixin,
    VideoMixin,
    AudioMixin,
    FfmpegCodecMixin,
    FfmpegIOMixin,
    FfmpegInferMixin,
    CacheBasicMixin,
    CacheVideoMixin,
    ProcessMixin,
):
    """
    Preloaded lipsync service for fast repeated inference.
    Модели держатся в памяти между запросами.
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = 'cuda',
        face_det_batch_size: int = 16,
        wav2lip_batch_size: int = 512,
        amp_dtype: str = 'fp16',
        segmentation_path: Optional[str] = None,
        sr_path: Optional[str] = None,
        modules_root: Optional[str] = None,
        realesrgan_path: Optional[str] = None,
        realesrgan_outscale: float = 1.0,
        use_fp16: bool = True,
        use_compile: bool = True,
        ffmpeg_threads: Optional[int] = None,
        ffmpeg_filter_threads: Optional[int] = None,
    ):
        # Базовые параметры
        self.device = device
        self.is_cuda = str(device).startswith('cuda')
        self.face_det_batch_size = face_det_batch_size
        self.wav2lip_batch_size = wav2lip_batch_size
        self.img_size = 96
        self.mel_step_size = 16
        self._static_cache = {}
        self._video_cache = {}

        # Доп.модули
        self.segmentation_model = None
        self.segmentation_enabled = False
        self._swap_regions_fn: Optional[Callable] = None
        self.sr_model = None
        self.sr_enabled = False
        self._sr_enhance_fn: Optional[Callable] = None
        self._modules_root = modules_root
        self._segmentation_path = segmentation_path
        self._sr_path = sr_path
        self._realesrgan_path = realesrgan_path
        self._realesrgan_outscale = max(1.0, float(realesrgan_outscale))
        self._sys_path_added = False
        self.realesrgan_enabled = False
        self._realesrgan_enhancer = None

        # Настройки FFmpeg
        if ffmpeg_threads is not None:
            self.ffmpeg_threads = int(ffmpeg_threads)
        else:
            env_threads = os.getenv("FFMPEG_THREADS")
            try:
                self.ffmpeg_threads = int(env_threads) if env_threads is not None else 16
            except (TypeError, ValueError):
                self.ffmpeg_threads = 16

        if ffmpeg_filter_threads is not None:
            self.ffmpeg_filter_threads = max(0, int(ffmpeg_filter_threads))
        else:
            env_filter_threads = os.getenv("FFMPEG_FILTER_THREADS")
            try:
                self.ffmpeg_filter_threads = max(0, int(env_filter_threads)) if env_filter_threads else 0
            except (TypeError, ValueError):
                self.ffmpeg_filter_threads = 0
        self._last_inference_breakdown = {}

        # Оптимизации
        self.use_fp16 = use_fp16 and self.is_cuda
        self.use_compile = use_compile
        # Выбор типа смешанной точности (FP16/BF16)
        try:
            amp_opt = (amp_dtype or 'fp16').lower()
        except Exception:
            amp_opt = 'fp16'
        self.amp_dtype = torch.float16 if amp_opt == 'fp16' else torch.bfloat16

        # Включаем CUDA оптимизации
        if self.is_cuda:
            try:
                torch.cuda.set_device(torch.device(self.device))
            except Exception:
                pass
            torch.backends.cudnn.benchmark = True
            print("✓ CuDNN benchmark mode enabled")
            if self.use_fp16:
                # Для CUDA включаем высокую точность матмулов в FP32 (ускоряет смешанный режим)
                if hasattr(torch, 'set_float32_matmul_precision'):
                    torch.set_float32_matmul_precision('high')
                    print("✓ Float32 matmul precision set to high")
        # Сброс денормалов для предотвращения NaN/Inf источников
        try:
            torch.set_flush_denormal(True)
            print("✓ Flush denormals enabled")
        except Exception:
            pass

        print(f"🚀 Starting Lipsync Service on {device}...")

        # Предзагрузка моделей
        start = time.time()
        self._load_models(checkpoint_path)
        load_time = time.time() - start

        print(f"✅ Service ready! Models loaded in {load_time:.2f}s")
        print(f"   - Face Detector: Ready")
        print(f"   - Wav2Lip Model: Ready")
        print(f"   - Audio Processor: Ready")
        print(f"   - Batch Size: {self.wav2lip_batch_size}")
        if self.use_fp16:
            amp_name = 'FP16' if self.amp_dtype == torch.float16 else 'BF16'
            print(f"   - AMP Mode: {amp_name}")
        if self.use_compile:
            print(f"   - Torch Compile: Enabled")


__all__ = ["LipsyncService"]
