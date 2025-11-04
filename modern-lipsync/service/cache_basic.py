from __future__ import annotations

from typing import Tuple
from pathlib import Path
import numpy as np
import cv2


class CacheBasicMixin:
    img_size: int

    def process_with_preloaded(
        self,
        audio_path: str,
        output_path: str | None = None,
        fps: float = 25.0,
        audio_waveform=None,
        audio_sample_rate: int = 16000,
        frame_sink=None,
        batch_size_override: int | None = None,
    ) -> dict:
        """Быстрая обработка на предзагруженном статическом лице."""
        if not self._static_cache:
            raise RuntimeError(
                "Статическое лицо не предзагружено. Сначала вызовите preload_static_face() "
                "или используйте process() с static=True"
            )

        cache_entry = next(iter(self._static_cache.values()))

        stats = {}
        import time
        total_start = time.time()

        full_frames = list(cache_entry['frames'])
        video_fps = cache_entry.get('fps', fps)
        cached_face = cache_entry['face'].copy()
        cached_coords = cache_entry['coords']
        static_face_resized = cache_entry.get('resized_face')

        stats['load_video_time'] = 0.0
        stats['face_detection_time'] = 0.0

        start = time.time()
        mel, mel_chunks = self._process_audio(
            audio_path, video_fps, audio_waveform, audio_sample_rate,
            write_temp_wav=(output_path is not None),
        )
        stats['process_audio_time'] = time.time() - start
        stats['num_mel_chunks'] = len(mel_chunks)

        full_frames = full_frames[:len(mel_chunks)]
        stats['num_frames'] = len(full_frames)
        stats['fps'] = video_fps

        face_det_results = [[cached_face, cached_coords]]

        start = time.time()
        self._run_inference(
            full_frames,
            face_det_results,
            mel_chunks,
            output_path,
            video_fps,
            True,
            static_face_resized,
            frame_sink=frame_sink,
            batch_size_override=batch_size_override,
        )
        stats['inference_time'] = time.time() - start

        stats['postprocess_time'] = 0.0
        stats['write_time'] = 0.0
        stats['total_time'] = time.time() - total_start
        stats['segmentation'] = self.segmentation_enabled
        stats['super_resolution'] = self.sr_enabled
        stats['real_esrgan'] = self.realesrgan_enabled

        return stats

    def preload_static_face(
        self,
        face_path: str,
        fps: float = 25.0,
        pads: Tuple[int, int, int, int] = (0, 10, 0, 0),
        resize_factor: int = 1,
        crop: Tuple[int, int, int, int] = (0, -1, 0, -1),
        rotate: bool = False,
        nosmooth: bool = False,
    ):
        """Предзагрузка данных для статического лица."""
        cache_key = self._static_cache_key(
            face_path, fps, pads, resize_factor, crop, rotate, nosmooth
        )
        if cache_key in self._static_cache:
            return

        full_frames, video_fps = self._load_video(
            face_path, True, fps, resize_factor, crop, rotate
        )
        frames_for_cache = tuple(frame.copy() for frame in full_frames)
        face_det_results = self.detect_faces([full_frames[0]], pads, nosmooth)
        cached_face, cached_coords = face_det_results[0]
        self._static_cache[cache_key] = {
            'frames': frames_for_cache,
            'fps': video_fps,
            'face': cached_face.copy(),
            'coords': cached_coords,
            'load_video_time': 0.0,
            'face_detection_time': 0.0,
            'resized_face': cv2.resize(cached_face, (self.img_size, self.img_size)),
        }

    def _static_cache_key(
        self,
        face_path: str,
        fps: float,
        pads: Tuple[int, int, int, int],
        resize_factor: int,
        crop: Tuple[int, int, int, int],
        rotate: bool,
        nosmooth: bool,
    ) -> Tuple:
        resolved_path = str(Path(face_path).resolve())
        return (
            resolved_path,
            float(fps),
            tuple(int(p) for p in pads),
            int(resize_factor),
            tuple(int(c) for c in crop),
            bool(rotate),
            bool(nosmooth),
        )
