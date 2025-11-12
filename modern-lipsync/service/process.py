from __future__ import annotations

from typing import Optional, Tuple, List, Callable
import time
import numpy as np
import cv2
import torch


class ProcessMixin:
    """Основной метод process() и сбор статистики."""

    def process(
        self,
        face_path: str,
        audio_path: str = "",
        output_path: Optional[str] = None,
        static: bool = False,
        fps: float = 25.0,
        pads: Tuple[int, int, int, int] = (0, 10, 0, 0),
        resize_factor: int = 1,
        crop: Tuple[int, int, int, int] = (0, -1, 0, -1),
        box: Tuple[int, int, int, int] = (-1, -1, -1, -1),
        rotate: bool = False,
        nosmooth: bool = False,
        audio_waveform: Optional[torch.Tensor] = None,
        audio_sample_rate: int = 16000,
        frame_sink: Optional[Callable[[np.ndarray], None]] = None,
        batch_size_override: Optional[int] = None,
        frame_offset: int = 0,
    ) -> dict:
        """Обработка одного запроса с предзагруженными моделями."""
        stats = {}
        total_start = time.time()

        # Разрешение использовать статический кэш только без заданного box
        can_use_cache = static and box == (-1, -1, -1, -1)
        cache_key = None
        cache_entry = None
        frames_for_cache = None
        video_cache_key = None
        video_cache_entry = None

        if can_use_cache:
            cache_key = self._static_cache_key(
                face_path, fps, pads, resize_factor, crop, rotate, nosmooth
            )
            cache_entry = self._static_cache.get(cache_key)
        elif box == (-1, -1, -1, -1):
            video_cache_key = self._video_cache_key(
                face_path, fps, pads, resize_factor, crop, rotate, nosmooth
            )
            video_cache_entry = self._video_cache.get(video_cache_key)

        static_face_resized = None
        if cache_entry:
            full_frames = list(cache_entry['frames'])
            video_fps = cache_entry['fps']
            stats['load_video_time'] = 0.0
            static_face_resized = cache_entry.get('resized_face')
        elif video_cache_entry and 'frames' in video_cache_entry:
            full_frames = list(video_cache_entry['frames'])
            video_fps = video_cache_entry.get('fps', fps)
            stats['load_video_time'] = 0.0
        else:
            start = time.time()
            full_frames, video_fps = self._load_video(
                face_path, static, fps, resize_factor, crop, rotate
            )
            stats['load_video_time'] = time.time() - start
            if can_use_cache:
                frames_for_cache = tuple(frame.copy() for frame in full_frames)
        if full_frames:
            frame_h, frame_w = full_frames[0].shape[:2]
            stats['frame_height'] = frame_h
            stats['frame_width'] = frame_w

        # Аудио
        start = time.time()
        mel, mel_chunks = self._process_audio(
            audio_path,
            video_fps,
            audio_waveform,
            audio_sample_rate,
            write_temp_wav=(output_path is not None),
        )
        stats['process_audio_time'] = time.time() - start
        stats['num_mel_chunks'] = len(mel_chunks)

        # Ограничиваем кадры длиной аудио
        # В dynamic mode система зациклит кадры через i % len(full_frames)
        # Если аудио короче видео, возьмутся первые N кадров
        # Если аудио длиннее видео, кадры зациклятся
        full_frames = full_frames[:len(mel_chunks)]
        stats['num_frames'] = len(full_frames)
        stats['fps'] = video_fps

        # Детекция лица
        if cache_entry:
            face_det_results = [[cache_entry['face'].copy(), cache_entry['coords']]]
            stats['face_detection_time'] = 0.0
            if static_face_resized is None and 'resized_face' in cache_entry:
                static_face_resized = cache_entry['resized_face']
        elif video_cache_entry and not static and box == (-1, -1, -1, -1):
            coords_list = video_cache_entry['coords']
            if len(coords_list) >= len(full_frames):
                coords_list = coords_list[:len(full_frames)]
                face_det_results = [
                    [frame[y1:y2, x1:x2], (y1, y2, x1, x2)]
                    for frame, (y1, y2, x1, x2) in zip(full_frames, coords_list)
                ]
                stats['face_detection_time'] = 0.0
            else:
                video_cache_entry = None

        if 'face_det_results' not in locals():
            start = time.time()
            if box[0] == -1:
                if not static:
                    face_det_results = self.detect_faces(full_frames, pads, nosmooth)
                else:
                    face_det_results = self.detect_faces([full_frames[0]], pads, nosmooth)
            else:
                y1, y2, x1, x2 = box
                face_det_results = [[f[y1:y2, x1:x2], (y1, y2, x1, x2)] for f in full_frames]
            detect_time = time.time() - start
            stats['face_detection_time'] = detect_time

            if can_use_cache and frames_for_cache is not None:
                cached_face, cached_coords = face_det_results[0]
                self._static_cache[cache_key] = {
                    'frames': frames_for_cache,
                    'fps': video_fps,
                    'face': cached_face.copy(),
                    'coords': cached_coords,
                    'load_video_time': stats['load_video_time'],
                    'face_detection_time': detect_time,
                    'resized_face': cv2.resize(cached_face, (self.img_size, self.img_size)),
                }
                static_face_resized = self._static_cache[cache_key]['resized_face']
            elif not static and video_cache_key and box == (-1, -1, -1, -1):
                coords_list = [coords for _, coords in face_det_results]
                self._video_cache[video_cache_key] = {
                    'coords': coords_list,
                    'fps': video_fps,
                    'num_frames': len(coords_list),
                    'face_detection_time': detect_time,
                }

        # Инференс
        start = time.time()
        self._run_inference(
            full_frames,
            face_det_results,
            mel_chunks,
            output_path,
            video_fps,
            static,
            static_face_resized,
            frame_sink=frame_sink,
            batch_size_override=batch_size_override,
            frame_offset=frame_offset,
        )
        stats['inference_time'] = time.time() - start

        breakdown = getattr(self, '_last_inference_breakdown', None)
        if isinstance(breakdown, dict):
            stats.update(breakdown)

        stats['total_time'] = time.time() - total_start
        stats['segmentation'] = bool(self.segmentation_enabled and self.segmentation_model is not None)
        stats['super_resolution'] = bool(self.sr_enabled and self.sr_model is not None)
        stats['real_esrgan'] = bool(self.realesrgan_enabled and self._realesrgan_enhancer is not None)

        return stats
