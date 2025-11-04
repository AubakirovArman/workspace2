from __future__ import annotations

from typing import Tuple, List
import numpy as np
import cv2


class CacheVideoMixin:
    def _video_cache_key(
        self,
        face_path: str,
        fps: float,
        pads: Tuple[int, int, int, int],
        resize_factor: int,
        crop: Tuple[int, int, int, int],
        rotate: bool,
        nosmooth: bool,
    ) -> Tuple:
        return ("video",) + self._static_cache_key(
            face_path, fps, pads, resize_factor, crop, rotate, nosmooth
        )

    def preload_video_cache(
        self,
        face_path: str,
        fps: float = 25.0,
        pads: Tuple[int, int, int, int] = (0, 10, 0, 0),
        resize_factor: int = 1,
        crop: Tuple[int, int, int, int] = (0, -1, 0, -1),
        rotate: bool = False,
        nosmooth: bool = False,
    ) -> None:
        """Предварительный расчёт детекции лица для видео, чтобы переиспользовать координаты."""
        cache_key = self._video_cache_key(
            face_path, fps, pads, resize_factor, crop, rotate, nosmooth
        )
        if cache_key in self._video_cache:
            return

        full_frames, video_fps = self._load_video(
            face_path, False, fps, resize_factor, crop, rotate
        )
        if not full_frames:
            raise ValueError("Video does not contain any frames")

        usable_frames: List[np.ndarray] = []
        coords: List[Tuple[int, int, int, int]] = []
        detect_time = 0.0

        scales_to_try = [1, 2, 3, 4]
        detection_success = False

        import time
        for scale in scales_to_try:
            frames_for_detection = full_frames
            scale_start = time.time()
            if scale > 1:
                frames_for_detection = []
                for frame in full_frames:
                    h, w = frame.shape[:2]
                    new_w = max(1, w // scale)
                    new_h = max(1, h // scale)
                    frames_for_detection.append(
                        cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
                    )
            try:
                detect_start = time.time()
                face_det_results = self.detect_faces(frames_for_detection, pads, nosmooth)
                detect_duration = time.time() - detect_start
                detect_time += detect_duration + (time.time() - scale_start - detect_duration)

                for original_frame, (_, coord_set) in zip(full_frames, face_det_results):
                    if scale > 1:
                        y1, y2, x1, x2 = coord_set
                        y1, y2 = int(y1 * scale), int(y2 * scale)
                        x1, x2 = int(x1 * scale), int(x2 * scale)
                        coord_set = (y1, y2, x1, x2)
                    usable_frames.append(original_frame.copy())
                    coords.append(coord_set)
                detection_success = True
                break
            except RuntimeError as runtime_err:
                if "Image too big for GPU" in str(runtime_err) and scale < scales_to_try[-1]:
                    print(f"⚠️ Face detection OOM at scale {scale}. Retrying with scale {scale + 1}...")
                    continue
                raise
            except ValueError:
                detect_time += time.time() - scale_start
                break

        if not detection_success and not coords:
            print("⚠️ Bulk face detection failed; retrying frame-by-frame with fallbacks")
            for idx, frame in enumerate(full_frames):
                success = False
                for scale in scales_to_try:
                    scaled_frame = frame
                    if scale > 1:
                        h, w = frame.shape[:2]
                        new_w = max(1, w // scale)
                        new_h = max(1, h // scale)
                        scaled_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
                    try:
                        frame_start = time.time()
                        result = self.detect_faces([scaled_frame], pads, nosmooth)[0]
                        detect_time += time.time() - frame_start
                        y1, y2, x1, x2 = result[1]
                        if scale > 1:
                            y1, y2 = int(y1 * scale), int(y2 * scale)
                            x1, x2 = int(x1 * scale), int(x2 * scale)
                        usable_frames.append(frame.copy())
                        coords.append((y1, y2, x1, x2))
                        success = True
                        break
                    except (ValueError, RuntimeError):
                        continue
                if not success:
                    print(f"   ⚠️ Пропущен кадр #{idx}: лицо не найдено ни при одном масштабировании")

        if not usable_frames:
            raise ValueError("Face not detected in any frame of the video avatar")

        self._video_cache[cache_key] = {
            "coords": coords,
            "frames": tuple(usable_frames),
            "fps": video_fps,
            "num_frames": len(coords),
            "face_detection_time": detect_time,
        }

