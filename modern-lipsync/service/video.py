from __future__ import annotations

from typing import List, Tuple
from pathlib import Path
import numpy as np
import cv2


class VideoMixin:
    img_size: int
    face_det_batch_size: int

    def get_smoothened_boxes(self, boxes: np.ndarray, T: int = 5) -> np.ndarray:
        """Сглаживание боксов детекции лица."""
        for i in range(len(boxes)):
            if i + T > len(boxes):
                window = boxes[len(boxes) - T:]
            else:
                window = boxes[i : i + T]
            boxes[i] = np.mean(window, axis=0)
        return boxes

    def detect_faces(
        self,
        images: List[np.ndarray],
        pads: Tuple[int, int, int, int] = (0, 10, 0, 0),
        nosmooth: bool = False,
    ) -> List[Tuple]:
        """Детекция лиц на батче кадров."""
        batch_size = self.face_det_batch_size

        while True:
            predictions = []
            try:
                for i in range(0, len(images), batch_size):
                    batch = np.array(images[i:i + batch_size])
                    predictions.extend(self.face_detector.get_detections_for_batch(batch))
            except RuntimeError as e:
                if batch_size == 1:
                    raise RuntimeError('Image too big for GPU')
                batch_size //= 2
                print(f'OOM recovery: reducing batch to {batch_size}')
                continue
            break

        results = []
        pady1, pady2, padx1, padx2 = pads

        for rect, image in zip(predictions, images):
            if rect is None:
                raise ValueError('Face not detected!')

            y1 = max(0, rect[1] - pady1)
            y2 = min(image.shape[0], rect[3] + pady2)
            x1 = max(0, rect[0] - padx1)
            x2 = min(image.shape[1], rect[2] + padx2)

            results.append([x1, y1, x2, y2])

        boxes = np.array(results)
        if not nosmooth:
            boxes = self.get_smoothened_boxes(boxes, T=5)

        results = [
            [image[y1:y2, x1:x2], (y1, y2, x1, x2)]
            for image, (x1, y1, x2, y2) in zip(images, boxes)
        ]

        return results

    def _load_video(
        self, face_path, static, fps, resize_factor, crop, rotate
    ) -> Tuple[List[np.ndarray], float]:
        """Загрузка кадров видео/изображения."""
        ext = Path(face_path).suffix.lower()

        if ext in ['.jpg', '.png', '.jpeg']:
            full_frames = [cv2.imread(face_path)]
            video_fps = fps
        else:
            video_stream = cv2.VideoCapture(face_path)
            video_fps = video_stream.get(cv2.CAP_PROP_FPS)

            full_frames = []
            while True:
                still_reading, frame = video_stream.read()
                if not still_reading:
                    video_stream.release()
                    break

                if resize_factor > 1:
                    frame = cv2.resize(
                        frame,
                        (frame.shape[1] // resize_factor,
                         frame.shape[0] // resize_factor)
                    )

                if rotate:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

                y1, y2, x1, x2 = crop
                if x2 == -1: x2 = frame.shape[1]
                if y2 == -1: y2 = frame.shape[0]

                frame = frame[y1:y2, x1:x2]
                full_frames.append(frame)

        return full_frames, video_fps

