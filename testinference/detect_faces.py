#!/usr/bin/env python3
import argparse
import time
import json
from pathlib import Path
import numpy as np
import cv2
import sys

# Добавляем путь к modern-lipsync для импорта face_detection
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'modern-lipsync'))

import face_detection


def main():
    parser = argparse.ArgumentParser(description='Batch face detection for frames_dir')
    parser.add_argument('--frames', type=str, required=True, help='Директория с кадрами PNG')
    parser.add_argument('--pads', nargs='+', type=int, default=[0, 10, 0, 0], help='Паддинги y1 y2 x1 x2')
    parser.add_argument('--device', type=str, default='cuda', help='cuda или cpu')
    parser.add_argument('--out', type=str, required=True, help='Путь для сохранения coords.npy')
    parser.add_argument('--log', type=str, default=None, help='Путь к JSONL лог-файлу')
    args = parser.parse_args()

    pady1, pady2, padx1, padx2 = args.pads
    frames_dir = Path(args.frames)
    files = sorted(frames_dir.glob('*.png'))
    if not files:
        raise RuntimeError('Нет кадров в frames_dir')

    # Load frames into memory (batching done by detector)
    images = [cv2.imread(str(p)) for p in files]

    # Check if local S3FD weights exist to avoid network download; else fallback
    s3fd_path = ROOT / 'modern-lipsync' / 'face_detection' / 'detection' / 'sfd' / 's3fd.pth'
    detector = None
    preds = [None] * len(images)
    t0 = time.time()
    if s3fd_path.exists():
        try:
            detector = face_detection.FaceAlignment(
                face_detection.LandmarksType._2D,
                flip_input=False,
                device=args.device
            )
            batch = np.array(images)
            preds = detector.get_detections_for_batch(batch)
        except Exception as e:
            print(f"⚠️  Face detector failed ({e}); using full-frame boxes.")
    else:
        print("⚠️  s3fd.pth not found; using full-frame boxes.")

    coords = []
    for rect, image in zip(preds, images):
        if rect is None:
            coords.append([0, image.shape[0], 0, image.shape[1]])
            continue
        y1 = max(0, rect[1] - pady1)
        y2 = min(image.shape[0], rect[3] + pady2)
        x1 = max(0, rect[0] - padx1)
        x2 = min(image.shape[1], rect[2] + padx2)
        coords.append([y1, y2, x1, x2])

    dt = time.time() - t0
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, np.array(coords, dtype=np.int32))

    print(f'✅ Координаты лиц сохранены: {out_path}  кадры={len(images)}  время={dt:.2f}s')

    # Логирование результатов в JSONL
    if args.log:
        try:
            log_path = Path(args.log)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                'step': 'detect_faces',
                'time_sec': round(dt, 4),
                'frames': len(images),
                'device': args.device,
                'pads': args.pads,
                's3fd_present': s3fd_path.exists(),
                'weights_path': str(s3fd_path) if s3fd_path.exists() else None,
                'out': str(out_path)
            }
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(payload, ensure_ascii=False) + '\n')
        except Exception as e:
            print(f"⚠️  Не удалось записать лог: {e}")


if __name__ == '__main__':
    main()
