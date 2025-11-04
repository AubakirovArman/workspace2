#!/usr/bin/env python3
import argparse
import time
from pathlib import Path
import cv2


def main():
    parser = argparse.ArgumentParser(description='Load avatar frames (video or single image)')
    parser.add_argument('--face', type=str, required=True, help='Путь к видео/картинке аватара')
    parser.add_argument('--fps', type=float, default=25.0, help='FPS для статики')
    parser.add_argument('--resize_factor', type=int, default=1, help='Фактор уменьшения разрешения')
    parser.add_argument('--crop', nargs='+', type=int, default=[0, -1, 0, -1], help='Обрезка top bottom left right')
    parser.add_argument('--rotate', action='store_true', help='Повернуть видео на 90° по часовой')
    parser.add_argument('--out', type=str, required=True, help='Директория для сохранения кадров PNG')
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    ext = Path(args.face).suffix.lower()
    t0 = time.time()
    frames = []

    if ext in ['.jpg', '.png', '.jpeg']:
        frame = cv2.imread(args.face)
        y1, y2, x1, x2 = args.crop
        if x2 == -1:
            x2 = frame.shape[1]
        if y2 == -1:
            y2 = frame.shape[0]
        frame = frame[y1:y2, x1:x2]
        frames = [frame]
        fps = args.fps
    else:
        cap = cv2.VideoCapture(args.face)
        fps = cap.get(cv2.CAP_PROP_FPS) or args.fps
        while True:
            ok, frame = cap.read()
            if not ok:
                cap.release()
                break
            if args.resize_factor > 1:
                frame = cv2.resize(
                    frame,
                    (frame.shape[1] // args.resize_factor, frame.shape[0] // args.resize_factor)
                )
            if args.rotate:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            y1, y2, x1, x2 = args.crop
            if x2 == -1:
                x2 = frame.shape[1]
            if y2 == -1:
                y2 = frame.shape[0]
            frame = frame[y1:y2, x1:x2]
            frames.append(frame)

    # Save frames
    for i, f in enumerate(frames):
        cv2.imwrite(str(out_dir / f'{i:06d}.png'), f)

    dt = time.time() - t0
    h, w = frames[0].shape[:2]
    print(f'✅ Кадры загружены: {len(frames)} шт., {w}x{h} @ {fps:.2f} fps, dir={out_dir}, время={dt:.2f}s')


if __name__ == '__main__':
    main()

