#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def analyze_gray(frames_dir: Path, coords_path: Path, sample: int = 24) -> dict:
    files = sorted(frames_dir.glob('*.png'))
    if not files:
        raise RuntimeError('Нет кадров для проверки')

    coords = np.load(str(coords_path))
    n = min(len(files), len(coords))
    if n == 0:
        raise RuntimeError('Совпадений кадров и координат нет')

    sample_count = min(sample, n)
    idxs = np.linspace(0, n - 1, sample_count).astype(int)

    mean_sat_thresh = 15.0
    std_bgr_thresh = 12.0

    flagged = 0
    stats = []
    for i in idxs:
        img = cv2.imread(str(files[i]))
        if img is None:
            continue
        y1, y2, x1, x2 = coords[i]
        y1, y2, x1, x2 = int(y1), int(y2), int(x1), int(x2)
        y1 = max(0, y1)
        x1 = max(0, x1)
        y2 = min(img.shape[0], y2)
        x2 = min(img.shape[1], x2)
        roi = img[y1:y2, x1:x2]
        if roi.size == 0:
            continue

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        sat = hsv[..., 1]
        mean_sat = float(np.mean(sat))
        std_bgr = float(np.mean(np.std(roi.reshape(-1, 3), axis=0)))
        is_gray = (mean_sat < mean_sat_thresh) and (std_bgr < std_bgr_thresh)
        flagged += int(is_gray)
        stats.append({
            'frame': int(i),
            'mean_sat': round(mean_sat, 2),
            'std_bgr': round(std_bgr, 2),
            'is_gray': bool(is_gray)
        })

    summary = {
        'checked_frames': int(len(idxs)),
        'flagged_gray': int(flagged),
        'flagged_ratio': round(flagged / max(1, len(idxs)), 4),
        'thresholds': {
            'mean_sat': mean_sat_thresh,
            'std_bgr': std_bgr_thresh,
        },
        'details': stats,
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description='Проверка серости ROI во вставленных кадрах')
    parser.add_argument('--frames', type=str, required=True, help='Директория объединённых кадров (после вставки патчей)')
    parser.add_argument('--coords', type=str, required=True, help='coords.npy для ROI')
    parser.add_argument('--sample', type=int, default=24, help='Сколько кадров проверять (равномерная выборка)')
    parser.add_argument('--out', type=str, default=None, help='Путь для JSON отчёта')
    args = parser.parse_args()

    frames_dir = Path(args.frames)
    coords_path = Path(args.coords)
    summary = analyze_gray(frames_dir, coords_path, sample=args.sample)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()

