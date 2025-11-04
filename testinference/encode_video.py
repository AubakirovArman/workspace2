#!/usr/bin/env python3
import argparse
import time
import json
import subprocess
from pathlib import Path
import cv2


def select_codec(fast: bool = False) -> tuple[str, list[str]]:
    """Выбор кодека без NVENC: используем только libx264."""
    if fast:
        return 'libx264', ['-preset', 'ultrafast', '-tune', 'zerolatency', '-crf', '28']
    return 'libx264', ['-crf', '23', '-preset', 'veryfast']


def main():
    parser = argparse.ArgumentParser(description='Encode frames dir + audio into mp4')
    parser.add_argument('--frames', type=str, required=True, help='Директория собранных кадров')
    parser.add_argument('--audio', type=str, required=True, help='Путь к аудио')
    parser.add_argument('--fps', type=float, default=25.0, help='FPS')
    parser.add_argument('--out', type=str, required=True, help='Выходной mp4')
    parser.add_argument('--fast', action='store_true', help='Ускоренное кодирование (низкая задержка)')
    parser.add_argument('--log', type=str, default=None, help='Путь к JSONL лог-файлу')
    args = parser.parse_args()

    frames_dir = Path(args.frames)
    files = sorted(frames_dir.glob('*.png'))
    if not files:
        raise RuntimeError('Нет кадров для кодирования')

    # Проверим размеры кадра
    sample = cv2.imread(str(files[0]))
    h, w = sample.shape[:2]
    # ffmpeg фильтр гарантирует чётные размеры
    vf = 'scale=w=trunc(iw/2)*2:h=trunc(ih/2)*2:flags=bicubic,format=yuv420p'
    codec, codec_opts = select_codec(fast=bool(args.fast))

    # Собираем список кадров через glob pattern
    # Используем ffmpeg для последовательности изображений
    # Прим.: %06d.png соответствует имени файлов
    input_pattern = str(frames_dir / '%06d.png')

    cmd = [
        'ffmpeg', '-y',
        '-r', str(args.fps), '-i', input_pattern,
        '-i', args.audio,
        '-vf', vf,
        '-c:v', codec,
    ] + codec_opts + [
        '-pix_fmt', 'yuv420p',
        '-c:a', 'aac', '-ar', '16000', '-b:a', '128k',
        '-shortest', '-movflags', '+faststart',
        args.out,
        '-loglevel', 'error'
    ]

    t0 = time.time()
    proc = subprocess.run(cmd)
    dt = time.time() - t0
    if proc.returncode != 0:
        raise RuntimeError('ffmpeg encoding failed')
    print(f'✅ Видео закодировано: {args.out}  {w}x{h}@{args.fps}  кодек={codec}  время={dt:.2f}s')

    # Логирование результатов в JSONL
    if args.log:
        try:
            log_path = Path(args.log)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                'step': 'encode_video',
                'time_sec': round(dt, 4),
                'codec': codec,
                'fps': args.fps,
                'resolution': {'w': w, 'h': h},
                'out': args.out,
                'fast': bool(args.fast)
            }
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(payload, ensure_ascii=False) + '\n')
        except Exception as e:
            print(f"⚠️  Не удалось записать лог: {e}")


if __name__ == '__main__':
    main()
