#!/usr/bin/env python3
import argparse
import time
import json
import numpy as np
from pathlib import Path


def chunk_mel(mel: np.ndarray, fps: float, mel_step_size: int) -> np.ndarray:
    """Разбить mel (n_mels, n_frames) на чанки длиной mel_step_size под заданный FPS.
    Использует правило из сервиса: mel_idx_multiplier = 80.0 / fps.
    """
    chunks = []
    mel_idx_multiplier = 80.0 / fps
    i = 0

    while True:
        start_idx = int(i * mel_idx_multiplier)
        if start_idx + mel_step_size > mel.shape[1]:
            chunks.append(mel[:, mel.shape[1] - mel_step_size:])
            break
        chunks.append(mel[:, start_idx:start_idx + mel_step_size])
        i += 1

    return np.asarray(chunks)


def main():
    parser = argparse.ArgumentParser(description='Chunk mel spectrogram into windows')
    parser.add_argument('--mel', type=str, required=True, help='Путь к mel.npy')
    parser.add_argument('--fps', type=float, required=True, help='FPS исходного видео/статической картинки')
    parser.add_argument('--step', type=int, default=16, help='mel_step_size (по умолчанию 16)')
    parser.add_argument('--out', type=str, required=True, help='Путь для сохранения mel_chunks.npy')
    parser.add_argument('--log', type=str, default=None, help='Путь к JSONL лог-файлу')
    args = parser.parse_args()

    mel = np.load(args.mel)
    t0 = time.time()
    chunks = chunk_mel(mel, args.fps, args.step)
    dt = time.time() - t0

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, chunks)

    print(f'✅ Чанки mel сохранены: {out_path}  количество={len(chunks)}  форма_чанка={chunks[0].shape}  время={dt:.2f}s')

    # Логирование результатов в JSONL
    if args.log:
        try:
            log_path = Path(args.log)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                'step': 'chunk_mel',
                'time_sec': round(dt, 4),
                'out': str(out_path),
                'chunks_count': int(len(chunks)),
                'fps': args.fps,
                'mel_step_size': args.step,
                'chunk_shape': list(chunks[0].shape)
            }
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(payload, ensure_ascii=False) + '\n')
        except Exception as e:
            print(f"⚠️  Не удалось записать лог: {e}")


if __name__ == '__main__':
    main()
