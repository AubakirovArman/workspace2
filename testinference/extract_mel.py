#!/usr/bin/env python3
import argparse
import time
import json
import numpy as np
from pathlib import Path
import sys

# Добавляем путь к модулю modern-lipsync для импорта utils/audio
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'modern-lipsync'))

from utils.audio import ModernAudioProcessor


def main():
    parser = argparse.ArgumentParser(description='Extract mel spectrogram from audio')
    parser.add_argument('--audio', type=str, required=True, help='Путь к аудио (.wav/.mp3/...)')
    parser.add_argument('--out', type=str, required=True, help='Путь для сохранения mel.npy')
    parser.add_argument('--log', type=str, default=None, help='Путь к JSONL лог-файлу')
    args = parser.parse_args()

    t0 = time.time()
    proc = ModernAudioProcessor()
    mel = proc.extract_audio_features(args.audio)
    dt = time.time() - t0

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, mel)

    print(f'✅ Mel сохранен: {out_path}  форма={mel.shape}  время={dt:.2f}s')

    # Логирование результатов в JSONL
    if args.log:
        try:
            log_path = Path(args.log)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                'step': 'extract_mel',
                'time_sec': round(dt, 4),
                'out': str(out_path),
                'audio': args.audio,
                'mel_shape': list(mel.shape)
            }
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(payload, ensure_ascii=False) + '\n')
        except Exception as e:
            print(f"⚠️  Не удалось записать лог: {e}")


if __name__ == '__main__':
    main()
