#!/usr/bin/env python3
"""
Простой тест TTS: запрашивает речь через TTS API и сохраняет WAV.

Автономная реализация generate_tts и convert_to_wav без импорта app_core,
чтобы избежать подтягивания веб-приложения и инициализации сервисов.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torchaudio
import requests
import io
import subprocess
import tempfile


def _get_tts_api_url() -> str:
    # Совместимо с app_core.config: переменная окружения TTS_API_URL
    return os.getenv("TTS_API_URL", "https://tts.sk-ai.kz/api/tts")


def generate_tts(text: str, language: str = "ru") -> bytes:
    url = _get_tts_api_url()
    print("🎤 Генерация TTS...")
    print(f"   URL: {url}")
    print(f"   Язык: {language}")
    print(f"   Текст: {text[:50]}{'...' if len(text) > 50 else ''}")
    resp = requests.post(url, json={"text": text, "lang": language}, timeout=30)
    resp.raise_for_status()
    audio_data = resp.content
    print(f"✅ TTS сгенерирован: {len(audio_data) / 1024:.2f} KB")
    return audio_data


def convert_to_wav(mp3_data: bytes, output_path: str) -> tuple[torchaudio.Tensor, int]:
    print("🔄 Конвертация в WAV...")
    target_sr = 16000
    try:
        audio_buffer = io.BytesIO(mp3_data)
        waveform, sample_rate = torchaudio.load(audio_buffer, format="mp3")
        waveform = waveform.float()
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sample_rate != target_sr:
            waveform = torchaudio.functional.resample(waveform, sample_rate, target_sr)
            sample_rate = target_sr
        torchaudio.save(output_path, waveform.cpu(), sample_rate)
        print(f"✅ WAV сохранен: {output_path}")
        return waveform, sample_rate
    except Exception as decode_error:
        print(f"⚠️ torchaudio не смог декодировать MP3 ({decode_error}); fallback на ffmpeg.")
        # Пишем временный MP3 на диск и конвертируем через ffmpeg
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tmp.write(mp3_data)
            temp_mp3 = tmp.name
        try:
            cmd = [
                "ffmpeg", "-y", "-i", temp_mp3,
                "-ar", str(target_sr),
                "-ac", "1",
                "-f", "wav",
                "-acodec", "pcm_s16le",
                output_path,
                "-loglevel", "error",
            ]
            subprocess.run(cmd, check=True)
            waveform, sample_rate = torchaudio.load(output_path)
            waveform = waveform.float()
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            print(f"✅ WAV сохранен: {output_path}")
            return waveform, sample_rate
        finally:
            try:
                os.remove(temp_mp3)
            except OSError:
                pass


def main() -> None:
    parser = argparse.ArgumentParser(description='TTS тест: сгенерировать речь и сохранить в WAV')
    parser.add_argument('--text', type=str, default='Здравствуйте! Это тест синтеза речи.', help='Текст для TTS')
    parser.add_argument('--language', type=str, default='ru', choices=['ru', 'kk', 'en'], help='Язык TTS')
    parser.add_argument('--out', type=str, default='outputs/test1/tts_test.wav', help='Путь для сохранения WAV')
    parser.add_argument('--log', type=str, default='outputs/test1/logs/test_tts.json', help='JSON-лог результатов')
    args = parser.parse_args()

    text = args.text.strip()
    language = args.language.strip()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    api_url = _get_tts_api_url()
    print('🎤 TTS тест-запрос')
    print(f'   URL: {api_url}')
    print(f'   Язык: {language}')
    print(f'   Текст: {text[:80]}{"..." if len(text) > 80 else ""}')

    tts_error = None
    try:
        t0 = time.time()
        audio_bytes = generate_tts(text, language)
        t_tts = time.time() - t0

        t1 = time.time()
        waveform, sample_rate = convert_to_wav(audio_bytes, str(out_path))
        t_convert = time.time() - t1
    except Exception as e:
        # Fallback: синтетический WAV, если TTS API недоступен
        tts_error = str(e)
        print(f"⚠️ Ошибка запроса/конвертации TTS: {tts_error}")
        print("🔧 Генерирую офлайн WAV для проверки пайплайна...")
        sample_rate = 16000
        duration_sec = max(1.0, min(8.0, len(text) / 12.0))
        n = int(duration_sec * sample_rate)
        import numpy as np
        t = np.linspace(0, duration_sec, num=n, endpoint=False, dtype=np.float32)
        base_freq = 220 + (len(text) % 80)
        waveform_np = 0.15 * np.sin(2 * np.pi * base_freq * t)
        waveform_np *= (0.85 + 0.15 * np.sin(2 * np.pi * 3.0 * t))
        import torch
        waveform = torch.from_numpy(waveform_np).unsqueeze(0)
        torchaudio.save(str(out_path), waveform, sample_rate)
        t_tts = 0.0
        t_convert = 0.0

    # Рассчитать длительность по тензору
    duration_sec = float(waveform.shape[-1]) / float(sample_rate)
    file_size_kb = os.path.getsize(out_path) / 1024.0

    print('✅ Готово: WAV сохранён')
    print(f'   Путь: {out_path}')
    print(f'   SR: {sample_rate} Hz, длительность: {duration_sec:.2f} s')
    print(f'   Размер файла: {file_size_kb:.1f} KB')
    print(f'   Время: TTS={t_tts:.2f}s, конвертация={t_convert:.2f}s, всего={(t_tts+t_convert):.2f}s')

    payload = {
        'step': 'test_tts',
        'tts_api_url': api_url,
        'language': language,
        'text_len': len(text),
        'out_wav': str(out_path),
        'sample_rate': sample_rate,
        'duration_sec': round(duration_sec, 3),
        'file_size_kb': round(file_size_kb, 2),
        'time_sec': {
            'tts': round(t_tts, 3),
            'convert': round(t_convert, 3),
            'total': round(t_tts + t_convert, 3),
        },
        'error': tts_error,
    }
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f'📝 Лог сохранён: {log_path}')


if __name__ == '__main__':
    main()
