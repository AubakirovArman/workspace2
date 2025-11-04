"""Text-to-speech helpers and audio conversion utilities.

Изменения:
- convert_to_wav умеет читать аудио из BytesIO без записи на диск
  (через soundfile, с фолбэком на torchaudio). Сохранение на диск
  выполняется только если передан output_path.
"""
from __future__ import annotations

import io
import os
import time
import subprocess
from typing import Tuple

import requests
import torchaudio
import torchaudio.functional as audio_fn
import soundfile as sf
import torch

from ..config import TEMP_DIR, TTS_API_URL


def generate_tts(text: str, language: str = 'ru') -> bytes:
    print('🎤 Генерация TTS...')
    print(f'   Текст: {text[:50]}{"..." if len(text) > 50 else ""}')
    print(f'   Язык: {language}')

    response = requests.post(
        TTS_API_URL,
        json={'text': text, 'lang': language},
        timeout=30
    )
    response.raise_for_status()

    audio_data = response.content
    print(f'✅ TTS сгенерирован: {len(audio_data) / 1024:.2f} KB')
    return audio_data

def convert_to_wav(mp3_or_wav_data: bytes, output_path: str | None = None) -> Tuple[torch.Tensor, int]:
    """Декодировать входные аудиоданные в моно WAV 16kHz.

    - Пытается читать напрямую из BytesIO через soundfile.read (без записи на диск).
    - Фолбэк на torchaudio.load с явным указанием формата (например, MP3).
    - Если оба способа не сработают и указан output_path, выполняет ffmpeg-конвертацию
      с записью файла по output_path.

    Возвращает тензор waveform (1, num_samples) и sample_rate.
    Если передан output_path, сохраняет WAV на диск, иначе работает полностью в памяти.
    """
    print('🔄 Конвертация аудио → WAV (in-memory)...')

    target_sr = 16000
    waveform: torch.Tensor
    sample_rate: int

    # 1) Попытка через soundfile.read (WAV/FLAC/OGG и др.)
    try:
        audio_buffer = io.BytesIO(mp3_or_wav_data)
        data, sr = sf.read(audio_buffer, dtype='float32', always_2d=True)
        # data: (num_samples, num_channels)
        # Преобразуем к моно
        mono = data.mean(axis=1, keepdims=True)  # (num_samples, 1)
        waveform = torch.from_numpy(mono.T.copy())  # (1, num_samples)
        sample_rate = int(sr)
    except Exception as sf_error:
        print(f"⚠️ soundfile.read не смог декодировать аудио ({sf_error}); fallback на torchaudio.")
        # 2) Попытка через torchaudio из BytesIO
        try:
            audio_buffer = io.BytesIO(mp3_or_wav_data)
            # torchaudio умеет читать mp3 из буфера при указании format
            waveform, sample_rate = torchaudio.load(audio_buffer, format='mp3')
            waveform = waveform.float()
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
        except Exception as ta_error:
            print(f"⚠️ torchaudio.load не смог декодировать аудио ({ta_error}).")
            if output_path is None:
                # Без пути для ffmpeg некуда писать — завершаем ошибкой
                raise RuntimeError("Не удалось декодировать аудио из памяти без output_path для ffmpeg")
            # 3) Фолбэк на ffmpeg: пишем временный MP3 и конвертируем в WAV по output_path
            temp_mp3 = os.path.join(TEMP_DIR, f'temp_{int(time.time())}.mp3')
            with open(temp_mp3, 'wb') as f:
                f.write(mp3_or_wav_data)
            cmd = [
                'ffmpeg', '-y', '-i', temp_mp3,
                '-ar', str(target_sr),
                '-ac', '1',
                '-f', 'wav',
                '-acodec', 'pcm_s16le',
                output_path,
                '-loglevel', 'error'
            ]
            subprocess.run(cmd, check=True)
            os.remove(temp_mp3)
            waveform, sample_rate = torchaudio.load(output_path)
            waveform = waveform.float()
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

    # Ресемплинг к 16kHz при необходимости
    if sample_rate != target_sr:
        waveform = audio_fn.resample(waveform, sample_rate, target_sr)
        sample_rate = target_sr

    # Опциональное сохранение WAV
    if output_path is not None:
        torchaudio.save(output_path, waveform.cpu(), sample_rate)
        print(f'✅ WAV сохранен: {output_path}')
    else:
        print('✅ WAV подготовлен в памяти (без сохранения на диск)')

    return waveform, sample_rate
