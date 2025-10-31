"""Text-to-speech helpers and audio conversion utilities."""
from __future__ import annotations

import io
import os
import time
import subprocess
from typing import Tuple

import requests
import torchaudio
import torchaudio.functional as audio_fn

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


def convert_to_wav(mp3_data: bytes, output_path: str) -> Tuple[torchaudio.Tensor, int]:
    print('🔄 Конвертация в WAV...')

    audio_buffer = io.BytesIO(mp3_data)
    target_sr = 16000

    try:
        waveform, sample_rate = torchaudio.load(audio_buffer, format='mp3')
        waveform = waveform.float()
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sample_rate != target_sr:
            waveform = audio_fn.resample(waveform, sample_rate, target_sr)
            sample_rate = target_sr

        torchaudio.save(output_path, waveform.cpu(), sample_rate)
        print(f'✅ WAV сохранен: {output_path}')
        return waveform, sample_rate
    except Exception as decode_error:
        print(f'⚠️ torchaudio не смог декодировать MP3 ({decode_error}); fallback на ffmpeg.')

        temp_mp3 = os.path.join(TEMP_DIR, f'temp_{int(time.time())}.mp3')
        with open(temp_mp3, 'wb') as f:
            f.write(mp3_data)

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
        print(f'✅ WAV сохранен: {output_path}')
        return waveform, sample_rate
