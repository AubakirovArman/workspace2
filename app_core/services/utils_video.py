"""Утилиты для потоковой обработки видео через ffmpeg (decode -> Python -> encode).

Минимальный набор: ffprobe() и process_video_streaming().
Этот файл лежит в пакете app_core.services чтобы его удобно импортировать из blueprints.
"""
from __future__ import annotations

import json
import shlex
import subprocess
import tempfile
from typing import Callable

import numpy as np


def ffprobe(video_path: str):
    """Возвращает (width, height, fps) для видео через ffprobe."""
    cmd = f"ffprobe -v error -print_format json -show_streams {shlex.quote(video_path)}"
    out = subprocess.check_output(cmd, shell=True).decode("utf-8")
    info = json.loads(out)
    vstream = next((s for s in info.get('streams', []) if s.get('codec_type') == 'video'), None)
    if vstream is None:
        raise RuntimeError('Видео поток не найден')

    w, h = int(vstream.get('width', 0)), int(vstream.get('height', 0))
    r = vstream.get('r_frame_rate', '30/1')
    try:
        num, den = map(int, r.split('/'))
        fps = num / den if den else 30.0
    except Exception:
        fps = 30.0
    return w, h, fps


def process_video_streaming(
    in_path: str, 
    out_path: str, 
    process_frame_fn: Callable, 
    use_nvdec: bool = False,
    encoder: str = 'libx264',
    crf: int = 20,
    preset: str = 'veryfast'
):
    """Потоковая обработка видео через два ffmpeg процесса.

    Оптимизировано под H200: NVDEC (опционально) для декодирования, libx264 для кодирования.

    Args:
        in_path: путь к входному видео
        out_path: путь к выходному видео
        process_frame_fn: функция обработки кадра (RGB uint8 -> RGB uint8)
        use_nvdec: использовать NVDEC для декодирования (H200 поддерживает до 7 NVDEC)
        encoder: 'libx264' (CPU, рекомендуется для H200) или 'h264_nvenc' (если доступен)
        crf: качество для libx264 (18-23 оптимально, меньше=лучше)
        preset: пресет libx264 ('ultrafast', 'veryfast', 'fast', 'medium')
    
    process_frame_fn(frame: np.ndarray) -> np.ndarray ожидает RGB uint8 кадр (H,W,3) и возвращает тот же формат.
    """
    w, h, fps = ffprobe(in_path)

    # Декодер: опционально NVDEC (H200 имеет 7x NVDEC, но не NVENC)
    dec_cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error']
    if use_nvdec:
        # NVDEC: аппаратное декодирование на GPU
        dec_cmd.extend(['-hwaccel', 'cuda', '-hwaccel_output_format', 'cuda'])
    
    dec_cmd.extend([
        '-i', in_path,
        '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-vsync', '0', '-'
    ])
    
    dec = subprocess.Popen(dec_cmd, stdout=subprocess.PIPE)

    # Энкодер: libx264 (CPU) с оптимальными параметрами для CFR и склейки
    keyint = int(2 * fps)  # GOP = 2 секунды
    
    if encoder == 'libx264':
        enc_cmd = [
            'ffmpeg', '-hide_banner', '-loglevel', 'error',
            '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{w}x{h}', '-r', str(fps), '-i', '-',
            '-i', in_path,
            '-map', '0:v:0', '-map', '1:a?',
            '-c:v', 'libx264',
            '-preset', preset,
            '-crf', str(crf),
            '-x264-params', f'keyint={keyint}:min-keyint={keyint}:scenecut=0:force-cfr=1',
            '-pix_fmt', 'yuv420p',
            '-c:a', 'aac', '-b:a', '192k', '-ar', '48000',
            '-movflags', '+faststart',
            '-shortest',
            out_path
        ]
    else:  # h264_nvenc (fallback, если вдруг доступен)
        enc_cmd = [
            'ffmpeg', '-hide_banner', '-loglevel', 'error',
            '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{w}x{h}', '-r', str(fps), '-i', '-',
            '-i', in_path,
            '-map', '0:v:0', '-map', '1:a?',
            '-c:v', 'h264_nvenc',
            '-preset', 'p5',
            '-rc', 'vbr', '-cq', str(crf),
            '-pix_fmt', 'yuv420p',
            '-c:a', 'aac', '-b:a', '192k', '-ar', '48000',
            '-movflags', '+faststart',
            '-shortest',
            out_path
        ]

    enc = subprocess.Popen(enc_cmd, stdin=subprocess.PIPE)

    frame_size = w * h * 3
    try:
        while True:
            raw = dec.stdout.read(frame_size)
            if not raw:
                break
            frame = np.frombuffer(raw, dtype=np.uint8)
            if frame.size != frame_size:
                # неполный кадр в конце
                break
            frame = frame.reshape((h, w, 3))  # RGB
            out_frame = process_frame_fn(frame)
            # Ожидаем выходной кадр того же размера и типа
            if out_frame.shape != (h, w, 3):
                raise RuntimeError('process_frame_fn вернул кадр некорректного размера')
            enc.stdin.write(out_frame.astype(np.uint8).tobytes())
    finally:
        dec.wait()
        try:
            enc.stdin.close()
        except Exception:
            pass
        enc.wait()
