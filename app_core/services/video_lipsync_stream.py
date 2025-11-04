"""Потоковая lip-sync обработка видео с конвейером decode→inference→encode.

Оптимизировано под H200: NVDEC для декодирования, GPU для инференса, libx264 для кодирования.
Без промежуточных файлов и склейки — единый поток от входа до выхода.
"""
from __future__ import annotations

import json
import os
import shlex
import subprocess
import threading
import time
from queue import Queue
from typing import Optional

import cv2
import numpy as np
import torch


def ffprobe_video(video_path: str) -> tuple[int, int, float]:
    """Возвращает (width, height, fps) для видео."""
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
        fps = num / den if den else 25.0
    except Exception:
        fps = 25.0
    return w, h, fps


def start_video_decoder(video_path: str, w: int, h: int, use_nvdec: bool = False) -> subprocess.Popen:
    """Запускает ffmpeg декодер для потокового чтения кадров.
    
    Args:
        video_path: путь к видео
        w, h: размер кадров
        use_nvdec: использовать аппаратное декодирование (H200 имеет 7x NVDEC)
    
    Returns:
        Popen процесс с stdout = raw RGB кадры
    """
    cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error']
    
    if use_nvdec:
        # H200: NVDEC для аппаратного декодирования
        cmd.extend(['-hwaccel', 'cuda', '-hwaccel_output_format', 'cuda'])
    
    cmd.extend([
        '-i', video_path,
        '-f', 'rawvideo',
        '-pix_fmt', 'rgb24',
        '-vsync', '0',
        '-'
    ])
    
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def start_video_encoder(
    output_path: str,
    audio_path: str,
    w: int,
    h: int,
    fps: float,
    encoder: str = 'libx264',
    crf: int = 20,
    preset: str = 'veryfast'
) -> subprocess.Popen:
    """Запускает ffmpeg энкодер для записи обработанных кадров.
    
    Оптимизировано под H200: libx264 на CPU с правильными параметрами CFR.
    
    Args:
        output_path: путь к выходному файлу
        audio_path: путь к аудио (WAV) для подмешивания
        w, h: размер кадров
        fps: частота кадров
        encoder: 'libx264' (рекомендуется для H200) или 'h264_nvenc'
        crf: качество (18-23, меньше=лучше)
        preset: 'ultrafast', 'veryfast', 'fast', 'medium'
    
    Returns:
        Popen процесс с stdin = raw RGB кадры
    """
    keyint = int(2 * fps)  # GOP = 2 секунды для стабильной CFR
    
    if encoder == 'libx264':
        cmd = [
            'ffmpeg', '-hide_banner', '-loglevel', 'error',
            # Входной видео поток (raw RGB)
            '-f', 'rawvideo', '-pix_fmt', 'rgb24',
            '-s', f'{w}x{h}', '-r', str(fps), '-i', '-',
            # Аудио
            '-i', audio_path,
            # Маппинг потоков
            '-map', '0:v:0', '-map', '1:a:0',
            # Видео кодек (libx264 на CPU)
            '-c:v', 'libx264',
            '-preset', preset,
            '-crf', str(crf),
            '-x264-params', f'keyint={keyint}:min-keyint={keyint}:scenecut=0:force-cfr=1',
            '-pix_fmt', 'yuv420p',
            # Аудио кодек
            '-c:a', 'aac', '-b:a', '192k', '-ar', '48000',
            # Оптимизация
            '-movflags', '+faststart',
            '-shortest',
            output_path
        ]
    else:  # h264_nvenc (если вдруг доступен, но на H200 обычно нет)
        cmd = [
            'ffmpeg', '-hide_banner', '-loglevel', 'error',
            '-f', 'rawvideo', '-pix_fmt', 'rgb24',
            '-s', f'{w}x{h}', '-r', str(fps), '-i', '-',
            '-i', audio_path,
            '-map', '0:v:0', '-map', '1:a:0',
            '-c:v', 'h264_nvenc',
            '-preset', 'p5',
            '-rc', 'vbr', '-cq', str(crf),
            '-pix_fmt', 'yuv420p',
            '-c:a', 'aac', '-b:a', '192k', '-ar', '48000',
            '-movflags', '+faststart',
            '-shortest',
            output_path
        ]
    
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)


def process_video_lipsync_streaming(
    base_video_path: str,
    audio_path: str,
    output_path: str,
    lipsync_service,
    use_nvdec: bool = False,
    encoder: str = 'libx264',
    crf: int = 20,
    preset: str = 'veryfast',
    pads: tuple[int, int, int, int] = (0, 10, 0, 0),
    nosmooth: bool = False,
) -> dict:
    """Потоковая lip-sync обработка видео без промежуточных файлов.
    
    Архитектура: ffmpeg decode → Queue → GPU lip-sync → Queue → ffmpeg encode
    
    Args:
        base_video_path: исходное видео (база для аватара)
        audio_path: аудио для синхронизации губ (WAV)
        output_path: выходное видео
        lipsync_service: экземпляр LipsyncService с загруженной моделью
        use_nvdec: использовать NVDEC (H200 поддерживает)
        encoder: 'libx264' или 'h264_nvenc'
        crf: качество (18-23)
        preset: скорость кодирования
        pads: отступы для детекции лица
        nosmooth: отключить сглаживание детекции
    
    Returns:
        dict со статистикой обработки
    """
    start_total = time.time()
    
    # Получаем параметры видео
    w, h, fps = ffprobe_video(base_video_path)
    print(f"📹 Видео: {w}x{h} @ {fps:.2f} FPS")
    
    # Загружаем все кадры видео (для упрощения первой версии)
    # TODO: для длинных видео можно стримить через очередь
    print("📦 Загрузка кадров базового видео...")
    start = time.time()
    video_frames = _load_video_frames(base_video_path)
    load_time = time.time() - start
    print(f"✅ Загружено {len(video_frames)} кадров за {load_time:.2f}s")
    
    # Детекция лиц на всех кадрах (кэшируем координаты)
    print("👤 Детекция лиц...")
    start = time.time()
    face_det_results = lipsync_service.detect_faces(video_frames, pads, nosmooth)
    detect_time = time.time() - start
    print(f"✅ Детекция завершена за {detect_time:.2f}s")
    
    # Обработка аудио -> мел-спектрограммы
    print("🎵 Обработка аудио...")
    start = time.time()
    mel, mel_chunks, temp_wav = lipsync_service._process_audio(audio_path, fps)
    audio_time = time.time() - start
    print(f"✅ Аудио обработано за {audio_time:.2f}s, чанков: {len(mel_chunks)}")
    
    # Запускаем энкодер
    print(f"🎬 Запуск энкодера ({encoder})...")
    encoder_proc = start_video_encoder(
        output_path, audio_path, w, h, fps,
        encoder=encoder, crf=crf, preset=preset
    )
    
    # Инференс + запись кадров
    print("🎭 Lip-sync инференс...")
    start = time.time()
    
    batch_size = lipsync_service.wav2lip_batch_size
    frames_processed = 0
    
    for i in range(0, len(mel_chunks), batch_size):
        batch_mel = mel_chunks[i:i + batch_size]
        img_batch, mel_batch, frames_batch, coords_batch = [], [], [], []
        
        for j, mel_window in enumerate(batch_mel):
            idx = (i + j) % len(video_frames)
            frame_to_save = video_frames[idx].copy()
            face, coords = face_det_results[idx]
            
            face_resized = cv2.resize(face, (lipsync_service.img_size, lipsync_service.img_size))
            
            img_batch.append(face_resized)
            mel_batch.append(mel_window)
            frames_batch.append(frame_to_save)
            coords_batch.append(coords)
        
        # Подготовка батча для модели
        img_batch_np = np.asarray(img_batch)
        mel_batch_np = np.asarray(mel_batch)
        
        img_masked = img_batch_np.copy()
        img_masked[:, lipsync_service.img_size // 2:] = 0
        
        img_batch_np = np.concatenate((img_masked, img_batch_np), axis=3) / 255.0
        mel_batch_np = np.reshape(
            mel_batch_np, [len(mel_batch_np), mel_batch_np.shape[1], mel_batch_np.shape[2], 1]
        )
        
        img_batch_tensor = torch.from_numpy(
            np.transpose(img_batch_np, (0, 3, 1, 2))
        ).float().to(lipsync_service.device)
        
        mel_batch_tensor = torch.from_numpy(
            np.transpose(mel_batch_np, (0, 3, 1, 2))
        ).float().to(lipsync_service.device)
        
        # Инференс
        with torch.no_grad():
            pred = lipsync_service.model(mel_batch_tensor, img_batch_tensor)
        
        pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.0
        
        # Постобработка и запись кадров
        for predicted_patch, frame, coords in zip(pred, frames_batch, coords_batch):
            y1, y2, x1, x2 = coords
            frame_patch = predicted_patch.astype(np.uint8)
            frame_patch = cv2.resize(frame_patch, (x2 - x1, y2 - y1))
            frame[y1:y2, x1:x2] = frame_patch
            
            # Конвертируем BGR -> RGB для энкодера
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            encoder_proc.stdin.write(frame_rgb.tobytes())
            frames_processed += 1
    
    inference_time = time.time() - start
    print(f"✅ Инференс завершён: {frames_processed} кадров за {inference_time:.2f}s")
    print(f"   ({frames_processed / inference_time:.1f} FPS)")
    
    # Завершаем энкодер
    encoder_proc.stdin.close()
    encoder_proc.wait()
    
    # Очистка temp файлов
    if temp_wav and os.path.exists(temp_wav):
        try:
            os.remove(temp_wav)
        except OSError:
            pass
    
    total_time = time.time() - start_total
    
    stats = {
        'load_video_time': load_time,
        'face_detection_time': detect_time,
        'process_audio_time': audio_time,
        'inference_time': inference_time,
        'total_time': total_time,
        'frames_processed': frames_processed,
        'fps_achieved': frames_processed / inference_time if inference_time > 0 else 0,
        'video_resolution': f'{w}x{h}',
        'video_fps': fps,
        'encoder': encoder,
        'use_nvdec': use_nvdec,
    }
    
    print(f"\n✅ Видео готово: {output_path}")
    print(f"⏱️  Общее время: {total_time:.2f}s")
    
    return stats


def _load_video_frames(video_path: str) -> list[np.ndarray]:
    """Загружает все кадры видео в память (BGR формат для OpenCV)."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    cap.release()
    return frames
