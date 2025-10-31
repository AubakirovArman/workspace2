"""
Parallel Lipsync Processing
Ускорение инференса через параллельную обработку на двух моделях
"""
from __future__ import annotations

import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from pydub import AudioSegment


def split_audio_file(audio_path: str, num_chunks: int = 2) -> List[Tuple[str, float, float]]:
    """
    Разбить аудио на N чанков для параллельной обработки
    
    Args:
        audio_path: Путь к аудио файлу
        num_chunks: Количество частей (по умолчанию 2 для двух моделей)
        
    Returns:
        List[(chunk_path, start_time, end_time)]
    """
    audio = AudioSegment.from_file(audio_path)
    duration_ms = len(audio)
    chunk_duration = duration_ms / num_chunks
    
    chunks = []
    temp_dir = tempfile.mkdtemp(prefix="audio_chunks_")
    
    for i in range(num_chunks):
        start_ms = int(i * chunk_duration)
        end_ms = int((i + 1) * chunk_duration) if i < num_chunks - 1 else duration_ms
        
        chunk = audio[start_ms:end_ms]
        chunk_path = os.path.join(temp_dir, f"chunk_{i:03d}.wav")
        chunk.export(chunk_path, format="wav")
        
        chunks.append((chunk_path, start_ms / 1000.0, end_ms / 1000.0))
    
    return chunks


def process_chunk_with_service(
    service,
    audio_chunk_path: str,
    chunk_index: int,
    use_cached: bool = True
) -> Tuple[int, str]:
    """
    Обработать один чанк аудио одной моделью
    
    Args:
        service: LipsyncService (GAN или NOGAN)
        audio_chunk_path: Путь к аудио чанку
        chunk_index: Индекс чанка (для сортировки)
        use_cached: Использовать предзагруженное лицо
        
    Returns:
        (chunk_index, output_video_path)
    """
    temp_output = tempfile.mktemp(suffix=f"_chunk_{chunk_index:03d}.mp4")
    
    if use_cached:
        service.process_with_preloaded(
            audio_path=audio_chunk_path,
            output_path=temp_output
        )
    else:
        # Для динамического режима нужен face_path
        raise NotImplementedError("Динамический режим в параллельной обработке пока не поддерживается")
    
    return chunk_index, temp_output


def merge_video_chunks(chunk_paths: List[str], output_path: str, fps: int = 25) -> None:
    """
    Склеить видео чанки в один файл
    
    Args:
        chunk_paths: Список путей к видео чанкам (в правильном порядке!)
        output_path: Путь для финального видео
        fps: FPS финального видео
    """
    # Создаём временный файл со списком для ffmpeg concat
    concat_file = tempfile.mktemp(suffix=".txt")
    
    with open(concat_file, 'w') as f:
        for chunk_path in chunk_paths:
            # Формат для ffmpeg concat demuxer
            f.write(f"file '{chunk_path}'\n")
    
    # Используем ffmpeg для склейки
    import subprocess
    cmd = [
        'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
        '-i', concat_file,
        '-c', 'copy',  # Копируем без перекодирования (быстро!)
        output_path
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    os.unlink(concat_file)
    
    if result.returncode != 0:
        raise RuntimeError(f"Ошибка склейки видео: {result.stderr}")


def parallel_lipsync_process(
    gan_service,
    nogan_service,
    audio_path: str,
    output_path: str,
    num_workers: int = 2,
    fps: int = 25,
    use_cached: bool = True,
    gan2_service=None,
    gan3_service=None,
    use_only_gan: bool = False
) -> dict:
    """
    Параллельная обработка аудио на нескольких моделях
    
    Args:
        gan_service: GAN LipsyncService #1
        nogan_service: NOGAN LipsyncService
        audio_path: Путь к полному аудио
        output_path: Путь для финального видео
        num_workers: Количество параллельных воркеров (2-3)
        fps: FPS финального видео
        use_cached: Использовать предзагруженное лицо
        gan2_service: Второй экземпляр GAN (для 3 моделей)
        gan3_service: Третий экземпляр GAN (для 3 моделей)
        use_only_gan: Использовать только GAN модели (игнорировать NOGAN)
        
    Returns:
        dict с информацией о времени обработки
    """
    start_time = time.time()
    
    # 1. Разбить аудио на чанки
    print(f"📦 Разбиваем аудио на {num_workers} частей...")
    split_start = time.time()
    audio_chunks = split_audio_file(audio_path, num_chunks=num_workers)
    split_time = time.time() - split_start
    print(f"✅ Аудио разбито за {split_time:.2f}s")
    
    # 2. Подготовка списка сервисов
    if use_only_gan:
        # Используем только GAN модели
        available_services = [gan_service]
        if gan2_service:
            available_services.append(gan2_service)
        if gan3_service:
            available_services.append(gan3_service)
        services = available_services
    else:
        # Используем GAN + NOGAN (старое поведение)
        services = [gan_service, nogan_service] if num_workers == 2 else [gan_service] * num_workers
    
    # 3. Параллельная обработка чанков
    print(f"🚀 Запуск параллельной обработки на {len(services)} моделях...")
    process_start = time.time()
    
    chunk_results = {}
    
    with ThreadPoolExecutor(max_workers=len(services)) as executor:
        futures = []
        
        for i, (chunk_path, start_t, end_t) in enumerate(audio_chunks):
            service = services[i % len(services)]
            
            # Определяем имя модели
            if service == gan_service:
                service_name = "GAN-1"
            elif gan2_service and service == gan2_service:
                service_name = "GAN-2"
            elif gan3_service and service == gan3_service:
                service_name = "GAN-3"
            elif service == nogan_service:
                service_name = "NOGAN"
            else:
                service_name = "GAN"
            
            print(f"   - Чанк {i}: {start_t:.2f}s-{end_t:.2f}s → {service_name}")
            
            future = executor.submit(
                process_chunk_with_service,
                service,
                chunk_path,
                i,
                use_cached
            )
            futures.append(future)
        
        # Ждём завершения всех задач
        for future in as_completed(futures):
            chunk_idx, video_path = future.result()
            chunk_results[chunk_idx] = video_path
            print(f"   ✅ Чанк {chunk_idx} готов")
    
    process_time = time.time() - process_start
    print(f"✅ Все чанки обработаны за {process_time:.2f}s")
    
    # 3. Склеить видео чанки
    print("🎬 Склеиваем видео чанки...")
    merge_start = time.time()
    
    # Сортируем чанки по индексу
    sorted_chunks = [chunk_results[i] for i in sorted(chunk_results.keys())]
    merge_video_chunks(sorted_chunks, output_path, fps)
    
    merge_time = time.time() - merge_start
    print(f"✅ Видео склеено за {merge_time:.2f}s")
    
    # 4. Очистка временных файлов
    print("🧹 Очистка временных файлов...")
    for chunk_path, _, _ in audio_chunks:
        try:
            os.unlink(chunk_path)
            # Удаляем также директорию чанков
            chunk_dir = os.path.dirname(chunk_path)
            if os.path.exists(chunk_dir):
                os.rmdir(chunk_dir)
        except Exception as e:
            print(f"⚠️ Не удалось удалить {chunk_path}: {e}")
    
    for video_path in chunk_results.values():
        try:
            os.unlink(video_path)
        except Exception as e:
            print(f"⚠️ Не удалось удалить {video_path}: {e}")
    
    total_time = time.time() - start_time
    
    return {
        "total_time": total_time,
        "split_time": split_time,
        "process_time": process_time,
        "merge_time": merge_time,
        "num_chunks": len(audio_chunks),
        "speedup": "~1.5-2x vs sequential"
    }


def estimate_optimal_chunks(audio_duration_seconds: float, num_models: int = 2) -> int:
    """
    Оценить оптимальное количество чанков для параллельной обработки
    
    Args:
        audio_duration_seconds: Длительность аудио в секундах
        num_models: Количество доступных моделей
        
    Returns:
        Оптимальное количество чанков
    """
    # Для коротких аудио (< 10s) не имеет смысла разбивать
    if audio_duration_seconds < 10:
        return 1
    
    # Для средних (10-30s) - 2 чанка
    if audio_duration_seconds < 30:
        return min(2, num_models)
    
    # Для длинных - можно больше, но не более num_models * 2
    # (иначе overhead от склейки превысит выигрыш)
    optimal = min(
        int(audio_duration_seconds / 15),  # Каждый чанк ~15s
        num_models * 2
    )
    
    return max(2, optimal)
