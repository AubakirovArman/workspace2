#!/usr/bin/env python3
"""
Тест производительности оптимизаций GPU для Wav2Lip
Сравнивает разные конфигурации batch_size, FP16, torch.compile
"""
import sys
import os
import time
from pathlib import Path

# Добавляем modern-lipsync в путь
sys.path.insert(0, str(Path(__file__).parent / "modern-lipsync"))

import torch
import numpy as np
from service import LipsyncService


def get_gpu_memory():
    """Получить использование памяти GPU"""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024**3  # GB
    return 0


def benchmark_config(checkpoint_path: str, face_path: str, audio_path: str, 
                     batch_size: int, use_fp16: bool, use_compile: bool, 
                     num_runs: int = 3):
    """Бенчмарк для конкретной конфигурации"""
    
    config_name = f"Batch={batch_size}, FP16={use_fp16}, Compile={use_compile}"
    print(f"\n{'='*70}")
    print(f"🧪 Тестирование: {config_name}")
    print(f"{'='*70}")
    
    # Очистка памяти GPU
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    try:
        # Инициализация сервиса
        init_start = time.time()
        service = LipsyncService(
            checkpoint_path=checkpoint_path,
            device='cuda',
            face_det_batch_size=16,
            wav2lip_batch_size=batch_size,
            use_fp16=use_fp16,
            use_compile=use_compile
        )
        init_time = time.time() - init_start
        
        # Запуск бенчмарков
        times = []
        for run in range(num_runs):
            print(f"\n  Прогон {run+1}/{num_runs}...")
            
            # Очистка кэша перед каждым прогоном
            service._static_cache.clear()
            service._video_cache.clear()
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            start = time.time()
            
            # Обработка
            stats = service.process(
                face_path=face_path,
                audio_path=audio_path,
                output_path=f'/tmp/test_output_{batch_size}_{use_fp16}_{use_compile}_{run}.mp4',
                static=True,
                fps=25.0
            )
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            elapsed = time.time() - start
            times.append(elapsed)
            
            print(f"    Время обработки: {elapsed:.2f}s")
            print(f"    - Загрузка видео: {stats['load_video_time']:.2f}s")
            print(f"    - Обработка аудио: {stats['process_audio_time']:.2f}s")
            print(f"    - Детекция лиц: {stats['face_detection_time']:.2f}s")
            print(f"    - Инференс модели: {stats['inference_time']:.2f}s")
        
        # Статистика
        avg_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)
        peak_memory = get_gpu_memory()
        
        print(f"\n  📊 Результаты:")
        print(f"    Инициализация: {init_time:.2f}s")
        print(f"    Среднее время: {avg_time:.2f}s ± {std_time:.2f}s")
        print(f"    Мин/Макс: {min_time:.2f}s / {max_time:.2f}s")
        print(f"    Пиковая память GPU: {peak_memory:.2f} GB")
        
        # Очистка
        del service
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return {
            'config': config_name,
            'batch_size': batch_size,
            'use_fp16': use_fp16,
            'use_compile': use_compile,
            'init_time': init_time,
            'avg_time': avg_time,
            'std_time': std_time,
            'min_time': min_time,
            'max_time': max_time,
            'peak_memory': peak_memory
        }
        
    except Exception as e:
        print(f"\n  ❌ Ошибка: {e}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return None


def main():
    """Главная функция тестирования"""
    
    # Проверка аргументов
    if len(sys.argv) < 4:
        print("Использование: python test_gpu_optimization.py <checkpoint> <face> <audio>")
        print("\nПример:")
        print("  python test_gpu_optimization.py Wav2Lip-SD-GAN.pt avatar.jpg audio_40s.wav")
        sys.exit(1)
    
    checkpoint_path = sys.argv[1]
    face_path = sys.argv[2]
    audio_path = sys.argv[3]
    
    # Проверка файлов
    if not os.path.exists(checkpoint_path):
        print(f"❌ Чекпоинт не найден: {checkpoint_path}")
        sys.exit(1)
    if not os.path.exists(face_path):
        print(f"❌ Аватар не найден: {face_path}")
        sys.exit(1)
    if not os.path.exists(audio_path):
        print(f"❌ Аудио не найден: {audio_path}")
        sys.exit(1)
    
    print("\n" + "="*70)
    print("🚀 ТЕСТ ОПТИМИЗАЦИЙ GPU ДЛЯ WAV2LIP")
    print("="*70)
    print(f"Чекпоинт: {checkpoint_path}")
    print(f"Аватар: {face_path}")
    print(f"Аудио: {audio_path}")
    
    if torch.cuda.is_available():
        print(f"\n🎮 GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Память: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("\n⚠️ CUDA не доступна! Тесты будут медленными.")
    
    # Конфигурации для тестирования
    configs = [
        # Базовая конфигурация (старая)
        {'batch_size': 128, 'use_fp16': False, 'use_compile': False},
        
        # Только увеличенный batch
        {'batch_size': 512, 'use_fp16': False, 'use_compile': False},
        
        # Batch + FP16
        {'batch_size': 512, 'use_fp16': True, 'use_compile': False},
        
        # Batch + FP16 + Compile (полная оптимизация)
        {'batch_size': 512, 'use_fp16': True, 'use_compile': True},
        
        # Экстремальный batch (если есть память)
        {'batch_size': 1024, 'use_fp16': True, 'use_compile': True},
    ]
    
    # Запуск тестов
    results = []
    for config in configs:
        result = benchmark_config(
            checkpoint_path=checkpoint_path,
            face_path=face_path,
            audio_path=audio_path,
            num_runs=2,  # Меньше прогонов для экономии времени
            **config
        )
        if result:
            results.append(result)
        
        # Пауза между тестами
        print("\n  ⏳ Пауза 5 секунд перед следующим тестом...")
        time.sleep(5)
    
    # Итоговая таблица
    print("\n" + "="*70)
    print("📈 ИТОГОВАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
    print("="*70)
    
    if not results:
        print("❌ Нет успешных результатов")
        return
    
    # Заголовок таблицы
    print(f"\n{'Конфигурация':<50} {'Время (s)':<15} {'Ускорение':<12} {'Память (GB)':<12}")
    print("-" * 90)
    
    # Базовое время для сравнения (первый результат)
    baseline_time = results[0]['avg_time']
    
    for result in results:
        config_str = f"B={result['batch_size']}, FP16={result['use_fp16']}, Comp={result['use_compile']}"
        time_str = f"{result['avg_time']:.2f} ± {result['std_time']:.2f}"
        speedup = baseline_time / result['avg_time']
        speedup_str = f"{speedup:.2f}x"
        memory_str = f"{result['peak_memory']:.2f}"
        
        print(f"{config_str:<50} {time_str:<15} {speedup_str:<12} {memory_str:<12}")
    
    # Лучший результат
    best_result = min(results, key=lambda x: x['avg_time'])
    print("\n" + "="*70)
    print(f"🏆 ЛУЧШАЯ КОНФИГУРАЦИЯ:")
    print(f"   Batch Size: {best_result['batch_size']}")
    print(f"   FP16: {best_result['use_fp16']}")
    print(f"   Compile: {best_result['use_compile']}")
    print(f"   Время: {best_result['avg_time']:.2f}s")
    print(f"   Ускорение: {baseline_time / best_result['avg_time']:.2f}x")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
