#!/usr/bin/env python3
"""
Простой тест параллельной обработки
Сравнивает скорость обычной и параллельной обработки
"""

import requests
import time
import sys

API_URL = "http://localhost:3000"

# Тестовый текст (должен дать ~15-20 секунд аудио)
TEST_TEXT = """
Добрый день! Сегодня мы протестируем систему параллельной обработки видео.
Эта инновационная технология позволяет значительно ускорить генерацию видео
за счёт одновременной работы двух нейросетевых моделей - GAN и NOGAN.
Система автоматически разбивает аудио на оптимальное количество частей,
обрабатывает их параллельно и затем склеивает результат в единое видео.
"""


def check_health():
    """Проверка готовности сервера"""
    print("🔍 Проверка статуса сервера...")
    try:
        response = requests.get(f"{API_URL}/api/health", timeout=5)
        data = response.json()
        
        print(f"   Статус: {data.get('status')}")
        print(f"   GAN модель: {'✅' if data.get('gan_model_loaded') else '❌'}")
        print(f"   NOGAN модель: {'✅' if data.get('nogan_model_loaded') else '❌'}")
        print(f"   Устройство: {data.get('device')}")
        
        if not data.get('gan_model_loaded') or not data.get('nogan_model_loaded'):
            print("\n⚠️ Не все модели загружены! Параллельная обработка недоступна.")
            return False
        
        return True
    except Exception as e:
        print(f"❌ Ошибка подключения: {e}")
        print(f"   Убедитесь, что сервер запущен: python app.py")
        return False


def test_sequential():
    """Тест обычной обработки"""
    print("\n🎯 Тест обычной обработки...")
    
    start = time.time()
    try:
        response = requests.post(
            f"{API_URL}/api/generate",
            json={"text": TEST_TEXT, "language": "ru"},
            timeout=120
        )
        elapsed = time.time() - start
        
        if response.status_code == 200:
            with open("test_sequential.mp4", "wb") as f:
                f.write(response.content)
            print(f"   ✅ Завершено за {elapsed:.2f}s")
            print(f"   📁 Сохранено: test_sequential.mp4")
            return elapsed
        else:
            error = response.json()
            print(f"   ❌ Ошибка: {error.get('error')}")
            return None
    except Exception as e:
        print(f"   ❌ Ошибка: {e}")
        return None


def test_parallel():
    """Тест параллельной обработки"""
    print("\n🚀 Тест параллельной обработки...")
    
    start = time.time()
    try:
        response = requests.post(
            f"{API_URL}/api/generate_parallel",
            json={"text": TEST_TEXT, "language": "ru"},
            timeout=120
        )
        elapsed = time.time() - start
        
        if response.status_code == 200:
            with open("test_parallel.mp4", "wb") as f:
                f.write(response.content)
            print(f"   ✅ Завершено за {elapsed:.2f}s")
            print(f"   📁 Сохранено: test_parallel.mp4")
            return elapsed
        else:
            error = response.json()
            print(f"   ❌ Ошибка: {error.get('error')}")
            return None
    except Exception as e:
        print(f"   ❌ Ошибка: {e}")
        return None


def main():
    print("=" * 60)
    print("🧪 Тест параллельной обработки")
    print("=" * 60)
    
    # Проверка готовности
    if not check_health():
        sys.exit(1)
    
    # Тест обычной обработки
    sequential_time = test_sequential()
    if sequential_time is None:
        print("\n❌ Тест обычной обработки провалился")
        sys.exit(1)
    
    # Тест параллельной обработки
    parallel_time = test_parallel()
    if parallel_time is None:
        print("\n❌ Тест параллельной обработки провалился")
        sys.exit(1)
    
    # Результаты
    speedup = sequential_time / parallel_time
    percentage = ((sequential_time - parallel_time) / sequential_time) * 100
    
    print("\n" + "=" * 60)
    print("📊 РЕЗУЛЬТАТЫ")
    print("=" * 60)
    print(f"Обычная обработка:      {sequential_time:.2f}s")
    print(f"Параллельная обработка: {parallel_time:.2f}s")
    print(f"Ускорение:              {speedup:.2f}x")
    print(f"Выигрыш:                {percentage:.1f}%")
    print("=" * 60)
    
    if speedup > 1.3:
        print("\n✅ УСПЕХ! Параллельная обработка значительно быстрее! 🚀")
    elif speedup > 1.1:
        print("\n✅ Параллельная обработка работает и даёт ускорение.")
    else:
        print("\n⚠️ Ускорение минимальное. Попробуйте более длинный текст.")
    
    print(f"\n📁 Результаты сохранены:")
    print(f"   - test_sequential.mp4")
    print(f"   - test_parallel.mp4")


if __name__ == "__main__":
    main()
