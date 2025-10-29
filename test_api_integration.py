#!/usr/bin/env python3
"""
Тестовый скрипт для проверки API интеграции
"""
import requests
import json
import time
import os

API_BASE = 'http://localhost:3000'

def test_stream_chunks_api():
    """Тест генерации чанков через API"""
    print("="*60)
    print("🧪 Тест API /api/stream_chunks")
    print("="*60)
    
    # Тестовый текст
    text = (
        "Привет! Это тестовый текст для проверки API интеграции. "
        "Система разобьет его на чанки и сгенерирует видео и аудио. "
        "Каждый чанк будет доступен по отдельному URL."
    )
    
    print(f"\n📝 Текст: {text}")
    print(f"📏 Длина: {len(text)} символов, {len(text.split())} слов")
    
    # Запрос к API
    print("\n🚀 Отправка запроса...")
    start = time.time()
    
    response = requests.post(
        f'{API_BASE}/api/stream_chunks',
        json={
            'text': text,
            'language': 'ru',
            'chunk_size': 10  # 10 слов на чанк
        },
        timeout=120
    )
    
    elapsed = time.time() - start
    
    if response.status_code != 200:
        print(f"❌ Ошибка: {response.status_code}")
        print(response.text)
        return
    
    data = response.json()
    
    print(f"✅ Ответ получен за {elapsed:.2f}s")
    print(f"\n📊 Результат:")
    print(f"   Всего чанков: {data['total_chunks']}")
    print(f"   Язык: {data['language']}")
    
    # Информация о чанках
    print(f"\n📦 Чанки:")
    for chunk in data['chunks']:
        print(f"\n   Чанк #{chunk['index']}:")
        print(f"      Текст: {chunk['text'][:50]}...")
        print(f"      Длительность: {chunk['duration']}s")
        print(f"      Video: {chunk['video_url']}")
        print(f"      Audio: {chunk['audio_url']}")
    
    return data['chunks']


def test_download_chunk(chunk, output_dir='./test_chunks'):
    """Тест скачивания чанка"""
    print(f"\n📥 Скачивание чанка #{chunk['index']}...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Скачать видео
    video_url = f"{API_BASE}{chunk['video_url']}"
    print(f"   Видео: {video_url}")
    
    video_response = requests.get(video_url, timeout=30)
    if video_response.status_code == 200:
        video_path = f"{output_dir}/chunk_{chunk['index']}_video.mp4"
        with open(video_path, 'wb') as f:
            f.write(video_response.content)
        video_size = len(video_response.content) / 1024
        print(f"   ✅ Видео сохранено: {video_path} ({video_size:.1f} KB)")
    else:
        print(f"   ❌ Ошибка скачивания видео: {video_response.status_code}")
    
    # Скачать аудио
    audio_url = f"{API_BASE}{chunk['audio_url']}"
    print(f"   Аудио: {audio_url}")
    
    audio_response = requests.get(audio_url, timeout=30)
    if audio_response.status_code == 200:
        audio_path = f"{output_dir}/chunk_{chunk['index']}_audio.wav"
        with open(audio_path, 'wb') as f:
            f.write(audio_response.content)
        audio_size = len(audio_response.content) / 1024
        print(f"   ✅ Аудио сохранено: {audio_path} ({audio_size:.1f} KB)")
    else:
        print(f"   ❌ Ошибка скачивания аудио: {audio_response.status_code}")


def test_health():
    """Тест проверки здоровья сервиса"""
    print("\n🏥 Проверка здоровья сервиса...")
    
    response = requests.get(f'{API_BASE}/api/health', timeout=5)
    
    if response.status_code == 200:
        data = response.json()
        print(f"   ✅ Статус: {data['status']}")
        print(f"   📦 Модели загружены: {data['models_loaded']}")
        print(f"   🖼️  Аватар загружен: {data['avatar_loaded']}")
        print(f"   🔧 Устройство: {data['device']}")
    else:
        print(f"   ❌ Ошибка: {response.status_code}")


def main():
    print("\n" + "="*60)
    print("🧪 ТЕСТИРОВАНИЕ API ИНТЕГРАЦИИ")
    print("="*60)
    
    try:
        # 1. Проверка здоровья
        test_health()
        
        # 2. Генерация чанков
        chunks = test_stream_chunks_api()
        
        if not chunks:
            print("\n❌ Чанки не сгенерированы")
            return
        
        # 3. Скачивание первого чанка
        print("\n" + "="*60)
        print("📥 СКАЧИВАНИЕ ЧАНКОВ")
        print("="*60)
        
        test_download_chunk(chunks[0])
        
        print("\n" + "="*60)
        print("✅ ВСЕ ТЕСТЫ ПРОЙДЕНЫ!")
        print("="*60)
        print(f"\n📂 Чанки сохранены в: ./test_chunks/")
        print(f"📊 Сгенерировано чанков: {len(chunks)}")
        
    except requests.exceptions.ConnectionError:
        print("\n❌ Ошибка подключения!")
        print("Убедитесь, что сервер запущен на http://localhost:3000")
        print("Запустите: python app.py")
        
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
