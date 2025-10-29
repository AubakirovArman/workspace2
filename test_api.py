#!/usr/bin/env python3
"""
Тестовый скрипт для проверки API говорящего аватара
"""
import requests
import sys
import time

API_URL = "http://localhost:3000"

def check_health():
    """Проверка состояния сервиса"""
    print("🔍 Проверка состояния сервиса...")
    try:
        response = requests.get(f"{API_URL}/api/health", timeout=5)
        data = response.json()
        
        print(f"✅ Сервис доступен")
        print(f"   Статус: {data['status']}")
        print(f"   Модели загружены: {data['models_loaded']}")
        print(f"   Аватар загружен: {data['avatar_loaded']}")
        print(f"   Устройство: {data['device']}")
        return True
    except requests.exceptions.ConnectionError:
        print("❌ Сервис недоступен")
        print("   Запустите сервер: python app.py")
        return False
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return False

def generate_video(text, language='ru'):
    """Генерация видео"""
    print(f"\n🎬 Генерация видео...")
    print(f"   Текст: {text}")
    print(f"   Язык: {language}")
    
    try:
        start = time.time()
        
        response = requests.post(
            f"{API_URL}/api/generate",
            json={'text': text, 'language': language},
            timeout=120  # 2 минуты максимум
        )
        
        if response.status_code == 200:
            duration = time.time() - start
            
            # Сохраняем видео
            output_file = f"test_video_{int(time.time())}.mp4"
            with open(output_file, 'wb') as f:
                f.write(response.content)
            
            file_size = len(response.content) / 1024  # KB
            
            print(f"\n✅ Видео готово!")
            print(f"   Время: {duration:.2f}s")
            print(f"   Размер: {file_size:.2f} KB")
            print(f"   Файл: {output_file}")
            return True
        else:
            error = response.json() if response.headers.get('content-type') == 'application/json' else response.text
            print(f"❌ Ошибка: {error}")
            return False
            
    except requests.exceptions.Timeout:
        print("❌ Таймаут (>2 минут). Сервис перегружен или зависла обработка")
        return False
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return False

def main():
    print("="*60)
    print("🎭 Тест API Говорящего Аватара")
    print("="*60 + "\n")
    
    # Проверка сервиса
    if not check_health():
        sys.exit(1)
    
    # Тестовые фразы
    test_cases = [
        ("Привет! Это тестовое сообщение.", "ru"),
        ("Hello! This is a test message.", "en"),
    ]
    
    for text, lang in test_cases:
        print("\n" + "="*60)
        if not generate_video(text, lang):
            print("⚠️ Тест не пройден")
        time.sleep(2)  # Пауза между запросами
    
    print("\n" + "="*60)
    print("✅ Тестирование завершено!")
    print("="*60)

if __name__ == '__main__':
    main()
