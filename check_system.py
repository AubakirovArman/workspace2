#!/usr/bin/env python3
"""
Проверка готовности системы к запуску
"""
import os
import sys

def check_file(path, name):
    """Проверка наличия файла"""
    if os.path.exists(path):
        size = os.path.getsize(path)
        size_str = f"{size/1024:.1f}KB" if size < 1024*1024 else f"{size/(1024*1024):.1f}MB"
        print(f"✅ {name}: {size_str}")
        return True
    else:
        print(f"❌ {name}: НЕ НАЙДЕН")
        return False

def check_command(cmd, name):
    """Проверка наличия команды"""
    ret = os.system(f"which {cmd} > /dev/null 2>&1")
    if ret == 0:
        print(f"✅ {name}: установлен")
        return True
    else:
        print(f"❌ {name}: НЕ УСТАНОВЛЕН")
        return False

def check_python_module(module, name):
    """Проверка наличия Python модуля"""
    try:
        __import__(module)
        print(f"✅ {name}: установлен")
        return True
    except ImportError:
        print(f"❌ {name}: НЕ УСТАНОВЛЕН")
        return False

def main():
    print("="*60)
    print("🔍 Проверка готовности системы")
    print("="*60)
    
    all_ok = True
    
    print("\n📁 Обязательные файлы:")
    all_ok &= check_file("/workspace/avatar.jpg", "Аватар")
    all_ok &= check_file("/workspace/app.py", "Flask сервер")
    all_ok &= check_file("/workspace/templates/index.html", "Веб-интерфейс")
    
    print("\n🧠 Модели:")
    has_gan = check_file("/workspace/Wav2Lip-SD-GAN.pt", "Wav2Lip-SD-GAN")
    has_nogan = check_file("/workspace/Wav2Lip-SD-NOGAN.pt", "Wav2Lip-SD-NOGAN")
    if not (has_gan or has_nogan):
        print("   ⚠️  Нужна хотя бы одна модель!")
        all_ok = False
    
    print("\n🛠️  Системные утилиты:")
    all_ok &= check_command("python3", "Python 3")
    all_ok &= check_command("ffmpeg", "ffmpeg")
    
    print("\n📦 Python модули:")
    modules = [
        ("flask", "Flask"),
        ("torch", "PyTorch"),
        ("cv2", "OpenCV"),
        ("numpy", "NumPy"),
        ("requests", "Requests"),
    ]
    
    for module, name in modules:
        all_ok &= check_python_module(module, name)
    
    print("\n" + "="*60)
    
    if all_ok:
        print("✅ ВСЕ ГОТОВО К ЗАПУСКУ!")
        print("\nЗапустите сервер:")
        print("  python app.py")
        print("\nИли используйте скрипт:")
        print("  ./start_web.sh")
        print("\nЗатем откройте:")
        print("  http://localhost:3000")
    else:
        print("❌ СИСТЕМА НЕ ГОТОВА")
        print("\nУстановите недостающие компоненты:")
        print("  pip install -r requirements_web.txt")
        print("  sudo apt-get install ffmpeg  # Ubuntu/Debian")
        print("  brew install ffmpeg          # macOS")
    
    print("="*60)
    
    return 0 if all_ok else 1

if __name__ == '__main__':
    sys.exit(main())
