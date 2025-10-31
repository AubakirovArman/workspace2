"""
Avatar Lipsync Web Application
Веб-сервис для создания говорящего аватара через TTS и Wav2Lip
Держит все модели в памяти для быстрой обработки
"""
from __future__ import annotations

import os
import sys

# Ensure modern-lipsync modules are importable before other imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'modern-lipsync'))

from app_core import create_app
from app_core.config import DEBUG, HOST, PORT
from app_core.services import init_lipsync_service

app = create_app()


def main() -> None:
    try:
        init_lipsync_service()
    except Exception as exc:  # pragma: no cover - startup diagnostics
        print(f"\n❌ Ошибка инициализации: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print(f"🌐 Запуск веб-сервера на http://{HOST}:{PORT}")
    print("📝 Откройте браузер и перейдите по адресу")
    print(f"   http://{HOST}:{PORT}\n")

    app.run(
        host=HOST,
        port=PORT,
        debug=DEBUG,
        threaded=True
    )


if __name__ == '__main__':
    main()
