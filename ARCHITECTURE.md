# 🎭 Веб-приложение "Говорящий Аватар" - Архитектура

## 📋 Обзор

Веб-сервис для создания видео с говорящим аватаром на основе текста. Система объединяет три технологии:
1. **TTS (Text-to-Speech)** - преобразование текста в речь
2. **Wav2Lip** - синхронизация губ с аудио
3. **Flask Web Server** - веб-интерфейс и API

## 🏗️ Архитектура

```
┌─────────────────────────────────────────────────────────────┐
│                     Веб-браузер (Клиент)                     │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  HTML/CSS/JavaScript интерфейс (index.html)          │   │
│  │  - Ввод текста                                       │   │
│  │  - Выбор языка                                       │   │
│  │  - Отображение аватара                               │   │
│  │  - Видео плеер                                       │   │
│  └──────────────────────────────────────────────────────┘   │
└───────────────────────────┬─────────────────────────────────┘
                            │ HTTP POST /api/generate
                            │ {text, language}
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              Flask Web Server (app.py) :3000                │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  API Endpoints:                                      │   │
│  │  • GET  /           → index.html                     │   │
│  │  • POST /api/generate → Генерация видео              │   │
│  │  • GET  /api/health  → Статус сервиса               │   │
│  │  • GET  /api/avatar  → Изображение аватара           │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  Предзагруженные ресурсы в памяти:                          │
│  • avatar.jpg (изображение аватара)                         │
│  • LipsyncService (модели Wav2Lip + Face Detection)         │
└───────────────────────────┬─────────────────────────────────┘
                            │
        ┌───────────────────┴────────────────────┐
        │                                        │
        ↓                                        ↓
┌─────────────────────┐              ┌─────────────────────────┐
│  TTS API (внешний)  │              │  LipsyncService         │
│  tts.sk-ai.kz       │              │  (modern-lipsync/)      │
├─────────────────────┤              ├─────────────────────────┤
│ POST /api/tts       │              │ • Wav2Lip Model         │
│ {text, lang}        │              │ • Face Detection        │
│                     │              │ • Audio Processor       │
│ ↓ возвращает MP3    │              │                         │
└─────────────────────┘              │ Процесс:                │
                                     │ 1. Detect face          │
                                     │ 2. Extract mel-spec     │
                                     │ 3. Sync lips            │
                                     │ 4. Generate video       │
                                     └─────────────────────────┘
```

## 🔄 Поток обработки запроса

### 1. Инициализация (при старте сервера)
```python
# app.py - init_service()
1. Загрузка Wav2Lip модели в память (5-10s)
2. Загрузка Face Detector в память (1-2s)
3. Предзагрузка avatar.jpg в память
4. Инициализация Audio Processor

→ Модели остаются в памяти для всех запросов!
```

### 2. Обработка запроса
```python
# POST /api/generate {text: "Привет", language: "ru"}

Шаг 1: TTS генерация (2-5s)
├─ Запрос к tts.sk-ai.kz API
├─ Получение MP3 аудио
└─ Конвертация MP3 → WAV (16kHz, mono, PCM)

Шаг 2: Подготовка данных (1-2s)
├─ Загрузка avatar.jpg (уже в памяти!)
├─ Извлечение mel-spectrogram из аудио
└─ Разделение на chunks

Шаг 3: Face Detection (1-3s)
├─ Детекция лица на изображении (модель уже загружена!)
├─ Определение координат
└─ Сглаживание боксов

Шаг 4: Lip-sync (5-15s на GPU, 20-40s на CPU)
├─ Подготовка батчей (изображение + аудио chunks)
├─ Инференс Wav2Lip модели (модель уже загружена!)
├─ Генерация синхронизированных фреймов
└─ Сборка видео

Шаг 5: Финализация (1-2s)
├─ Объединение видео с аудио (ffmpeg)
├─ Сохранение в outputs/
└─ Возврат MP4 клиенту

Итого: 10-30s в зависимости от CPU/GPU
```

## 📁 Файловая структура

```
/workspace/
├── app.py                          # Flask сервер (главный файл)
├── start_web.sh                    # Скрипт быстрого запуска
├── test_api.py                     # Тестирование API
│
├── templates/
│   └── index.html                  # Веб-интерфейс
│
├── modern-lipsync/                 # Модули Wav2Lip
│   ├── service.py                  # LipsyncService класс
│   ├── models/                     # Модели Wav2Lip, SyncNet
│   ├── face_detection/             # Детекция лиц
│   └── utils/                      # Audio processing
│
├── avatar.jpg                      # Изображение аватара (предзагружено)
├── Wav2Lip-SD-GAN.pt              # Модель Wav2Lip (предзагружена)
│
├── outputs/                        # Сгенерированные видео
│   └── avatar_*.mp4
│
├── temp_web/                       # Временные файлы
│   ├── audio_*.wav
│   └── temp_*.mp3
│
├── requirements_web.txt            # Python зависимости
├── README_WEB.md                   # Полная документация
└── QUICKSTART.md                   # Быстрый старт
```

## 🧩 Ключевые компоненты

### 1. Flask Server (`app.py`)

**Глобальные переменные:**
```python
lipsync_service = None      # Предзагруженный сервис Wav2Lip
avatar_preloaded = None     # Изображение аватара в памяти
```

**Основные функции:**
- `init_service()` - Инициализация с загрузкой моделей
- `generate_tts()` - Вызов TTS API
- `convert_to_wav()` - Конвертация аудио в нужный формат
- Endpoints для веб-интерфейса и API

### 2. LipsyncService (`modern-lipsync/service.py`)

**Предзагруженные модели:**
```python
self.model              # Wav2Lip нейросеть (JIT)
self.face_detector      # Face detection модель
self.audio_processor    # Обработчик аудио
```

**Методы:**
- `detect_faces()` - Детекция лиц (использует предзагруженную модель)
- `process()` - Полный цикл обработки
- `_load_video()` - Загрузка видео/изображения
- `_process_audio()` - Извлечение mel-spectrogram
- `_run_inference()` - Инференс модели

### 3. Web Interface (`templates/index.html`)

**Компоненты:**
- Textarea для ввода текста
- Radio buttons для выбора языка
- Кнопка генерации
- Video player для результата
- Статистика обработки

**JavaScript:**
- Асинхронные запросы к API
- Обработка Blob для видео
- Progress индикаторы
- Download функция

## ⚡ Оптимизации

### Предзагрузка моделей
```python
# ❌ Медленно (загрузка при каждом запросе)
def process(face, audio):
    model = load_model()  # 5-10s каждый раз!
    result = model.predict(...)
    return result

# ✅ Быстро (загрузка один раз)
# При старте:
global_model = load_model()  # 5-10s только один раз

# При запросе:
def process(face, audio):
    result = global_model.predict(...)  # Мгновенно!
    return result
```

### Кэширование аватара
```python
# Изображение загружается один раз при старте
avatar_preloaded = cv2.imread(AVATAR_IMAGE)

# При каждом запросе используется из памяти
face_data = avatar_preloaded.copy()  # Быстрая операция
```

### Batch Processing
```python
# Обработка фреймов батчами для эффективности GPU
batch_size = 128  # Обрабатываем 128 фреймов за раз
for i in range(0, len(frames), batch_size):
    batch = frames[i:i + batch_size]
    results = model(batch)  # GPU эффективно!
```

## 🎯 API Спецификация

### POST `/api/generate`

**Request:**
```json
{
  "text": "Привет! Как дела?",
  "language": "ru"
}
```

**Response:**
- Content-Type: `video/mp4`
- Binary video data

**Errors:**
```json
{
  "error": "Текст не может быть пустым"
}
```

### GET `/api/health`

**Response:**
```json
{
  "status": "ready",
  "models_loaded": true,
  "avatar_loaded": true,
  "device": "cuda"
}
```

## 🔧 Конфигурация

### Изменяемые параметры

```python
# app.py
AVATAR_IMAGE = '/workspace/avatar.jpg'          # Путь к аватару
CHECKPOINT_PATH = '/workspace/Wav2Lip-SD-GAN.pt'  # Модель
TTS_API_URL = 'https://tts.sk-ai.kz/api/tts'   # TTS сервис

# Производительность
face_det_batch_size = 16    # Батч для face detection
wav2lip_batch_size = 128    # Батч для lip-sync

# Сеть
port = 3000                  # Порт сервера
```

## 📊 Производительность

### Время выполнения (секунды)

| Этап | CPU | GPU (CUDA) | Примечание |
|------|-----|------------|------------|
| Инициализация (один раз) | 10-15 | 5-8 | При старте сервера |
| TTS генерация | 2-5 | 2-5 | Зависит от TTS API |
| Конвертация аудио | 1-2 | 1-2 | ffmpeg |
| Face detection | 2-4 | 1-2 | Статичное изображение |
| Lip-sync (10s аудио) | 20-40 | 5-10 | Основная нагрузка |
| Финализация | 1-2 | 1-2 | Сборка видео |
| **Итого на запрос** | **30-60** | **12-25** | |

### Использование памяти

- Модель Wav2Lip: ~200-300 MB
- Face Detection: ~100-150 MB
- Аватар в памяти: ~1-5 MB
- Временные буферы: ~50-100 MB
- **Итого**: ~500 MB - 1 GB

## 🚀 Развертывание

### Development
```bash
python app.py
```

### Production (с gunicorn)
```bash
pip install gunicorn
gunicorn -w 2 -b 0.0.0.0:3000 --timeout 120 app:app
```

### Docker
```dockerfile
FROM python:3.9
WORKDIR /app
COPY requirements_web.txt .
RUN pip install -r requirements_web.txt
COPY . .
CMD ["python", "app.py"]
```

## 🔒 Безопасность

### Рекомендации для production:

1. **Rate limiting** - ограничение запросов
2. **Аутентификация** - JWT/API ключи
3. **Валидация** - длина текста, тип файлов
4. **HTTPS** - SSL сертификат
5. **CORS** - настройка разрешенных доменов
6. **Очистка** - автоудаление старых файлов

## 📈 Масштабирование

### Горизонтальное
- Nginx load balancer
- Несколько инстансов Flask
- Shared storage для outputs

### Вертикальное
- Больше GPU памяти
- Увеличение batch_size
- SSD диски для быстрого I/O

---

**Автор:** AI Assistant  
**Дата:** 28 октября 2025  
**Версия:** 1.0
