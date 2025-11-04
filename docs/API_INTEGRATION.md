# 🔌 API для интеграции с другими сайтами

## Обзор

Новый API endpoint `/api/stream_chunks` позволяет получать видео и аудио чанками для интеграции с внешними сайтами.

---

## 📡 Endpoint: `/api/stream_chunks`

### Запрос

**POST** `http://localhost:3000/api/stream_chunks`

**Headers:**
```
Content-Type: application/json
```

**Body:**
```json
{
  "text": "Ваш длинный текст для озвучки. Он будет автоматически разбит на чанки.",
  "language": "ru",
  "chunk_size": 15
}
```

**Параметры:**
- `text` (обязательно) - Текст для озвучки
- `language` (опционально, default: "ru") - Язык: `ru`, `kk`, `en`
- `chunk_size` (опционально, default: 15) - Количество слов в одном чанке

---

### Ответ

```json
{
  "success": true,
  "total_chunks": 3,
  "language": "ru",
  "chunks": [
    {
      "index": 0,
      "text": "Первый чанк текста из пятнадцати слов",
      "video_url": "/api/chunk/video/20251028_123456_000000_0",
      "audio_url": "/api/chunk/audio/20251028_123456_000000_0",
      "duration": 3.5
    },
    {
      "index": 1,
      "text": "Второй чанк текста...",
      "video_url": "/api/chunk/video/20251028_123456_100000_1",
      "audio_url": "/api/chunk/audio/20251028_123456_100000_1",
      "duration": 4.2
    }
  ]
}
```

**Поля ответа:**
- `success` - Статус выполнения
- `total_chunks` - Общее количество чанков
- `language` - Использованный язык
- `chunks` - Массив чанков:
  - `index` - Порядковый номер чанка
  - `text` - Текст чанка
  - `video_url` - URL для получения видео
  - `audio_url` - URL для получения аудио
  - `duration` - Длительность в секундах

---

## 📥 Получение чанков

### Видео чанк

**GET** `http://localhost:3000/api/chunk/video/{chunk_id}`

**Response:**
- Type: `video/mp4`
- Содержит видео аватара с синхронизацией губ

### Аудио чанк

**GET** `http://localhost:3000/api/chunk/audio/{chunk_id}`

**Response:**
- Type: `audio/wav`
- Содержит WAV аудио (16kHz, mono, PCM)

---

## 💻 Примеры использования

### JavaScript (Fetch API)

```javascript
// 1. Запрос генерации чанков
async function generateChunks(text, language = 'ru') {
  const response = await fetch('http://localhost:3000/api/stream_chunks', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      text: text,
      language: language,
      chunk_size: 15
    })
  });
  
  const data = await response.json();
  return data.chunks;
}

// 2. Воспроизведение чанков последовательно
async function playChunks(chunks) {
  for (const chunk of chunks) {
    const videoUrl = `http://localhost:3000${chunk.video_url}`;
    const audioUrl = `http://localhost:3000${chunk.audio_url}`;
    
    console.log(`Играю чанк ${chunk.index}: ${chunk.text}`);
    
    // Воспроизведение видео
    const videoElement = document.getElementById('video');
    videoElement.src = videoUrl;
    await videoElement.play();
    
    // Ждем окончания
    await new Promise(resolve => {
      videoElement.onended = resolve;
    });
  }
}

// Использование
const text = "Длинный текст для озвучки и генерации видео с аватаром.";
const chunks = await generateChunks(text, 'ru');
await playChunks(chunks);
```

---

### Python (requests)

```python
import requests

# 1. Генерация чанков
def generate_chunks(text, language='ru', chunk_size=15):
    url = 'http://localhost:3000/api/stream_chunks'
    payload = {
        'text': text,
        'language': language,
        'chunk_size': chunk_size
    }
    
    response = requests.post(url, json=payload)
    data = response.json()
    return data['chunks']

# 2. Скачивание чанков
def download_chunk(chunk, output_dir='./chunks'):
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Скачать видео
    video_url = f"http://localhost:3000{chunk['video_url']}"
    video_response = requests.get(video_url)
    
    video_path = f"{output_dir}/video_{chunk['index']}.mp4"
    with open(video_path, 'wb') as f:
        f.write(video_response.content)
    
    # Скачать аудио
    audio_url = f"http://localhost:3000{chunk['audio_url']}"
    audio_response = requests.get(audio_url)
    
    audio_path = f"{output_dir}/audio_{chunk['index']}.wav"
    with open(audio_path, 'wb') as f:
        f.write(audio_response.content)
    
    print(f"Чанк {chunk['index']} скачан: {video_path}, {audio_path}")

# Использование
text = "Длинный текст для озвучки и генерации видео."
chunks = generate_chunks(text, language='ru')

for chunk in chunks:
    download_chunk(chunk)
```

---

### cURL

```bash
# 1. Генерация чанков
curl -X POST http://localhost:3000/api/stream_chunks \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Привет! Это тестовый текст для генерации видео с аватаром.",
    "language": "ru",
    "chunk_size": 10
  }' | jq .

# 2. Скачивание видео чанка
curl -o chunk_video_0.mp4 \
  http://localhost:3000/api/chunk/video/20251028_123456_000000_0

# 3. Скачивание аудио чанка
curl -o chunk_audio_0.wav \
  http://localhost:3000/api/chunk/audio/20251028_123456_000000_0
```

---

## 🎯 Сценарии использования

### 1. Стриминг на веб-сайте

```html
<!DOCTYPE html>
<html>
<head>
    <title>Avatar Streaming</title>
</head>
<body>
    <video id="avatar-video" width="640" height="480" autoplay></video>
    <audio id="avatar-audio" autoplay></audio>
    
    <script>
        async function streamAvatar(text) {
            const response = await fetch('http://localhost:3000/api/stream_chunks', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text, language: 'ru' })
            });
            
            const { chunks } = await response.json();
            const video = document.getElementById('avatar-video');
            
            for (const chunk of chunks) {
                video.src = `http://localhost:3000${chunk.video_url}`;
                await new Promise(resolve => video.onended = resolve);
            }
        }
        
        streamAvatar("Ваш текст для озвучки");
    </script>
</body>
</html>
```

### 2. Интеграция с чат-ботом

```javascript
// Отправка текста от бота
async function sendBotMessage(message) {
    const chunks = await generateChunks(message, 'ru');
    
    // Показываем видео аватара для каждого чанка
    for (const chunk of chunks) {
        await displayAvatarChunk(chunk);
    }
}

async function displayAvatarChunk(chunk) {
    const video = document.createElement('video');
    video.src = `http://localhost:3000${chunk.video_url}`;
    video.controls = true;
    
    document.getElementById('chat-messages').appendChild(video);
    await video.play();
}
```

### 3. Обработка на сервере

```python
# Генерация и сохранение всех чанков
def process_long_text(text, language='ru'):
    chunks = generate_chunks(text, language)
    
    results = []
    for chunk in chunks:
        download_chunk(chunk, output_dir='./generated_chunks')
        results.append({
            'index': chunk['index'],
            'text': chunk['text'],
            'duration': chunk['duration']
        })
    
    return results

# Объединение чанков в один файл
def merge_chunks(chunk_count, output_file='final_video.mp4'):
    import subprocess
    
    # Создаем список файлов для ffmpeg
    with open('chunks_list.txt', 'w') as f:
        for i in range(chunk_count):
            f.write(f"file './generated_chunks/video_{i}.mp4'\n")
    
    # Объединяем с помощью ffmpeg
    subprocess.run([
        'ffmpeg', '-f', 'concat', '-safe', '0',
        '-i', 'chunks_list.txt',
        '-c', 'copy', output_file
    ])
    
    print(f"Видео объединено: {output_file}")
```

---

## ⚙️ Настройка

### Размер чанков

Оптимальный размер зависит от применения:

```json
{
  "chunk_size": 10   // Короткие чанки (2-3 сек) - для быстрого отклика
}
```

```json
{
  "chunk_size": 20   // Средние чанки (4-6 сек) - баланс
}
```

```json
{
  "chunk_size": 30   // Длинные чанки (7-10 сек) - меньше запросов
}
```

---

## 🔧 Дополнительные endpoints

### Проверка здоровья

```bash
GET /api/health
```

**Response:**
```json
{
  "status": "ready",
  "models_loaded": true,
  "avatar_loaded": true,
  "device": "cuda"
}
```

### Очистка старых файлов

```bash
POST /api/cleanup
```

Удаляет файлы старше 1 часа из папки outputs.

---

## 📊 Производительность

**Время генерации одного чанка (15 слов):**
- TTS генерация: ~2-3 сек
- Lip-sync обработка: ~2-3 сек
- **Итого:** ~4-6 сек на чанк

**Для текста в 100 слов:**
- Чанков: ~7
- Время: ~28-42 сек (последовательная обработка)

---

## ⚠️ Важные моменты

1. **CORS**: API настроен с CORS для использования с других доменов
2. **Очистка**: Файлы чанков удаляются автоматически через 1 час
3. **Параллельность**: Запросы обрабатываются последовательно (threaded=True)
4. **Формат**: Видео - MP4, Аудио - WAV 16kHz mono

---

## 🚀 Быстрый тест

```bash
# Тест API
curl -X POST http://localhost:3000/api/stream_chunks \
  -H "Content-Type: application/json" \
  -d '{"text": "Привет мир! Это тестовое сообщение.", "language": "ru"}' | jq .
```

---

**Готово для интеграции!** 🎉
