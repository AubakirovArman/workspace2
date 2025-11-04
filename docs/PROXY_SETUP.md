# 🔧 Настройка для работы через Reverse Proxy

## 🎯 Проблема

Если ваш сайт использует reverse proxy или роутинг с префиксами (например `/r/`), запросы приходят как:

```
❌ http://your-site.com/r/api/stream_chunks
```

Вместо:

```
✅ http://localhost:3000/api/stream_chunks
```

---

## ✅ Решение

Теперь API поддерживает **оба варианта**:

### Без префикса:
```
POST /api/stream_chunks
GET  /api/chunk/video/{id}
GET  /api/chunk/audio/{id}
GET  /api/health
POST /api/generate
POST /api/generate_stream
GET  /api/avatar
POST /api/cleanup
```

### С префиксом `/r/`:
```
POST /r/api/stream_chunks
GET  /r/api/chunk/video/{id}
GET  /r/api/chunk/audio/{id}
GET  /r/api/health
POST /r/api/generate
POST /r/api/generate_stream
GET  /r/api/avatar
POST /r/api/cleanup
```

---

## 📊 Логирование

Теперь все входящие запросы логируются:

```
================================================================================
📨 Входящий запрос:
   Метод: POST
   URL: http://localhost:3000/r/api/stream_chunks
   Path: /r/api/stream_chunks
   Remote IP: 100.64.0.27
   Headers:
      Host: localhost:3000
      Content-Type: application/json
      Origin: https://your-site.com
   Body: {"text":"Hello world","language":"ru"}
================================================================================
📤 Ответ: 200
```

Это поможет отладить проблемы с интеграцией.

---

## 🌐 Примеры использования

### JavaScript (с префиксом)

```javascript
// С префиксом /r/
const response = await fetch('http://your-api.com/r/api/stream_chunks', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ text: 'Привет!', language: 'ru' })
});

const { chunks } = await response.json();

// URL'ы в ответе будут с /r/ префиксом
for (const chunk of chunks) {
  // chunk.video_url = "/r/api/chunk/video/..."
  // chunk.audio_url = "/r/api/chunk/audio/..."
}
```

### Python (с префиксом)

```python
import requests

# С префиксом /r/
response = requests.post(
    'http://your-api.com/r/api/stream_chunks',
    json={'text': 'Привет!', 'language': 'ru'}
)

chunks = response.json()['chunks']
```

---

## 🔍 Отладка

### 1. Проверьте логи сервера

Запустите сервер и смотрите логи в консоли:

```bash
python app.py
```

Вы увидите все входящие запросы с полными заголовками.

### 2. Проверьте CORS

Если получаете ошибку CORS:

```
Access to fetch at 'http://...' from origin 'https://...' 
has been blocked by CORS policy
```

**Решение:** CORS уже включен для всех доменов в `app.py`:

```python
CORS(app)  # Разрешает все домены
```

Если нужно ограничить домены:

```python
CORS(app, origins=['https://your-site.com'])
```

### 3. Проверьте OPTIONS запросы

Браузер отправляет preflight OPTIONS запрос перед POST:

```
OPTIONS /r/api/stream_chunks HTTP/1.1
```

**Решение:** Flask автоматически обрабатывает OPTIONS при использовании CORS.

---

## 🚀 Nginx Reverse Proxy

Если используете Nginx, добавьте в конфигурацию:

```nginx
location /r/ {
    proxy_pass http://localhost:3000/;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
    
    # CORS headers
    add_header 'Access-Control-Allow-Origin' '*';
    add_header 'Access-Control-Allow-Methods' 'GET, POST, OPTIONS';
    add_header 'Access-Control-Allow-Headers' 'Content-Type';
    
    # Handle OPTIONS
    if ($request_method = 'OPTIONS') {
        return 204;
    }
}
```

Теперь запросы на `http://your-site.com/r/api/*` будут проксироваться на `http://localhost:3000/r/api/*`

---

## 🧪 Тестирование

### Без префикса:
```bash
curl -X POST http://localhost:3000/api/stream_chunks \
  -H "Content-Type: application/json" \
  -d '{"text":"Hello","language":"ru"}'
```

### С префиксом /r/:
```bash
curl -X POST http://localhost:3000/r/api/stream_chunks \
  -H "Content-Type: application/json" \
  -d '{"text":"Hello","language":"ru"}'
```

Оба запроса должны работать одинаково! ✅

---

## 📋 Checklist для интеграции

- [ ] Проверьте, какой URL используется на вашем сайте
- [ ] Добавьте префикс `/r/` если нужно
- [ ] Проверьте логи сервера для отладки
- [ ] Убедитесь, что CORS настроен правильно
- [ ] Проверьте, что OPTIONS запросы обрабатываются
- [ ] Тестируйте с curl перед интеграцией в код

---

## 💡 Подсказка

**Смотрите логи в реальном времени:**

```bash
python app.py | grep "📨"
```

Это покажет только входящие запросы для быстрой отладки.

---

**Теперь API работает с любыми префиксами!** 🎉
