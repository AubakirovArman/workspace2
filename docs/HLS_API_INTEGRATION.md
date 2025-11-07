# Realtime HLS API Integration

Эта справка описывает минимальный набор запросов, необходимых для использования режима realtime/HLS в стороннем веб-приложении. Предполагается, что Avatar сервис уже запущен (по умолчанию `http://<host>:3030`) и держит модели в памяти.

## 1. Старт потока

`POST /api/stream/start`

Payload (JSON):

```json
{
  "text": "Привет! Я готов к интеграции.",
  "language": "ru",
  "mode": "dynamic"
}
```

- `text` – обязательный, текст.
- `language` – `ru`, `kk` или `en`.
- `mode` – сейчас поддерживается только `dynamic`.

Ответ: `200 OK`

```json
{
  "session_id": "20241121-123045-abc123",
  "playlist_url": "/stream/20241121-123045-abc123/playlist.m3u8",
  "status": "processing"
}
```

> `playlist_url` отдаётся **относительным** путём. На фронтенде добавьте ваш хост/протокол, например `new URL(playlistUrl, window.location.origin).href`.

Возможные ошибки: `400` (пустой текст / неподдерживаемый язык), `500` (внутренняя ошибка).

### Что делать на фронтенде

1. Сразу после ответа инициализировать hls.js (или другой HLS-плеер) и прикрепить `playlist_url` к `<video>` тегу.
2. Разрешить автозапуск со звуком (или показать кнопку Play).
3. Пока первый сегмент (~1 секунда видео) не сгенерирован, плеер будет ждать. При обычном тексте задержка 4–6 секунд, при длинных блоках 6–10 секунд.

Пример инициализации на hls.js:

```javascript
import Hls from 'hls.js';

const video = document.getElementById('avatar-video');
const hls = new Hls({
  lowLatencyMode: true,
  liveSyncDurationCount: 1,
  liveMaxLatencyDurationCount: 3,
  backBufferLength: 0
});
hls.loadSource(new URL(playlistUrl, window.location.origin).href);
hls.attachMedia(video);
hls.on(Hls.Events.MANIFEST_PARSED, () => video.play().catch(() => {}));
hls.on(Hls.Events.ERROR, (_, data) => {
  if (data.fatal && data.type === Hls.ErrorTypes.NETWORK_ERROR) {
    hls.startLoad();
  }
});

// Автоплей после первого сегмента
hls.on(Hls.Events.FRAG_BUFFERED, () => {
  if (video.paused) {
    video.play().catch(() => {});
  }
});
```

## 2. Проверка статуса

`GET /api/stream/status/<session_id>`

Ответ при `200 OK`:

```json
{
  "session_id": "20241121-123045-abc123",
  "status": "processing",          // или "ready", "failed"
  "playlist_url": "http://<host>:3030/hls/.../index.m3u8",
  "mp4_url": "http://<host>:3030/hls/.../output.mp4",
  "error": null,
  "created_at": "2024-11-21T12:30:45.123456",
  "finished_at": null
}
```

- Запрос можно делать раз в 2–3 секунды, чтобы показать прогресс пользователю.
- Когда `status == "ready"`, можно предложить скачать конечный MP4 (`mp4_url`).

Коды ошибок: `404` – неизвестная сессия; `500` – внутренняя ошибка.

### Как добиться мгновенного старта

- Отправляйте `POST /api/stream/start` как можно раньше (например, сразу после того как пользователь завершил ввод текста).
- Сохраняйте `session_id` и `playlist_url` и сразу инициализируйте HLS-плеер — первый сегмент появится автоматически как только FFmpeg закодирует ~1 секунду видео.
- Не блокируйте плеер на проверку статуса: `playlist.m3u8` обновляется инкрементально, hls.js сам догрузит свежие сегменты.
- При сетевых сбоях вызывайте `hls.startLoad()` — это перезапустит загрузку плейлиста без перезапуска сессии.
- Если используете собственный плеер, запрашивайте `playlist.m3u8` до тех пор, пока файл не превысит 200 байт (после добавления первого `#EXTINF`).

## 3. Очистка (опционально)

Сервис сам удаляет старые HLS-сессии. Если необходимо мгновенно освободить место, можно вызвать `POST /api/cleanup` с `session_id` (см. `app_core/routes/api/cleanup.py`).

## 4. Раздача HLS статикой

- HLS сегменты лежат в `temp_web/segments/<session_id>/`.
- Для внешнего доступа лучше проксировать их веб-сервером (Nginx/CloudFront). Пример location:

```
location /hls/ {
    alias /home/arman/musetalk/avatar/temp_web/segments/;
    add_header Cache-Control "no-cache";
    types {
        application/vnd.apple.mpegurl m3u8;
        video/mp2t ts;
    }
}
```

## 5. Авторизация и CORS

- Если сервис открывается из другого приложения, включите CORS (например, `pip install flask-cors`) и whitelist домены.
- Добавьте простой API-key: проверяйте заголовок (`X-Avatar-Key`) перед запуском `start_stream_job`.

## 6. Потоковая схема

```
Client → POST /api/stream/start → session_id, playlist_url
Client video player ↘
                    fetch playlist (.m3u8) ↘
                       fetch segments (.ts) ← вызывается сервисом HLS streamer
Client → периодический GET /api/stream/status → следим за завершением / ошибками
```

## 7. Частые вопросы

- **Почему нет звука сразу?** Нужно дождаться первого сегмента (6–10 сек). Плеер должен быть настроен на autoplay/auto-retry (см. `templates/realtime.html`).
- **Можно ли менять голос?** Добавьте поле `voice` в `POST /api/stream/start` и прокиньте в `generate_tts`.
- **Как узнать прогресс?** Используйте `status` и вычисляйте число готовых сегментов через частоту `.ts` файлов.

Этого достаточно, чтобы встроить realtime-avatar в сторонний фронтенд: стартуем поток, прикручиваем hls.js, обновляем интерфейс по статусу, раздаём статический каталог сегментов и при необходимости защищаем API ключом.
