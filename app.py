"""
Avatar Lipsync Web Application
Веб-сервис для создания говорящего аватара через TTS и Wav2Lip
Держит все модели в памяти для быстрой обработки
"""
import os
import sys
import time
import subprocess
import io
from pathlib import Path
from datetime import datetime
import requests
import shutil

from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
import torch
import torchaudio
import torchaudio.functional as audio_fn
import logging

# Добавляем путь к modern-lipsync
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'modern-lipsync'))

from service import LipsyncService

app = Flask(__name__)
CORS(app)

# Настройка логирования для отладки
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Middleware для логирования всех запросов
@app.before_request
def log_request_info():
    logger.info('='*80)
    logger.info(f'📨 Входящий запрос:')
    logger.info(f'   Метод: {request.method}')
    logger.info(f'   URL: {request.url}')
    logger.info(f'   Path: {request.path}')
    logger.info(f'   Remote IP: {request.remote_addr}')
    logger.info(f'   Headers:')
    for header, value in request.headers:
        logger.info(f'      {header}: {value}')
    if request.method in ['POST', 'PUT', 'PATCH']:
        logger.info(f'   Body: {request.get_data(as_text=True)[:500]}...')
    logger.info('='*80)

@app.after_request
def log_response_info(response):
    logger.info(f'📤 Ответ: {response.status_code}')
    return response

# Конфигурация
AVATAR_IMAGE = '/workspace/avatar.jpg'
CHECKPOINT_PATH = '/workspace/Wav2Lip-SD-GAN.pt'
CHECKPOINT_PATH_NOGAN = '/workspace/Wav2Lip-SD-NOGAN.pt'  # Для realtime2
TTS_API_URL = 'https://tts.sk-ai.kz/api/tts'
OUTPUT_DIR = '/workspace/outputs'
TEMP_DIR = '/workspace/temp_web'

# Создаем необходимые директории
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# Глобальные сервисы с предзагруженными моделями
lipsync_service = None  # GAN модель (для realtime)
lipsync_service_nogan = None  # NOGAN модель (для realtime2)
avatar_preloaded = None


def init_service():
    """Инициализация сервиса с предзагрузкой моделей и аватара"""
    global lipsync_service, lipsync_service_nogan, avatar_preloaded
    
    print("\n" + "="*60)
    print("🚀 Инициализация Avatar Lipsync Service")
    print("="*60)
    
    # Проверка файлов
    if not os.path.exists(AVATAR_IMAGE):
        raise FileNotFoundError(f"Аватар не найден: {AVATAR_IMAGE}")
    
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Модель GAN не найдена: {CHECKPOINT_PATH}")
    
    if not os.path.exists(CHECKPOINT_PATH_NOGAN):
        raise FileNotFoundError(f"Модель NOGAN не найдена: {CHECKPOINT_PATH_NOGAN}")
    
    print(f"✅ Аватар найден: {AVATAR_IMAGE}")
    print(f"✅ Модель GAN найдена: {CHECKPOINT_PATH}")
    print(f"✅ Модель NOGAN найдена: {CHECKPOINT_PATH_NOGAN}")
    
    # Определяем устройство
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🔧 Устройство: {device}")
    if device == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
        if hasattr(torch, 'set_float32_matmul_precision'):
            torch.set_float32_matmul_precision('high')
    
    # Загружаем GAN сервис (для realtime)
    print("\n📦 Загрузка GAN модели в память...")
    start = time.time()
    
    lipsync_service = LipsyncService(
        checkpoint_path=CHECKPOINT_PATH,
        device=device,
        face_det_batch_size=16,
        wav2lip_batch_size=128
    )
    model_ready_time = time.time()
    print(f"✅ GAN модель загружена за {model_ready_time - start:.2f}s")
    
    # Предзагрузка данных для статичного аватара (ускоряет /realtime)
    preload_start = time.time()
    lipsync_service.preload_static_face(
        face_path=AVATAR_IMAGE,
        fps=25.0,
        pads=(0, 50, 0, 0)
    )
    print(f"⚡ Предобработка аватара (GAN) завершена за {time.time() - preload_start:.2f}s")
    
    # Загружаем NOGAN сервис (для realtime2)
    print("\n📦 Загрузка NOGAN модели в память...")
    start = time.time()
    
    lipsync_service_nogan = LipsyncService(
        checkpoint_path=CHECKPOINT_PATH_NOGAN,
        device=device,
        face_det_batch_size=16,
        wav2lip_batch_size=128
    )
    model_ready_time = time.time()
    print(f"✅ NOGAN модель загружена за {model_ready_time - start:.2f}s")
    
    # Предзагрузка данных для NOGAN
    preload_start = time.time()
    lipsync_service_nogan.preload_static_face(
        face_path=AVATAR_IMAGE,
        fps=25.0,
        pads=(0, 50, 0, 0)
    )
    print(f"⚡ Предобработка аватара (NOGAN) завершена за {time.time() - preload_start:.2f}s")
    
    # Предзагрузка аватара в память
    print(f"\n🖼️  Предзагрузка аватара...")
    import cv2
    avatar_preloaded = cv2.imread(AVATAR_IMAGE)
    print(f"✅ Аватар загружен: {avatar_preloaded.shape}")
    
    print("\n" + "="*60)
    print("✅ Сервис полностью готов к работе!")
    print("="*60 + "\n")


def generate_tts(text: str, language: str = 'ru') -> bytes:
    """
    Генерация TTS через API
    
    Args:
        text: Текст для озвучки
        language: Язык (ru, kk, en)
        
    Returns:
        Аудио данные в формате MP3
    """
    print(f"🎤 Генерация TTS...")
    print(f"   Текст: {text[:50]}{'...' if len(text) > 50 else ''}")
    print(f"   Язык: {language}")
    
    try:
        response = requests.post(
            TTS_API_URL,
            json={'text': text, 'lang': language},
            timeout=30
        )
        response.raise_for_status()
        
        audio_data = response.content
        print(f"✅ TTS сгенерирован: {len(audio_data) / 1024:.2f} KB")
        return audio_data
        
    except Exception as e:
        print(f"❌ Ошибка TTS: {e}")
        raise


def convert_to_wav(mp3_data: bytes, output_path: str):
    """Конвертация MP3 в WAV 16kHz mono без лишних файлов"""
    print(f"🔄 Конвертация в WAV...")
    
    audio_buffer = io.BytesIO(mp3_data)
    waveform = None
    sample_rate = None
    target_sr = 16000
    
    try:
        waveform, sample_rate = torchaudio.load(audio_buffer, format='mp3')
        waveform = waveform.float()
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sample_rate != target_sr:
            waveform = audio_fn.resample(waveform, sample_rate, target_sr)
            sample_rate = target_sr
        
        torchaudio.save(output_path, waveform.cpu(), sample_rate)
        print(f"✅ WAV сохранен: {output_path}")
        return waveform, sample_rate
    except Exception as decode_error:
        print(f"⚠️ torchaudio не смог декодировать MP3 ({decode_error}); fallback на ffmpeg.")
        
        temp_mp3 = os.path.join(TEMP_DIR, f'temp_{int(time.time())}.mp3')
        with open(temp_mp3, 'wb') as f:
            f.write(mp3_data)
        
        cmd = [
            'ffmpeg', '-y', '-i', temp_mp3,
            '-ar', str(target_sr),
            '-ac', '1',
            '-f', 'wav',
            '-acodec', 'pcm_s16le',
            output_path,
            '-loglevel', 'error'
        ]
        subprocess.run(cmd, check=True)
        os.remove(temp_mp3)
        
        waveform, sample_rate = torchaudio.load(output_path)
        waveform = waveform.float()
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        print(f"✅ WAV сохранен: {output_path}")
        return waveform, sample_rate


@app.route('/')
def index():
    """Главная страница - создание и скачивание"""
    return render_template('index.html')


@app.route('/realtime')
def realtime():
    """Страница реалтайм озвучки (GAN модель)"""
    return render_template('realtime.html')


@app.route('/realtime2')
def realtime2():
    """Страница реалтайм озвучки (NOGAN модель - лучше для персонажей без зубов)"""
    return render_template('realtime2.html')


@app.route('/realtime3')
def realtime3():
    """Страница реалтайм озвучки с настройками (экспериментальная)"""
    return render_template('realtime3.html')


@app.route('/api-test')
def api_test():
    """Страница тестирования API интеграции"""
    return render_template('api_test.html')


@app.route('/test-long')
def test_long():
    """Страница тестирования длинного текста"""
    return send_file('/workspace/test_long_text.html')


@app.route('/api/health')
@app.route('/r/api/health')  # Поддержка префикса /r/
def health():
    """Проверка состояния сервиса"""
    return jsonify({
        'status': 'ready',
        'models_loaded': lipsync_service is not None,
        'avatar_loaded': avatar_preloaded is not None,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    })


@app.route('/api/generate', methods=['POST'])
@app.route('/r/api/generate', methods=['POST'])  # Поддержка префикса /r/
def generate_avatar_speech():
    """
    Генерация говорящего аватара
    
    POST /api/generate
    {
        "text": "Текст для озвучки",
        "language": "ru"  // опционально, по умолчанию ru
    }
    
    Returns:
        Video file (MP4) с говорящим аватаром
    """
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'Требуется поле "text"'}), 400
        
        text = data['text'].strip()
        language = data.get('language', 'ru')
        
        if not text:
            return jsonify({'error': 'Текст не может быть пустым'}), 400
        
        if language not in ['ru', 'kk', 'en']:
            return jsonify({'error': 'Неподдерживаемый язык'}), 400
        
        print("\n" + "="*60)
        print(f"🎬 Новый запрос генерации")
        print("="*60)
        print(f"Текст: {text}")
        print(f"Язык: {language}")
        
        start_total = time.time()
        
        # 1. Генерация TTS
        start = time.time()
        audio_data = generate_tts(text, language)
        tts_time = time.time() - start
        
        # 2. Конвертация в WAV
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        audio_path = os.path.join(TEMP_DIR, f'audio_{timestamp}.wav')
        
        start = time.time()
        audio_waveform, audio_sample_rate = convert_to_wav(audio_data, audio_path)
        convert_time = time.time() - start
        
        # 3. Генерация lip-sync (модели уже в памяти!)
        output_path = os.path.join(OUTPUT_DIR, f'avatar_{timestamp}.mp4')
        
        print(f"\n🎭 Генерация lip-sync...")
        start = time.time()
        
        stats = lipsync_service.process(
            face_path=AVATAR_IMAGE,
            audio_path=audio_path,
            output_path=output_path,
            static=True,  # Статичное изображение
            pads=(0, 50, 0, 0),
            fps=25.0,
            audio_waveform=audio_waveform,
            audio_sample_rate=audio_sample_rate
        )
        
        lipsync_time = time.time() - start
        total_time = time.time() - start_total
        
        # Очистка временных файлов
        if os.path.exists(audio_path):
            os.remove(audio_path)
        
        # Статистика
        print(f"\n📊 Статистика:")
        print(f"   TTS генерация:    {tts_time:.2f}s")
        print(f"   Конвертация:      {convert_time:.2f}s")
        print(f"   Lip-sync:         {lipsync_time:.2f}s")
        print(f"     - Загрузка видео:   {stats['load_video_time']:.2f}s")
        print(f"     - Обработка аудио:  {stats['process_audio_time']:.2f}s")
        print(f"     - Детекция лица:    {stats['face_detection_time']:.2f}s")
        print(f"     - Инференс модели:  {stats['inference_time']:.2f}s")
        print(f"   ─────────────────────────")
        print(f"   ИТОГО:            {total_time:.2f}s")
        print(f"\n✅ Видео готово: {output_path}")
        print("="*60 + "\n")
        
        # Отправляем видео файл
        return send_file(
            output_path,
            mimetype='video/mp4',
            as_attachment=True,
            download_name=f'avatar_speech.mp4'
        )
        
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/avatar')
@app.route('/r/api/avatar')  # Поддержка префикса /r/
def get_avatar():
    """Получить изображение аватара"""
    return send_file(AVATAR_IMAGE, mimetype='image/jpeg')


@app.route('/api/generate_stream', methods=['POST'])
@app.route('/r/api/generate_stream', methods=['POST'])  # Поддержка префикса /r/
def generate_stream_chunk():
    """
    Генерация видео-чанка с lip-sync для реалтайм озвучки
    Быстрая обработка коротких фрагментов
    
    POST /api/generate_stream
    {
        "text": "Текст чанка",
        "language": "ru",
        "chunk_index": 0
    }
    
    Returns:
        Video file (MP4) - с синхронизацией губ
    """
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'Требуется поле "text"'}), 400
        
        text = data['text'].strip()
        language = data.get('language', 'ru')
        chunk_index = data.get('chunk_index', 0)
        
        if not text:
            return jsonify({'error': 'Текст не может быть пустым'}), 400
        
        print(f"\n� Реалтайм чанк #{chunk_index}: {text[:50]}...")
        start_total = time.time()
        
        # 1. TTS генерация
        start = time.time()
        audio_data = generate_tts(text, language)
        tts_time = time.time() - start
        
        # 2. Конвертация в WAV
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        audio_path = os.path.join(TEMP_DIR, f'chunk_{chunk_index}_{timestamp}.wav')
        
        start = time.time()
        audio_waveform, audio_sample_rate = convert_to_wav(audio_data, audio_path)
        convert_time = time.time() - start
        
        # 3. Быстрая генерация lip-sync
        output_path = os.path.join(OUTPUT_DIR, f'chunk_{chunk_index}_{timestamp}.mp4')
        
        start = time.time()
        stats = lipsync_service.process(
            face_path=AVATAR_IMAGE,
            audio_path=audio_path,
            output_path=output_path,
            static=True,
            pads=(0, 50, 0, 0),
            fps=25.0,
            audio_waveform=audio_waveform,
            audio_sample_rate=audio_sample_rate
        )
        lipsync_time = time.time() - start
        total_time = time.time() - start_total
        
        # Очистка временных файлов
        if os.path.exists(audio_path):
            os.remove(audio_path)
        
        print(f"✅ Чанк #{chunk_index} готов за {total_time:.2f}s (TTS: {tts_time:.2f}s, Sync: {lipsync_time:.2f}s)")
        
        # Отправляем видео
        return send_file(
            output_path,
            mimetype='video/mp4',
            as_attachment=False,
            download_name=f'chunk_{chunk_index}.mp4'
        )
        
    except Exception as e:
        print(f"\n❌ Ошибка stream: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/generate_stream_nogan', methods=['POST'])
@app.route('/r/api/generate_stream_nogan', methods=['POST'])  # Поддержка префикса /r/
def generate_stream_chunk_nogan():
    """
    Генерация видео-чанка с lip-sync для реалтайм озвучки (GAN модель с зубами)
    Использует GAN модель для генерации зубов
    
    POST /api/generate_stream_nogan
    {
        "text": "Текст чанка",
        "language": "ru",
        "chunk_index": 0
    }
    
    Returns:
        Video file (MP4) - с синхронизацией губ (GAN - с зубами)
    """
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'Требуется поле "text"'}), 400
        
        text = data['text'].strip()
        language = data.get('language', 'ru')
        chunk_index = data.get('chunk_index', 0)
        
        if not text:
            return jsonify({'error': 'Текст не может быть пустым'}), 400
        
        print(f"\n🦷 Реалтайм2 (GAN с зубами) чанк #{chunk_index}: {text[:50]}...")
        start_total = time.time()
        
        # 1. TTS генерация
        start = time.time()
        audio_data = generate_tts(text, language)
        tts_time = time.time() - start
        
        # 2. Конвертация в WAV
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        audio_path = os.path.join(TEMP_DIR, f'chunk_gan2_{chunk_index}_{timestamp}.wav')
        
        start = time.time()
        audio_waveform, audio_sample_rate = convert_to_wav(audio_data, audio_path)
        convert_time = time.time() - start
        
        # 3. Быстрая генерация lip-sync с GAN моделью (будут зубы!)
        output_path = os.path.join(OUTPUT_DIR, f'chunk_gan2_{chunk_index}_{timestamp}.mp4')
        
        start = time.time()
        stats = lipsync_service.process(  # ← Используем GAN вместо NOGAN!
            face_path=AVATAR_IMAGE,
            audio_path=audio_path,
            output_path=output_path,
            static=True,
            pads=(0, 50, 0, 0),
            fps=25.0,
            audio_waveform=audio_waveform,
            audio_sample_rate=audio_sample_rate
        )
        lipsync_time = time.time() - start
        total_time = time.time() - start_total
        
        # Очистка временных файлов
        if os.path.exists(audio_path):
            os.remove(audio_path)
        
        print(f"✅ GAN чанк #{chunk_index} готов за {total_time:.2f}s (TTS: {tts_time:.2f}s, Sync: {lipsync_time:.2f}s)")
        
        # Отправляем видео
        return send_file(
            output_path,
            mimetype='video/mp4',
            as_attachment=False,
            download_name=f'chunk_gan2_{chunk_index}.mp4'
        )
        
    except Exception as e:
        print(f"\n❌ Ошибка stream GAN: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/generate_stream_custom', methods=['POST'])
@app.route('/r/api/generate_stream_custom', methods=['POST'])
def generate_stream_chunk_custom():
    """
    Генерация видео-чанка с настраиваемыми параметрами
    
    POST /api/generate_stream_custom
    {
        "text": "Текст чанка",
        "language": "ru",
        "chunk_index": 0,
        "model": "gan" или "nogan",
        "pad_top": 0,
        "pad_bottom": 10,
        "pad_left": 0,
        "pad_right": 0
    }
    
    Returns:
        Video file (MP4) - с синхронизацией губ
    """
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'Требуется поле "text"'}), 400
        
        text = data['text'].strip()
        language = data.get('language', 'ru')
        chunk_index = data.get('chunk_index', 0)
        model_type = data.get('model', 'gan')  # gan или nogan
        
        # Параметры padding
        pad_top = data.get('pad_top', 0)
        pad_bottom = data.get('pad_bottom', 50)
        pad_left = data.get('pad_left', 0)
        pad_right = data.get('pad_right', 0)
        
        if not text:
            return jsonify({'error': 'Текст не может быть пустым'}), 400
        
        print(f"\n🎛️ Кастомный чанк #{chunk_index} ({model_type.upper()}):")
        print(f"   Pads: top={pad_top}, bottom={pad_bottom}, left={pad_left}, right={pad_right}")
        print(f"   Текст: {text[:50]}...")
        
        start_total = time.time()
        
        # 1. TTS генерация
        start = time.time()
        audio_data = generate_tts(text, language)
        tts_time = time.time() - start
        
        # 2. Конвертация в WAV
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        audio_path = os.path.join(TEMP_DIR, f'chunk_custom_{chunk_index}_{timestamp}.wav')
        
        start = time.time()
        audio_waveform, audio_sample_rate = convert_to_wav(audio_data, audio_path)
        convert_time = time.time() - start
        
        # 3. Выбор модели
        service = lipsync_service if model_type == 'gan' else lipsync_service_nogan
        
        # 4. Генерация lip-sync с кастомными параметрами
        output_path = os.path.join(OUTPUT_DIR, f'chunk_custom_{chunk_index}_{timestamp}.mp4')
        
        start = time.time()
        stats = service.process(
            face_path=AVATAR_IMAGE,
            audio_path=audio_path,
            output_path=output_path,
            static=True,
            pads=(pad_top, pad_bottom, pad_left, pad_right),  # Кастомные padding!
            fps=25.0,
            audio_waveform=audio_waveform,
            audio_sample_rate=audio_sample_rate
        )
        lipsync_time = time.time() - start
        total_time = time.time() - start_total
        
        # Очистка временных файлов
        if os.path.exists(audio_path):
            os.remove(audio_path)
        
        print(f"✅ Кастом чанк #{chunk_index} готов за {total_time:.2f}s")
        
        # Отправляем видео
        return send_file(
            output_path,
            mimetype='video/mp4',
            as_attachment=False,
            download_name=f'chunk_custom_{chunk_index}.mp4'
        )
        
    except Exception as e:
        print(f"\n❌ Ошибка custom stream: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/stream_chunks', methods=['POST'])
@app.route('/r/api/stream_chunks', methods=['POST'])  # Поддержка префикса /r/
def stream_chunks():
    """
    API для стриминга видео и аудио чанками (для интеграции с другими сайтами)
    
    POST /api/stream_chunks
    {
        "text": "Длинный текст для озвучки",
        "language": "ru",  // опционально, по умолчанию ru
        "chunk_size": 15   // опционально, количество слов в чанке
    }
    
    Returns:
        JSON с массивом чанков:
        {
            "total_chunks": 5,
            "chunks": [
                {
                    "index": 0,
                    "text": "Текст чанка",
                    "video_url": "/api/chunk/video/abc123",
                    "audio_url": "/api/chunk/audio/abc123",
                    "duration": 3.5
                },
                ...
            ]
        }
    """
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'Требуется поле "text"'}), 400
        
        text = data['text'].strip()
        language = data.get('language', 'ru')
        chunk_size = data.get('chunk_size', 15)  # слов в чанке
        
        if not text:
            return jsonify({'error': 'Текст не может быть пустым'}), 400
        
        if language not in ['ru', 'kk', 'en']:
            return jsonify({'error': 'Неподдерживаемый язык'}), 400
        
        print(f"\n🎬 API Stream Chunks: {len(text)} символов, язык: {language}")
        
        # Разбиваем текст на чанки
        words = text.split()
        text_chunks = []
        for i in range(0, len(words), chunk_size):
            chunk_text = ' '.join(words[i:i + chunk_size])
            text_chunks.append(chunk_text)
        
        print(f"📝 Разбито на {len(text_chunks)} чанков")
        
        # Генерируем все чанки
        chunks_info = []
        
        for idx, chunk_text in enumerate(text_chunks):
            print(f"\n🎤 Чанк {idx+1}/{len(text_chunks)}: {chunk_text[:50]}...")
            
            try:
                # 1. Генерация TTS
                audio_data = generate_tts(chunk_text, language)
                
                # 2. Конвертация в WAV
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                chunk_id = f"{timestamp}_{idx}"
                audio_path = os.path.join(TEMP_DIR, f'chunk_audio_{chunk_id}.wav')
                audio_waveform, audio_sample_rate = convert_to_wav(audio_data, audio_path)
                
                # 3. Генерация видео с lip-sync
                video_path = os.path.join(OUTPUT_DIR, f'chunk_video_{chunk_id}.mp4')
                stats = lipsync_service.process(
                    face_path=AVATAR_IMAGE,
                    audio_path=audio_path,
                    output_path=video_path,
                    static=True,
                    pads=(0, 50, 0, 0),
                    fps=25.0,
                    audio_waveform=audio_waveform,
                    audio_sample_rate=audio_sample_rate
                )
                
                # Получаем длительность аудио
                duration = round(audio_waveform.shape[1] / audio_sample_rate, 2)
                
                # Копируем аудио в outputs для доступа
                audio_output_path = os.path.join(OUTPUT_DIR, f'chunk_audio_{chunk_id}.wav')
                shutil.copy(audio_path, audio_output_path)
                
                # Удаляем временный файл
                os.remove(audio_path)
                
                chunks_info.append({
                    'index': idx,
                    'text': chunk_text,
                    'video_url': f'/api/chunk/video/{chunk_id}',
                    'audio_url': f'/api/chunk/audio/{chunk_id}',
                    'duration': duration
                })
                
                print(f"✅ Чанк {idx+1} готов ({duration:.2f}s)")
                
            except Exception as e:
                print(f"❌ Ошибка чанка {idx}: {e}")
                chunks_info.append({
                    'index': idx,
                    'text': chunk_text,
                    'error': str(e)
                })
        
        print(f"\n✅ Все чанки готовы: {len(chunks_info)}")
        
        return jsonify({
            'success': True,
            'total_chunks': len(chunks_info),
            'chunks': chunks_info,
            'language': language
        })
        
    except Exception as e:
        print(f"\n❌ Ошибка API: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/chunk/video/<chunk_id>')
@app.route('/r/api/chunk/video/<chunk_id>')  # Поддержка префикса /r/
def get_chunk_video(chunk_id):
    """Получить видео чанк по ID"""
    try:
        video_path = os.path.join(OUTPUT_DIR, f'chunk_video_{chunk_id}.mp4')
        if not os.path.exists(video_path):
            return jsonify({'error': 'Видео не найдено'}), 404
        
        return send_file(
            video_path,
            mimetype='video/mp4',
            as_attachment=False
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/chunk/audio/<chunk_id>')
@app.route('/r/api/chunk/audio/<chunk_id>')  # Поддержка префикса /r/
def get_chunk_audio(chunk_id):
    """Получить аудио чанк по ID"""
    try:
        audio_path = os.path.join(OUTPUT_DIR, f'chunk_audio_{chunk_id}.wav')
        if not os.path.exists(audio_path):
            return jsonify({'error': 'Аудио не найдено'}), 404
        
        return send_file(
            audio_path,
            mimetype='audio/wav',
            as_attachment=False
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/cleanup', methods=['POST'])
@app.route('/r/api/cleanup', methods=['POST'])  # Поддержка префикса /r/
def cleanup():
    """Очистка старых выходных файлов"""
    try:
        # Удаляем файлы старше 1 часа
        now = time.time()
        removed = 0
        
        for filename in os.listdir(OUTPUT_DIR):
            filepath = os.path.join(OUTPUT_DIR, filename)
            if os.path.isfile(filepath):
                if now - os.path.getmtime(filepath) > 3600:  # 1 час
                    os.remove(filepath)
                    removed += 1
        
        return jsonify({
            'message': f'Удалено файлов: {removed}'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # Инициализация сервиса при старте
    try:
        init_service()
    except Exception as e:
        print(f"\n❌ Ошибка инициализации: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Запуск Flask
    print(f"🌐 Запуск веб-сервера на http://localhost:3000")
    print(f"📝 Откройте браузер и перейдите по адресу")
    print(f"   http://localhost:3000\n")
    
    app.run(
        host='0.0.0.0',
        port=3000,
        debug=False,
        threaded=True
    )
