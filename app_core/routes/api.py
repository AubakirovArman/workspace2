"""Blueprint with JSON API routes."""
from __future__ import annotations

import os
import shutil
import time
from datetime import datetime
from pathlib import Path

import torch
from flask import Blueprint, jsonify, request, send_file

from .. import state
from ..config import AVATAR_FPS, AVATAR_IMAGE, AVATAR_PREVIEW_PATH, OUTPUT_DIR, TEMP_DIR
from ..services import convert_to_wav, generate_tts

api_bp = Blueprint('api', __name__)


def _collect_service_features(service):
    if service is None:
        return None
    return {
        'segmentation': bool(
            getattr(service, 'segmentation_enabled', False)
            and getattr(service, 'segmentation_model', None) is not None
        ),
        'super_resolution': bool(
            getattr(service, 'sr_enabled', False)
            and getattr(service, 'sr_model', None) is not None
        ),
        'real_esrgan': bool(
            getattr(service, 'realesrgan_enabled', False)
            and getattr(service, '_realesrgan_enhancer', None) is not None
        )
    }


@api_bp.route('/api/health')
def health():
    gan_ready = state.lipsync_service_gan is not None
    nogan_ready = state.lipsync_service_nogan is not None

    if gan_ready:
        status = 'ready'
    elif nogan_ready:
        status = 'degraded'
    else:
        status = 'offline'

    return jsonify({
        'status': status,
        'hd_model_loaded': gan_ready,  # backward compatibility
        'gan_model_loaded': gan_ready,
        'nogan_model_loaded': nogan_ready,
        'models': {
            'gan': _collect_service_features(state.lipsync_service_gan),
            'nogan': _collect_service_features(state.lipsync_service_nogan)
        },
        'avatar_loaded': state.avatar_preloaded is not None,
        'avatar_mode': 'static' if state.avatar_static_mode else 'dynamic',
        'avatar_can_dynamic': _avatar_supports_dynamic(),
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    })


def _register_r_alias(rule: str, view_func, methods=None):
    api_bp.add_url_rule(rule, view_func=view_func, methods=methods)


_register_r_alias('/r/api/health', health)


@api_bp.route('/api/generate', methods=['POST'])
def generate_avatar_speech():
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

        supports_dynamic = _avatar_supports_dynamic()
        static_mode = state.avatar_static_mode
        if 'static_mode' in data:
            try:
                requested_static = _coerce_optional_bool(data.get('static_mode'))
            except ValueError as bool_err:
                return jsonify({'error': str(bool_err)}), 400

            if requested_static is False and not supports_dynamic:
                print("⚠️ Запрошен динамический режим, но текущий аватар не является видео. Используется статичный режим.")
                static_mode = True
            else:
                static_mode = requested_static

        print("\n" + "=" * 60)
        print("🎬 Новый запрос генерации")
        print("=" * 60)
        print(f"Текст: {text}")
        print(f"Язык: {language}")
        print(f"Режим аватара: {'статичный' if static_mode else 'динамический'}")

        start_total = time.time()

        start = time.time()
        audio_data = generate_tts(text, language)
        tts_time = time.time() - start

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        audio_path = os.path.join(TEMP_DIR, f'audio_{timestamp}.wav')
        start = time.time()
        audio_waveform, audio_sample_rate = convert_to_wav(audio_data, audio_path)
        convert_time = time.time() - start

        output_path = os.path.join(OUTPUT_DIR, f'avatar_{timestamp}.mp4')
        supports_dynamic = _avatar_supports_dynamic()
        static_mode = state.avatar_static_mode
        if 'static_mode' in data:
            try:
                requested_static = _coerce_optional_bool(data.get('static_mode'))
            except ValueError as bool_err:
                return jsonify({'error': str(bool_err)}), 400

            if requested_static is False and not supports_dynamic:
                print("⚠️ Запрошен динамический режим, но текущий аватар не является видео. Используется статичный режим.")
                static_mode = True
            else:
                static_mode = requested_static

        service = state.lipsync_service_gan or state.lipsync_service_nogan
        if service is None:
            return jsonify({'error': 'Модель lipsync не загружена'}), 503

        model_label = 'GAN' if service is state.lipsync_service_gan else 'NoGAN'
        print(f"\n🎭 Генерация lip-sync ({model_label})...")
        start = time.time()

        if static_mode:
            try:
                service.preload_static_face(
                    face_path=AVATAR_IMAGE,
                    fps=AVATAR_FPS,
                    pads=(0, 50, 0, 0),
                )
            except Exception as preload_error:
                print(f"⚠️ Не удалось обновить кэш статического лица ({preload_error})")
        else:
            try:
                service.preload_video_cache(
                    face_path=AVATAR_IMAGE,
                    fps=AVATAR_FPS,
                    pads=(0, 50, 0, 0)
                )
            except Exception as preload_error:
                print(f"⚠️ Не удалось подготовить детекцию для видео ({preload_error})")

        stats = service.process(
            face_path=AVATAR_IMAGE,
            audio_path=audio_path,
            output_path=output_path,
            static=static_mode,
            pads=(0, 50, 0, 0),
            fps=AVATAR_FPS,
            audio_waveform=audio_waveform,
            audio_sample_rate=audio_sample_rate
        )

        lipsync_time = time.time() - start
        total_time = time.time() - start_total

        if os.path.exists(audio_path):
            os.remove(audio_path)

        print("\n📊 Статистика:")
        print(f"   TTS генерация:    {tts_time:.2f}s")
        print(f"   Конвертация:      {convert_time:.2f}s")
        print(f"   Lip-sync:         {lipsync_time:.2f}s")
        print(f"     - Загрузка видео:   {stats['load_video_time']:.2f}s")
        print(f"     - Обработка аудио:  {stats['process_audio_time']:.2f}s")
        print(f"     - Детекция лица:    {stats['face_detection_time']:.2f}s")
        print(f"     - Инференс модели:  {stats['inference_time']:.2f}s")
        print(f"   Режим аватара:    {'статичный' if static_mode else 'динамический'}")
        print("   Постобработка:")
        print(f"     - Сегментация:      {'✅' if stats.get('segmentation') else '❌'}")
        print(f"     - ESRGAN:           {'✅' if stats.get('super_resolution') else '❌'}")
        print(f"     - Real-ESRGAN:      {'✅' if stats.get('real_esrgan') else '❌'}")
        print("   ─────────────────────────")
        print(f"   ИТОГО:            {total_time:.2f}s")
        print(f"\n✅ Видео готово: {output_path}")
        print("=" * 60 + "\n")

        return send_file(
            output_path,
            mimetype='video/mp4',
            as_attachment=True,
            download_name='avatar_speech.mp4'
        )

    except Exception as e:  # pragma: no cover - runtime logging
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


_register_r_alias('/r/api/generate', generate_avatar_speech, methods=['POST'])


@api_bp.route('/api/avatar')
def get_avatar():
    avatar_path = AVATAR_IMAGE if Path(AVATAR_IMAGE).suffix.lower() in _IMAGE_EXTENSIONS else AVATAR_PREVIEW_PATH

    if avatar_path and os.path.exists(avatar_path):
        return send_file(avatar_path, mimetype='image/jpeg')

    return jsonify({'error': 'Аватар недоступен'}), 404


_register_r_alias('/r/api/avatar', get_avatar)




@api_bp.route('/api/stream_chunks', methods=['POST'])
def stream_chunks():
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

        service = state.lipsync_service_gan or state.lipsync_service_nogan
        if service is None:
            return jsonify({'error': 'Модель lipsync не загружена'}), 503

        model_label = 'GAN' if service is state.lipsync_service_gan else 'NoGAN'

        print(f"\n🎬 API Stream (single clip): {len(text)} символов, язык: {language}")
        print(f"🛠️ Используется модель: {model_label}")
        print(f"🎯 Режим аватара: {'статичный' if static_mode else 'динамический'}")

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        clip_id = f"clip_{timestamp}"

        print("🎤 Генерация полного TTS...")
        audio_data = generate_tts(text, language)

        temp_audio_path = os.path.join(TEMP_DIR, f'audio_{clip_id}.wav')
        audio_waveform, audio_sample_rate = convert_to_wav(audio_data, temp_audio_path)

        if static_mode:
            try:
                service.preload_static_face(
                    face_path=AVATAR_IMAGE,
                    fps=AVATAR_FPS,
                    pads=(0, 50, 0, 0)
                )
            except Exception as preload_error:
                print(f"⚠️ Не удалось обновить кэш статического лица ({preload_error})")
        else:
            try:
                service.preload_video_cache(
                    face_path=AVATAR_IMAGE,
                    fps=AVATAR_FPS,
                    pads=(0, 50, 0, 0)
                )
            except Exception as preload_error:
                print(f"⚠️ Не удалось подготовить детекцию для видео ({preload_error})")

        video_path = os.path.join(OUTPUT_DIR, f'{clip_id}.mp4')
        stats = service.process(
            face_path=AVATAR_IMAGE,
            audio_path=temp_audio_path,
            output_path=video_path,
            static=static_mode,
            pads=(0, 50, 0, 0),
            fps=AVATAR_FPS,
            audio_waveform=audio_waveform,
            audio_sample_rate=audio_sample_rate
        )

        duration = round(audio_waveform.shape[1] / audio_sample_rate, 2)

        audio_output_path = os.path.join(OUTPUT_DIR, f'{clip_id}.wav')
        shutil.copy(temp_audio_path, audio_output_path)
        os.remove(temp_audio_path)

        print("\n📊 Статистика (single clip):")
        print(f"   Детекция лица:       {stats['face_detection_time']:.2f}s")
        print(f"   Режим аватара:       {'статичный' if static_mode else 'динамический'}")
        print(f"   Всего времени: {stats['total_time']:.2f}s")

        return jsonify({
            'success': True,
            'total_chunks': 1,
            'chunks': [{
                'index': 0,
                'text': text,
                'video_url': f'/api/chunk/video/{clip_id}',
                'audio_url': f'/api/chunk/audio/{clip_id}',
                'duration': duration
            }],
            'language': language,
            'mode': 'static' if static_mode else 'dynamic'
        })

    except Exception as e:
        print(f"\n❌ Ошибка API: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


_register_r_alias('/r/api/stream_chunks', stream_chunks, methods=['POST'])


@api_bp.route('/api/chunk/video/<chunk_id>')
def get_chunk_video(chunk_id):
    try:
        candidates = [
            os.path.join(OUTPUT_DIR, f'chunk_video_{chunk_id}.mp4'),
            os.path.join(OUTPUT_DIR, f'{chunk_id}.mp4'),
        ]
        video_path = next((path for path in candidates if os.path.exists(path)), None)
        if video_path is None:
            return jsonify({'error': 'Видео не найдено'}), 404

        return send_file(
            video_path,
            mimetype='video/mp4',
            as_attachment=False
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500


_register_r_alias('/r/api/chunk/video/<chunk_id>', get_chunk_video)


@api_bp.route('/api/chunk/audio/<chunk_id>')
def get_chunk_audio(chunk_id):
    try:
        candidates = [
            os.path.join(OUTPUT_DIR, f'chunk_audio_{chunk_id}.wav'),
            os.path.join(OUTPUT_DIR, f'{chunk_id}.wav'),
        ]
        audio_path = next((path for path in candidates if os.path.exists(path)), None)
        if audio_path is None:
            return jsonify({'error': 'Аудио не найдено'}), 404

        return send_file(
            audio_path,
            mimetype='audio/wav',
            as_attachment=False
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500


_register_r_alias('/r/api/chunk/audio/<chunk_id>', get_chunk_audio)


@api_bp.route('/api/cleanup', methods=['POST'])
def cleanup():
    try:
        now = time.time()
        removed = 0

        for filename in os.listdir(OUTPUT_DIR):
            filepath = os.path.join(OUTPUT_DIR, filename)
            if os.path.isfile(filepath) and now - os.path.getmtime(filepath) > 3600:
                os.remove(filepath)
                removed += 1

        return jsonify({'message': f'Удалено файлов: {removed}'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


_register_r_alias('/r/api/cleanup', cleanup, methods=['POST'])
_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
_VIDEO_EXTENSIONS = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v'}


def _avatar_supports_dynamic() -> bool:
    return Path(AVATAR_IMAGE).suffix.lower() in _VIDEO_EXTENSIONS


def _coerce_optional_bool(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    raise ValueError("Не удалось разобрать значение static_mode")
