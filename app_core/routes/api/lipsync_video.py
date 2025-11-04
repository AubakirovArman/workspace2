"""Lip-sync на фиксированном видео IMG_3899.MOV (потоковая обработка без склейки)."""
from __future__ import annotations

import os
import time
from datetime import datetime

from flask import jsonify, render_template, request, send_file

from . import api_bp, register_route
from ... import state
from ...config import OUTPUT_DIR, TEMP_DIR
from ...services import convert_to_wav, generate_tts
from ...services.video_lipsync_stream import process_video_lipsync_streaming

# Фиксированное видео-база
BASE_VIDEO_PATH = '/home/arman/musetalk/avatar/IMG_3899.MOV'


@api_bp.route('/api/lipsync/video', methods=['GET', 'POST'])
def lipsync_video():
    """Lip-sync на фиксированном видео IMG_3899.MOV.
    
    GET: возвращает HTML-форму для ввода текста.
    POST: принимает text + language → TTS → lip-sync на видео-основе.
    
    Оптимизировано под H200: NVDEC decode → GPU inference → libx264 encode.
    """
    if request.method == 'GET':
        return render_template('lipsync_video.html')
    
    if not os.path.exists(BASE_VIDEO_PATH):
        return jsonify({'error': f'Видео-база не найдена: {BASE_VIDEO_PATH}'}), 404
    
    # Проверяем доступность модели
    service = state.lipsync_service_gan or state.lipsync_service_nogan
    if service is None:
        return jsonify({'error': 'Модель lipsync не загружена'}), 503
    
    try:
        # Получаем параметры
        data = request.get_json() if request.is_json else request.form
        text = data.get('text', '').strip()
        language = data.get('language', 'ru')
        
        if not text:
            return jsonify({'error': 'Требуется поле "text"'}), 400
        
        if language not in ['ru', 'kk', 'en']:
            return jsonify({'error': 'Неподдерживаемый язык'}), 400
        
        # Опциональные параметры кодирования (для H200)
        use_nvdec = bool(data.get('use_nvdec', False))
        encoder = data.get('encoder', 'libx264')
        crf = int(data.get('crf', 20))
        preset = data.get('preset', 'veryfast')
        
        print("\n" + "=" * 60)
        print("🎬 Lip-sync на видео IMG_3899.MOV")
        print("=" * 60)
        print(f"Текст: {text}")
        print(f"Язык: {language}")
        print(f"NVDEC: {'✅' if use_nvdec else '❌'}")
        print(f"Энкодер: {encoder} (crf={crf}, preset={preset})")
        
        start_total = time.time()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Генерируем TTS
        print("\n🎤 Генерация TTS...")
        start = time.time()
        audio_data = generate_tts(text, language)
        tts_time = time.time() - start
        print(f"✅ TTS: {tts_time:.2f}s")
        
        # Сохраняем аудио
        audio_path = os.path.join(TEMP_DIR, f'tts_lipsync_{timestamp}.wav')
        audio_waveform, audio_sample_rate = convert_to_wav(audio_data, audio_path)
        
        # Выходной путь
        output_path = os.path.join(OUTPUT_DIR, f'lipsync_video_{timestamp}.mp4')
        
        # Запускаем потоковую обработку
        stats = process_video_lipsync_streaming(
            base_video_path=BASE_VIDEO_PATH,
            audio_path=audio_path,
            output_path=output_path,
            lipsync_service=service,
            use_nvdec=use_nvdec,
            encoder=encoder,
            crf=crf,
            preset=preset,
            pads=(0, 10, 0, 0),
            nosmooth=False
        )
        
        # Очистка временного аудио
        if os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except OSError:
                pass
        
        total_time = time.time() - start_total
        
        print("\n📊 Статистика:")
        print(f"   TTS генерация:      {tts_time:.2f}s")
        print(f"   Загрузка видео:     {stats['load_video_time']:.2f}s")
        print(f"   Детекция лиц:       {stats['face_detection_time']:.2f}s")
        print(f"   Обработка аудио:    {stats['process_audio_time']:.2f}s")
        print(f"   Инференс:           {stats['inference_time']:.2f}s")
        print(f"   Обработано кадров:  {stats['frames_processed']}")
        print(f"   Скорость:           {stats['fps_achieved']:.1f} FPS")
        print(f"   Разрешение:         {stats['video_resolution']}")
        print(f"   Энкодер:            {stats['encoder']}")
        print(f"   NVDEC:              {'✅' if stats['use_nvdec'] else '❌'}")
        print("   ─────────────────────────")
        print(f"   ИТОГО:              {total_time:.2f}s")
        print(f"\n✅ Видео готово: {output_path}")
        print("=" * 60 + "\n")
        
        return send_file(
            output_path,
            mimetype='video/mp4',
            as_attachment=True,
            download_name='lipsync_video.mp4'
        )
    
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# Register /r/ alias
register_route('/r/api/lipsync/video', lipsync_video, methods=['GET', 'POST'])
