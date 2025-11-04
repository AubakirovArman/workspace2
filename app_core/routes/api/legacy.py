"""Legacy parallel generation endpoint maintained for compatibility."""
from __future__ import annotations

import os
import time
from datetime import datetime

from flask import jsonify, request, send_file

from ... import state
from ...config import AVATAR_FPS, TEMP_DIR, OUTPUT_DIR
from ...services import (
    convert_to_wav,
    estimate_optimal_chunks,
    generate_tts,
    parallel_lipsync_process,
)
from . import api_bp, register_route


@api_bp.route("/api/generate_parallel", methods=["POST"])
def generate_avatar_speech_parallel():
    """Explicit multi-worker generation pipeline."""
    try:
        data = request.get_json()
        if not data or "text" not in data:
            return jsonify({"error": "Требуется поле \"text\""}), 400

        text = data["text"].strip()
        language = data.get("language", "ru")
        num_workers = int(data.get("num_workers", 3))
        use_only_gan = data.get("use_only_gan", True)

        if not text:
            return jsonify({"error": "Текст не может быть пустым"}), 400
        if language not in ["ru", "kk", "en"]:
            return jsonify({"error": "Неподдерживаемый язык"}), 400

        gan_services = state.get_all_gan_services(include_none=True)

        if not gan_services or gan_services[0] is None:
            return jsonify({
                "error": "Для параллельной обработки требуется хотя бы одна GAN модель",
                "available_models": {
                    "gan": bool(gan_services and gan_services[0]),
                    "gan2": bool(len(gan_services) > 1 and gan_services[1]),
                    "gan3": bool(len(gan_services) > 2 and gan_services[2]),
                    "gan4": bool(len(gan_services) > 3 and gan_services[3]),
                    "gan5": bool(len(gan_services) > 4 and gan_services[4]),
                    "gan6": bool(len(gan_services) > 5 and gan_services[5]),
                    "gan7": bool(len(gan_services) > 6 and gan_services[6]),
                    "gan8": bool(len(gan_services) > 7 and gan_services[7]),
                    "nogan": state.lipsync_service_nogan is not None,
                },
            }), 503

        available_gan_services = [svc for svc in gan_services if svc is not None]
        available_models = len(available_gan_services)
        if not use_only_gan and state.lipsync_service_nogan:
            available_models += 1

        print("\n" + "=" * 60)
        print(f"🚀 ПАРАЛЛЕЛЬНАЯ ГЕНЕРАЦИЯ ({available_models} модели)")
        print("=" * 60)
        print(f"Режим: {'только GAN' if use_only_gan else 'GAN + NOGAN'}")
        print(f"Текст: {text}")
        print(f"Язык: {language}")
        print(f"Воркеры: {num_workers}")

        start_total = time.time()

        start = time.time()
        audio_data = generate_tts(text, language)
        tts_time = time.time() - start

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        audio_path = os.path.join(TEMP_DIR, f"audio_parallel_{timestamp}.wav")
        start = time.time()
        audio_waveform, audio_sample_rate = convert_to_wav(audio_data, audio_path)
        convert_time = time.time() - start

        audio_duration = audio_waveform.shape[-1] / audio_sample_rate
        print(f"📊 Длительность аудио: {audio_duration:.2f}s")

        optimal_chunks = estimate_optimal_chunks(audio_duration, num_models=available_models)
        print(f"📦 Оптимальное разбиение: {optimal_chunks} чанков")

        output_path = os.path.join(OUTPUT_DIR, f"avatar_parallel_{timestamp}.mp4")

        parallel_stats = parallel_lipsync_process(
            gan_service=state.lipsync_service_gan,
            nogan_service=state.lipsync_service_nogan,
            audio_path=audio_path,
            output_path=output_path,
            num_workers=optimal_chunks,
            fps=AVATAR_FPS,
            use_cached=True,
            gan_extra_services=available_gan_services[1:],
            use_only_gan=use_only_gan,
        )

        total_time = time.time() - start_total

        if os.path.exists(audio_path):
            os.remove(audio_path)

        print("\n📊 Статистика параллельной обработки:")
        print(f"   TTS генерация:    {tts_time:.2f}s")
        print(f"   Конвертация:      {convert_time:.2f}s")
        print(f"   Разбиение аудио:  {parallel_stats['split_time']:.2f}s")
        print(f"   Параллельная обработка: {parallel_stats['process_time']:.2f}s")
        print(f"   Склейка видео:    {parallel_stats['merge_time']:.2f}s")
        print(f"   Количество чанков: {parallel_stats['num_chunks']}")
        print("   ─────────────────────────")
        print(f"   ИТОГО:            {total_time:.2f}s")
        print(f"   Ускорение:        {parallel_stats['speedup']}")
        print(f"\n✅ Видео готово: {output_path}")
        print("=" * 60 + "\n")

        return send_file(
            output_path,
            mimetype="video/mp4",
            as_attachment=True,
            download_name="avatar_speech_parallel.mp4",
        )

    except Exception as exc:
        print(f"\n❌ Ошибка параллельной обработки: {exc}")
        import traceback

        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


register_route("/r/api/generate_parallel", generate_avatar_speech_parallel, methods=["POST"])
