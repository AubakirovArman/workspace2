"""Primary avatar generation endpoint."""
from __future__ import annotations

import os
import time
from datetime import datetime

from flask import jsonify, request, send_file

from ... import state
from ...config import AVATAR_FPS, AVATAR_IMAGE, OUTPUT_DIR
from ...services import convert_to_wav, generate_tts, estimate_optimal_chunks
from . import api_bp, register_route
from .helpers import (
    avatar_supports_dynamic,
    coerce_optional_bool,
    encode_video_with_audio,
    generate_frames_parallel,
    generate_frames_single,
    DEFAULT_BATCH_SIZE,
)


@api_bp.route("/api/generate", methods=["POST"])
def generate_avatar_speech():
    try:
        data = request.get_json()
        if not data or "text" not in data:
            return jsonify({"error": "Требуется поле \"text\""}), 400

        text = data["text"].strip()
        language = data.get("language", "ru")

        if not text:
            return jsonify({"error": "Текст не может быть пустым"}), 400
        if language not in ["ru", "kk", "en"]:
            return jsonify({"error": "Неподдерживаемый язык"}), 400

        supports_dynamic = avatar_supports_dynamic()
        static_mode = state.avatar_static_mode
        if "static_mode" in data:
            try:
                requested_static = coerce_optional_bool(data.get("static_mode"))
            except ValueError as bool_err:
                return jsonify({"error": str(bool_err)}), 400

            if requested_static is False and not supports_dynamic:
                print(
                    "⚠️ Запрошен динамический режим, но текущий аватар не является видео. Используется статичный режим."
                )
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

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        start = time.time()
        audio_waveform, audio_sample_rate = convert_to_wav(audio_data, None)
        convert_time = time.time() - start

        output_path = os.path.join(OUTPUT_DIR, f"avatar_{timestamp}.mp4")

        gan_services = [svc for svc in state.get_all_gan_services() if svc]
        service_pool = list(gan_services)
        if not service_pool and state.lipsync_service_nogan is not None:
            service_pool = [state.lipsync_service_nogan]

        if not service_pool:
            return jsonify({"error": "Модель lipsync не загружена"}), 503

        primary_service = service_pool[0]

        audio_duration_seconds = audio_waveform.shape[1] / float(audio_sample_rate)
        base_segments = estimate_optimal_chunks(audio_duration_seconds, max(1, len(service_pool)))
        desired_segments = max(len(service_pool), base_segments) if base_segments > 1 else 1
        use_parallel = len(service_pool) > 1 and desired_segments > 1

        if use_parallel:
            device_labels = ", ".join(str(getattr(svc, "device", "cuda")) for svc in service_pool)
            print(f"🧠 Параллельная генерация на {len(service_pool)} GPU: {device_labels}")
        else:
            model_label = "GAN" if primary_service is state.lipsync_service_gan else "NoGAN"
            print(f"🎭 Генерация lip-sync ({model_label}, единичный поток)...")

        prepare_start = time.time()

        for index, svc in enumerate(service_pool, start=1):
            try:
                if static_mode:
                    svc.preload_static_face(
                        face_path=AVATAR_IMAGE,
                        fps=AVATAR_FPS,
                        pads=(0, 50, 0, 0),
                    )
                else:
                    svc.preload_video_cache(
                        face_path=AVATAR_IMAGE,
                        fps=AVATAR_FPS,
                        pads=(0, 50, 0, 0),
                    )
            except Exception as preload_error:
                suffix = f" #{index}" if len(service_pool) > 1 else ""
                print(f"⚠️ Не удалось подготовить аватар{suffix}: {preload_error}")

        prepare_time = time.time() - prepare_start

        frames_start = time.time()

        if use_parallel:
            frames, stats, active_chunks = generate_frames_parallel(
                service_pool=service_pool,
                static_mode=static_mode,
                audio_waveform=audio_waveform,
                audio_sample_rate=audio_sample_rate,
                desired_chunks=desired_segments,
                batch_size=DEFAULT_BATCH_SIZE,
            )
            print(
                f"✅ Параллельная генерация завершена: {len(frames)} кадров, {active_chunks} чанков (запрошено {desired_segments})"
            )
        else:
            frames, stats = generate_frames_single(
                service=primary_service,
                static_mode=static_mode,
                audio_waveform=audio_waveform,
                audio_sample_rate=audio_sample_rate,
                batch_size=DEFAULT_BATCH_SIZE,
            )

        frames_time = time.time() - frames_start

        encode_start = time.time()
        encode_video_with_audio(
            frames=frames,
            output_path=output_path,
            audio_waveform=audio_waveform,
            audio_sample_rate=audio_sample_rate,
            fps=AVATAR_FPS,
            codec_service=primary_service,
            segments=desired_segments,
        )

        encode_time = time.time() - encode_start

        lipsync_time = prepare_time + frames_time
        pipeline_time = lipsync_time + encode_time
        total_time = time.time() - start_total

        print("\n📊 Статистика:")
        print(f"   TTS генерация:    {tts_time:.2f}s")
        print(f"   Конвертация:      {convert_time:.2f}s")
        print(f"   Lip-sync (prep):  {prepare_time:.2f}s")
        print(f"   Lip-sync (frames): {frames_time:.2f}s")
        print(f"   Lip-sync сумма:   {lipsync_time:.2f}s")
        print(f"   Кодирование:      {encode_time:.2f}s")
        print(f"   Пайплайн (видео): {pipeline_time:.2f}s")
        if use_parallel:
            print(f"   Активные GPU:     {len(service_pool)}")
            print(f"   Чанки обработки:  {active_chunks}")
        else:
            print(f"     - Загрузка видео:   {stats.get('load_video_time', 0):.2f}s")
            print(f"     - Обработка аудио:  {stats.get('process_audio_time', 0):.2f}s")
            print(f"     - Детекция лица:    {stats.get('face_detection_time', 0):.2f}s")
            print(f"     - Инференс модели:  {stats.get('inference_time', 0):.2f}s")
        print(f"   Режим аватара:    {'статичный' if static_mode else 'динамический'}")
        print("   ─────────────────────────")
        print(f"   ИТОГО:            {total_time:.2f}s")
        print(f"\n✅ Видео готово: {output_path}")
        print("=" * 60 + "\n")

        return send_file(
            output_path,
            mimetype="video/mp4",
            as_attachment=True,
            download_name="avatar_speech.mp4",
        )

    except Exception as exc:
        print(f"\n❌ Ошибка: {exc}")
        import traceback

        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


register_route("/r/api/generate", generate_avatar_speech, methods=["POST"])
