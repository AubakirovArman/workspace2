"""Legacy streaming endpoints."""
from __future__ import annotations

import os
from datetime import datetime

import torchaudio
from flask import jsonify, request

from ... import state
from ...config import AVATAR_FPS, AVATAR_IMAGE, OUTPUT_DIR
from ...services import convert_to_wav, generate_tts
from . import api_bp, register_route
from .helpers import avatar_supports_dynamic, coerce_optional_bool, write_waveform_to_wav


@api_bp.route("/api/stream_chunks", methods=["POST"])
def stream_chunks():
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

        service = state.lipsync_service_gan or state.lipsync_service_nogan
        if service is None:
            return jsonify({"error": "Модель lipsync не загружена"}), 503

        model_label = "GAN" if service is state.lipsync_service_gan else "NoGAN"

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

        print(f"\n🎬 API Stream (single clip): {len(text)} символов, язык: {language}")
        print(f"🛠️ Используется модель: {model_label}")
        print(f"🎯 Режим аватара: {'статичный' if static_mode else 'динамический'}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        clip_id = f"clip_{timestamp}"

        print("🎤 Генерация полного TTS...")
        audio_data = generate_tts(text, language)

        audio_waveform, audio_sample_rate = convert_to_wav(audio_data, None)

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
                    pads=(0, 50, 0, 0),
                )
            except Exception as preload_error:
                print(f"⚠️ Не удалось подготовить детекцию для видео ({preload_error})")

        video_path = os.path.join(OUTPUT_DIR, f"{clip_id}.mp4")
        stats = service.process(
            face_path=AVATAR_IMAGE,
            audio_path="",
            output_path=video_path,
            static=static_mode,
            pads=(0, 50, 0, 0),
            fps=AVATAR_FPS,
            audio_waveform=audio_waveform,
            audio_sample_rate=audio_sample_rate,
        )

        duration = round(audio_waveform.shape[1] / audio_sample_rate, 2)

        audio_output_path = os.path.join(OUTPUT_DIR, f"{clip_id}.wav")
        try:
            torchaudio.save(audio_output_path, audio_waveform.cpu(), audio_sample_rate, backend="sox_io")
        except (TypeError, RuntimeError):
            write_waveform_to_wav(audio_output_path, audio_waveform, audio_sample_rate)

        print("\n📊 Статистика (single clip):")
        print(f"   Детекция лица:       {stats['face_detection_time']:.2f}s")
        print(f"   Режим аватара:       {'статичный' if static_mode else 'динамический'}")
        print(f"   Всего времени: {stats['total_time']:.2f}s")

        return jsonify({
            "success": True,
            "total_chunks": 1,
            "chunks": [
                {
                    "index": 0,
                    "text": text,
                    "video_url": f"/api/chunk/video/{clip_id}",
                    "audio_url": f"/api/chunk/audio/{clip_id}",
                    "duration": duration,
                }
            ],
            "language": language,
            "mode": "static" if static_mode else "dynamic",
        })

    except Exception as exc:
        print(f"\n❌ Ошибка API: {exc}")
        import traceback

        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


register_route("/r/api/stream_chunks", stream_chunks, methods=["POST"])
