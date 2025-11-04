"""Segmented generation endpoints."""
from __future__ import annotations

import os

from flask import jsonify, send_file, request

from ... import state
from ...services import load_segment_metadata, run_segmented_lipsync
from . import api_bp, register_route


@api_bp.route("/api/segment/generate", methods=["POST"])
def generate_segmented_avatar():
    try:
        data = request.get_json() or {}
        text = (data.get("text") or "").strip()
        language = data.get("language", "ru")
        segments_raw = data.get("segments")
        try:
            segments = int(segments_raw) if segments_raw is not None else 0
        except (TypeError, ValueError):
            return jsonify({"error": "Некорректное значение segments"}), 400
        batch_size = int(data.get("batch_size", 1024))

        if not text:
            return jsonify({"error": "Текст не может быть пустым"}), 400
        if language not in ["ru", "kk", "en"]:
            return jsonify({"error": "Неподдерживаемый язык"}), 400
        if batch_size <= 0:
            return jsonify({"error": "Batch size должен быть больше нуля"}), 400

        print("\n" + "=" * 60)
        print("🎞️ Сегментированная генерация")
        print("=" * 60)
        print(f"Текст: {len(text)} символов")
        print(f"Язык: {language}")
        print(f"Сегменты (запрошено): {segments if segments > 0 else 'auto'}")
        print(f"Batch size override: {batch_size}")

        result = run_segmented_lipsync(
            text=text,
            language=language,
            segments=segments,
            batch_size=batch_size,
        )

        timings = result.timings.to_dict()
        payload = {
            "job_id": result.job_id,
            "video_url": f"/api/segment/video/{result.job_id}",
            "video_filename": os.path.basename(result.video_path),
            "segments": result.segments,
            "requested_segments": result.requested_segments,
            "total_frames": result.total_frames,
            "resolution": {
                "width": int(result.resolution[1]),
                "height": int(result.resolution[0]),
            },
            "timings": timings,
            "segment_results": [segment.to_dict() for segment in result.segment_results],
            "inference_stats": result.inference_stats,
            "capture_workers": result.capture_workers,
            "capture_chunks": result.capture_chunks,
            "avatar_mode": "static" if state.avatar_static_mode else "dynamic",
        }

        print("📊 Тайминги:")
        for key, value in timings.items():
            if isinstance(value, (int, float)):
                print(f"  - {key}: {value:.2f}s")
            else:
                print(f"  - {key}: {value}")
        print(f"   Сегментов использовано: {result.segments}")
        print(f"✅ Видео: {payload['video_filename']}")
        print("=" * 60 + "\n")

        return jsonify(payload)

    except ValueError as value_error:
        return jsonify({"error": str(value_error)}), 400
    except Exception as exc:
        print(f"\n❌ Ошибка сегментации: {exc}")
        import traceback

        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


@api_bp.route("/api/segment/video/<job_id>")
def get_segment_video(job_id):
    metadata = load_segment_metadata(job_id)
    if metadata is None:
        return jsonify({"error": "Результат не найден"}), 404

    video_path = metadata.get("video_path")
    if not video_path or not os.path.exists(video_path):
        return jsonify({"error": "Видео недоступно"}), 404

    return send_file(video_path, mimetype="video/mp4", as_attachment=False)


@api_bp.route("/api/segment/status/<job_id>")
def get_segment_status(job_id):
    metadata = load_segment_metadata(job_id)
    if metadata is None:
        return jsonify({"error": "Результат не найден"}), 404
    return jsonify(metadata)


register_route("/r/api/segment/generate", generate_segmented_avatar, methods=["POST"])
register_route("/r/api/segment/video/<job_id>", get_segment_video)
register_route("/r/api/segment/status/<job_id>", get_segment_status)
