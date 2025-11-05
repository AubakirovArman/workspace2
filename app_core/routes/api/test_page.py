"""API для страницы тестового пайплайна."""
from __future__ import annotations

import traceback

from flask import jsonify, request, send_file

from . import api_bp, register_route
from ...services import generate_dynamic_test_video


@api_bp.route("/api/test/generate", methods=["POST"])
def generate_test_video():
    try:
        data = request.get_json(silent=True) or {}
        text = (data.get("text") or "").strip()
        language = (data.get("language") or "ru").strip() or "ru"

        if not text:
            return jsonify({"error": "Введите текст для генерации"}), 400

        video_path, summary = generate_dynamic_test_video(text, language)
        if not video_path.exists():
            return jsonify({"error": "Видео не удалось создать"}), 500

        encode_info = summary.get("encode")
        encode_time = ""
        if isinstance(encode_info, dict):
            encode_time = encode_info.get("encode_wall_time", "")

        response = send_file(
            str(video_path),
            mimetype="video/mp4",
            as_attachment=False,
            download_name="dynamic_test.mp4",
            conditional=False,
        )
        response.headers["X-Pipeline-FPS"] = str(summary.get("fps", "25"))
        response.headers["X-Pipeline-Inference-Time"] = str(summary.get("inference_wall_time", ""))
        response.headers["X-Pipeline-Encode-Time"] = str(encode_time)
        response.headers["Cache-Control"] = "no-store"
        return response
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


register_route("/r/api/test/generate", generate_test_video, methods=["POST"])
