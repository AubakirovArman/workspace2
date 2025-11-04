"""Chunk retrieval endpoints."""
from __future__ import annotations

import os

from flask import jsonify, send_file

from ...config import OUTPUT_DIR
from . import api_bp, register_route


@api_bp.route("/api/chunk/video/<chunk_id>")
def get_chunk_video(chunk_id):
    try:
        candidates = [
            os.path.join(OUTPUT_DIR, f"chunk_video_{chunk_id}.mp4"),
            os.path.join(OUTPUT_DIR, f"{chunk_id}.mp4"),
        ]
        video_path = next((path for path in candidates if os.path.exists(path)), None)
        if video_path is None:
            return jsonify({"error": "Видео не найдено"}), 404

        return send_file(video_path, mimetype="video/mp4", as_attachment=False)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


register_route("/r/api/chunk/video/<chunk_id>", get_chunk_video)


@api_bp.route("/api/chunk/audio/<chunk_id>")
def get_chunk_audio(chunk_id):
    try:
        candidates = [
            os.path.join(OUTPUT_DIR, f"chunk_audio_{chunk_id}.wav"),
            os.path.join(OUTPUT_DIR, f"{chunk_id}.wav"),
        ]
        audio_path = next((path for path in candidates if os.path.exists(path)), None)
        if audio_path is None:
            return jsonify({"error": "Аудио не найдено"}), 404

        return send_file(audio_path, mimetype="audio/wav", as_attachment=False)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


register_route("/r/api/chunk/audio/<chunk_id>", get_chunk_audio)
