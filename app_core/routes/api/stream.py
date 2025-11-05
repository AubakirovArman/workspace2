"""API endpoints for low-latency HLS streaming."""
from __future__ import annotations

from flask import jsonify, request

from ...services import get_stream_job, start_stream_job
from . import api_bp, register_route


@api_bp.route("/api/stream/start", methods=["POST"])
def stream_start():
    payload = request.get_json(force=True, silent=True) or {}
    text = (payload.get("text") or "").strip()
    language = (payload.get("language") or "ru").strip()
    mode = (payload.get("mode") or "dynamic").strip().lower()

    if not text:
        return jsonify({"error": "Введите текст для генерации"}), 400
    if language not in {"ru", "en", "kk"}:
        return jsonify({"error": "Неподдерживаемый язык"}), 400
    if mode != "dynamic":
        return jsonify({"error": "HLS поток поддерживает только динамический режим"}), 400

    try:
        job = start_stream_job(text=text, language=language, dynamic=True)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 500

    return jsonify(
        {
            "session_id": job.session_id,
            "playlist_url": job.playlist_url(),
            "status": job.status,
        }
    )


@api_bp.route("/api/stream/status/<session_id>", methods=["GET"])
def stream_status(session_id: str):
    job = get_stream_job(session_id)
    if job is None:
        return jsonify({"error": "Сессия не найдена"}), 404

    response = {
        "session_id": job.session_id,
        "status": job.status,
        "error": job.error,
        "playlist_url": job.playlist_url(),
        "mp4_url": job.mp4_url(),
        "created_at": job.created_at,
        "finished_at": job.finished_at,
    }
    if job.status == "ready" and job.summary:
        response["summary"] = job.summary
    return jsonify(response)


register_route("/r/api/stream/start", stream_start, methods=["POST"])
register_route("/r/api/stream/status/<session_id>", stream_status, methods=["GET"])

