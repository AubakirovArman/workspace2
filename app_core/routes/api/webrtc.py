"""WebRTC signalling endpoint."""
from __future__ import annotations

from flask import jsonify, request, send_file

from ... import state
from ...services import start_webrtc_session, get_stream_session
from . import api_bp, register_route


@api_bp.route("/api/webrtc/start", methods=["POST"])
def webrtc_start():
    try:
        data = request.get_json() or {}
        sdp = data.get("sdp")
        offer_type = data.get("type", "offer")
        text = (data.get("text") or "").strip()
        language = data.get("language", "ru")
        low_latency = bool(data.get("low_latency", False))
        mode = (data.get("mode") or "").strip().lower()
        dynamic_mode = mode == "dynamic" or bool(data.get("dynamic", False))

        fps_value = data.get("fps", 25.0)
        try:
            fps = float(fps_value)
        except (TypeError, ValueError):
            return jsonify({"error": "Некорректное значение fps"}), 400

        batch_size = data.get("batch_size")
        if batch_size is not None:
            try:
                batch_size = int(batch_size)
                if batch_size <= 0:
                    raise ValueError
            except (TypeError, ValueError):
                return jsonify({"error": "Некорректное значение batch_size"}), 400
        base_video_path = data.get("base_video_path")

        if not sdp or not offer_type:
            return jsonify({"error": "Недействительный SDP или тип"}), 400
        if not text:
            return jsonify({"error": "Требуется непустой текст"}), 400
        if language not in ["ru", "kk", "en"]:
            return jsonify({"error": "Неподдерживаемый язык"}), 400

        service = state.lipsync_service_gan or state.lipsync_service_nogan
        if service is None:
            return jsonify({"error": "Модель lipsync не загружена"}), 503

        answer, session_id = start_webrtc_session(
            sdp=sdp,
            offer_type=offer_type,
            text=text,
            language=language,
            low_latency=low_latency,
            dynamic_mode=dynamic_mode,
            fps=fps,
            batch_size=batch_size,
            base_video_path=base_video_path,
        )

        return jsonify({"sdp": answer.sdp, "type": answer.type, "session_id": session_id})
    except Exception as exc:
        print(f"\n❌ WebRTC error: {exc}")
        import traceback

        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


register_route("/r/api/webrtc/start", webrtc_start, methods=["POST"])


@api_bp.route("/api/webrtc/result/<session_id>", methods=["GET"])
def webrtc_result(session_id: str):
    session = get_stream_session(session_id)
    if session is None:
        return jsonify({"error": "Сессия не найдена"}), 404
    if session.error:
        return jsonify({"error": session.error}), 500
    if not session.result_ready or not session.output_video_path.exists():
        return jsonify({"status": "processing"}), 202
    return send_file(
        str(session.output_video_path),
        mimetype="video/mp4",
        as_attachment=False,
        conditional=False,
    )


register_route("/r/api/webrtc/result/<session_id>", webrtc_result, methods=["GET"])
