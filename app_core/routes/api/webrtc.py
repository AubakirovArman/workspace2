"""WebRTC signalling endpoint."""
from __future__ import annotations

from flask import jsonify, request

from ... import state
from ...services import start_webrtc_session
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

        if not sdp or not offer_type:
            return jsonify({"error": "Недействительный SDP или тип"}), 400
        if not text:
            return jsonify({"error": "Требуется непустой текст"}), 400
        if language not in ["ru", "kk", "en"]:
            return jsonify({"error": "Неподдерживаемый язык"}), 400

        service = state.lipsync_service_gan or state.lipsync_service_nogan
        if service is None:
            return jsonify({"error": "Модель lipsync не загружена"}), 503

        answer = start_webrtc_session(
            sdp=sdp,
            offer_type=offer_type,
            text=text,
            language=language,
            low_latency=low_latency,
        )

        return jsonify({"sdp": answer.sdp, "type": answer.type})
    except Exception as exc:
        print(f"\n❌ WebRTC error: {exc}")
        import traceback

        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


register_route("/r/api/webrtc/start", webrtc_start, methods=["POST"])
