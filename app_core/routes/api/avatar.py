"""Avatar preview endpoints."""
from __future__ import annotations

import os
from pathlib import Path

from flask import jsonify, send_file

from ...config import AVATAR_IMAGE, AVATAR_PREVIEW_PATH
from . import api_bp, register_route
from .helpers import IMAGE_EXTENSIONS


@api_bp.route("/api/avatar")
def get_avatar():
    candidate = Path(AVATAR_IMAGE)
    if candidate.suffix.lower() not in IMAGE_EXTENSIONS:
        avatar_path = AVATAR_PREVIEW_PATH
    else:
        avatar_path = AVATAR_IMAGE

    if avatar_path and os.path.exists(avatar_path):
        return send_file(avatar_path, mimetype="image/jpeg")

    return jsonify({"error": "Аватар недоступен"}), 404


register_route("/r/api/avatar", get_avatar)
