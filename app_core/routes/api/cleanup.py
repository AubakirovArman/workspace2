"""Maintenance endpoints for temporary files."""
from __future__ import annotations

import os
import time

from flask import jsonify

from ...config import OUTPUT_DIR
from . import api_bp, register_route


@api_bp.route("/api/cleanup", methods=["POST"])
def cleanup():
    try:
        now = time.time()
        removed = 0

        for filename in os.listdir(OUTPUT_DIR):
            filepath = os.path.join(OUTPUT_DIR, filename)
            if os.path.isfile(filepath) and now - os.path.getmtime(filepath) > 3600:
                os.remove(filepath)
                removed += 1

        return jsonify({"message": f"Удалено файлов: {removed}"})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


register_route("/r/api/cleanup", cleanup, methods=["POST"])
