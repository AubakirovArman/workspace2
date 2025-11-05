"""Expose generated HLS segments and playlists."""
from __future__ import annotations

from pathlib import Path

from flask import Blueprint, abort, send_from_directory

from ..config import TEMP_DIR

stream_files_bp = Blueprint("stream_files", __name__)

STREAM_ROOT = Path(TEMP_DIR) / "hls"


@stream_files_bp.route("/stream/<session_id>/<path:filename>")
def serve_stream_file(session_id: str, filename: str):
    session_dir = STREAM_ROOT / session_id
    if not session_dir.exists():
        abort(404)
    return send_from_directory(
        session_dir,
        filename,
        conditional=True,
        cache_timeout=0,
    )

