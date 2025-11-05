"""Blueprint registration helpers."""
from __future__ import annotations

from flask import Flask

from .api import api_bp
from .pages import pages_bp
from .stream_files import stream_files_bp


def register_routes(app: Flask) -> None:
    app.register_blueprint(pages_bp)
    app.register_blueprint(api_bp)
    app.register_blueprint(stream_files_bp)
