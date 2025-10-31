"""Flask application factory."""
from __future__ import annotations

from flask import Flask
from flask_cors import CORS

from .config import BASE_DIR
from .logging_setup import configure_logging, register_request_logging
from .routes import register_routes


def create_app() -> Flask:
    configure_logging()

    app = Flask(
        __name__,
        template_folder=str(BASE_DIR / 'templates'),
        static_folder=str(BASE_DIR / 'static')
    )
    CORS(app)

    register_request_logging(app)
    register_routes(app)

    return app
