"""Modular JSON API blueprint."""
from __future__ import annotations

from flask import Blueprint

api_bp = Blueprint("api", __name__)


def register_route(rule: str, view_func, methods=None) -> None:
    """Expose alias-friendly registration helper."""
    api_bp.add_url_rule(rule, view_func=view_func, methods=methods)


# Import modules so that decorators run immediately on blueprint creation.
from . import avatar  # noqa: E402,F401
from . import cleanup  # noqa: E402,F401
from . import chunk  # noqa: E402,F401
from . import generate  # noqa: E402,F401
from . import health  # noqa: E402,F401
from . import legacy  # noqa: E402,F401
from . import lipsync_video  # noqa: E402,F401
from . import segment  # noqa: E402,F401
from . import stream  # noqa: E402,F401
from . import test_page  # noqa: E402,F401

__all__ = ["api_bp", "register_route"]
