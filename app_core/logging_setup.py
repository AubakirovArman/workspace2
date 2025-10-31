"""Logging configuration and request tracing middleware."""
from __future__ import annotations

import logging
from flask import request

LOGGER = logging.getLogger("avatar_lipsync")
_CONFIGURED = False


def configure_logging() -> logging.Logger:
    global _CONFIGURED
    if not _CONFIGURED:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        _CONFIGURED = True
    return LOGGER


def register_request_logging(app) -> None:
    if getattr(app, "_request_logging_registered", False):
        return

    logger = LOGGER

    @app.before_request
    def log_request_info():  # type: ignore[unused-ignore]
        logger.info('=' * 80)
        logger.info('📨 Входящий запрос:')
        logger.info('   Метод: %s', request.method)
        logger.info('   URL: %s', request.url)
        logger.info('   Path: %s', request.path)
        logger.info('   Remote IP: %s', request.remote_addr)
        logger.info('   Headers:')
        for header, value in request.headers:
            logger.info('      %s: %s', header, value)
        if request.method in ['POST', 'PUT', 'PATCH']:
            logger.info('   Body: %s...', request.get_data(as_text=True)[:500])
        logger.info('=' * 80)

    @app.after_request
    def log_response_info(response):  # type: ignore[unused-ignore]
        logger.info('📤 Ответ: %s', response.status_code)
        return response

    app._request_logging_registered = True
