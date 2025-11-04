"""Health endpoint exposing model readiness."""
from __future__ import annotations

import torch
from flask import jsonify

from ... import state
from . import api_bp, register_route
from .helpers import avatar_supports_dynamic


def _collect_service_features(service):
    if service is None:
        return None
    return {
        "segmentation": bool(
            getattr(service, "segmentation_enabled", False)
            and getattr(service, "segmentation_model", None) is not None
        ),
        "super_resolution": bool(
            getattr(service, "sr_enabled", False)
            and getattr(service, "sr_model", None) is not None
        ),
        "real_esrgan": bool(
            getattr(service, "realesrgan_enabled", False)
            and getattr(service, "_realesrgan_enhancer", None) is not None
        ),
    }


@api_bp.route("/api/health")
def health():
    gan_slots = state.get_all_gan_services(include_none=True)
    slot_ready = [svc is not None for svc in gan_slots]
    nogan_ready = state.lipsync_service_nogan is not None

    primary_ready = slot_ready[0] if slot_ready else False
    total_gan_models = sum(slot_ready)
    status = "ready" if primary_ready else ("degraded" if nogan_ready else "offline")
    parallel_models = total_gan_models + (1 if nogan_ready else 0)

    def _slot(index: int):
        return gan_slots[index] if index < len(gan_slots) else None

    def _flag(index: int) -> bool:
        return slot_ready[index] if index < len(slot_ready) else False

    models_payload = {
        name: _collect_service_features(_slot(idx))
        for idx, name in enumerate(["gan", "gan2", "gan3", "gan4", "gan5", "gan6", "gan7", "gan8"])
    }
    models_payload["nogan"] = _collect_service_features(state.lipsync_service_nogan)

    return jsonify({
        "status": status,
        "hd_model_loaded": primary_ready,
        "gan_model_loaded": _flag(0),
        "gan2_model_loaded": _flag(1),
        "gan3_model_loaded": _flag(2),
        "gan4_model_loaded": _flag(3),
        "gan5_model_loaded": _flag(4),
        "gan6_model_loaded": _flag(5),
        "gan7_model_loaded": _flag(6),
        "gan8_model_loaded": _flag(7),
        "nogan_model_loaded": nogan_ready,
        "models_loaded": parallel_models > 0,
        "parallel_models_count": parallel_models,
        "parallel_gan_count": total_gan_models,
        "models": models_payload,
        "avatar_loaded": state.avatar_preloaded is not None,
        "avatar_mode": "static" if state.avatar_static_mode else "dynamic",
        "avatar_can_dynamic": avatar_supports_dynamic(),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    })


register_route("/r/api/health", health)
