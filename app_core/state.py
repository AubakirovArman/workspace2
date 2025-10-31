"""Shared application state for preloaded services and assets."""
from __future__ import annotations

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from service import LipsyncService

lipsync_service_gan: Optional["LipsyncService"] = None
lipsync_service_nogan: Optional["LipsyncService"] = None
lipsync_service_gan2: Optional["LipsyncService"] = None  # Второй экземпляр GAN
lipsync_service_gan3: Optional["LipsyncService"] = None  # Третий экземпляр GAN
avatar_preloaded = None
avatar_static_mode: bool = True


def set_state(
    gan_service: "LipsyncService",
    nogan_service: Optional["LipsyncService"],
    avatar,
    static_mode: bool,
    gan2_service: Optional["LipsyncService"] = None,
    gan3_service: Optional["LipsyncService"] = None
):
    global lipsync_service_gan, lipsync_service_nogan, avatar_preloaded, avatar_static_mode
    global lipsync_service_gan2, lipsync_service_gan3
    lipsync_service_gan = gan_service
    lipsync_service_nogan = nogan_service
    lipsync_service_gan2 = gan2_service
    lipsync_service_gan3 = gan3_service
    avatar_preloaded = avatar
    avatar_static_mode = static_mode
