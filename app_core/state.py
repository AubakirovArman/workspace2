"""Shared application state for preloaded services and assets."""
from __future__ import annotations

from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from service import LipsyncService

lipsync_service_gan: Optional["LipsyncService"] = None
lipsync_service_nogan: Optional["LipsyncService"] = None
lipsync_service_gan2: Optional["LipsyncService"] = None  # Второй экземпляр GAN
lipsync_service_gan3: Optional["LipsyncService"] = None  # Третий экземпляр GAN
lipsync_service_gan4: Optional["LipsyncService"] = None  # Четвёртый экземпляр GAN
lipsync_service_gan5: Optional["LipsyncService"] = None  # Пятый экземпляр GAN
lipsync_service_gan6: Optional["LipsyncService"] = None  # Шестой экземпляр GAN
lipsync_service_gan7: Optional["LipsyncService"] = None  # Седьмой экземпляр GAN
lipsync_service_gan8: Optional["LipsyncService"] = None  # Восьмой экземпляр GAN
_GAN_POOL_TARGET = 8
lipsync_service_gan_pool: List[Optional["LipsyncService"]] = [None] * _GAN_POOL_TARGET
avatar_preloaded = None
avatar_static_mode: bool = True


def set_state(
    gan_service: "LipsyncService",
    nogan_service: Optional["LipsyncService"],
    avatar,
    static_mode: bool,
    *additional_gan_services: Optional["LipsyncService"],
):
    global lipsync_service_gan, lipsync_service_nogan, avatar_preloaded, avatar_static_mode
    global lipsync_service_gan2, lipsync_service_gan3, lipsync_service_gan4
    global lipsync_service_gan5, lipsync_service_gan6, lipsync_service_gan7, lipsync_service_gan8
    global lipsync_service_gan_pool

    lipsync_service_gan = gan_service
    lipsync_service_nogan = nogan_service

    pool: List[Optional["LipsyncService"]] = [gan_service, *additional_gan_services]
    if len(pool) < _GAN_POOL_TARGET:
        pool.extend([None] * (_GAN_POOL_TARGET - len(pool)))
    else:
        pool = pool[:_GAN_POOL_TARGET]

    lipsync_service_gan_pool = pool
    lipsync_service_gan2 = pool[1]
    lipsync_service_gan3 = pool[2]
    lipsync_service_gan4 = pool[3]
    lipsync_service_gan5 = pool[4]
    lipsync_service_gan6 = pool[5]
    lipsync_service_gan7 = pool[6]
    lipsync_service_gan8 = pool[7]

    avatar_preloaded = avatar
    avatar_static_mode = static_mode


def get_all_gan_services(include_none: bool = False) -> List[Optional["LipsyncService"]]:
    if include_none:
        return list(lipsync_service_gan_pool)
    return [svc for svc in lipsync_service_gan_pool if svc]
