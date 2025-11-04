from __future__ import annotations

import os
import subprocess
import ctypes
from typing import Tuple, List


class FfmpegCodecMixin:
    def _select_ffmpeg_codec(self) -> Tuple[str, List[str]]:
        """Вернуть выбранный видеокодек и его параметры для FFmpeg."""
        codec_name = os.getenv("FFMPEG_CODEC", "libx264").lower()

        if codec_name == "libsvtav1":
            preset = os.getenv("FFMPEG_SVT_PRESET", "9")
            crf = os.getenv("FFMPEG_CRF", "35")
            svt_params = os.getenv("FFMPEG_SVT_PARAMS")

            codec_opts: List[str] = ['-preset', preset, '-crf', crf]
            if svt_params:
                codec_opts += ['-svtav1-params', svt_params]

            return ('libsvtav1', codec_opts)

        # Fallback: libx264
        preset = os.getenv("FFMPEG_PRESET", "ultrafast")
        crf = os.getenv("FFMPEG_CRF", "24")
        x264_params = os.getenv(
            "FFMPEG_X264_PARAMS",
            "sliced-threads=1:rc-lookahead=10:sync-lookahead=10:bframes=0",
        )

        codec_opts = ['-preset', preset, '-crf', crf]
        if x264_params:
            codec_opts += ['-x264-params', x264_params]

        return ('libx264', codec_opts)
