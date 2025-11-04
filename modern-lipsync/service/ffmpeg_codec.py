from __future__ import annotations

import os
import subprocess
import ctypes
from typing import Tuple, List


class FfmpegCodecMixin:
    def _select_ffmpeg_codec(self) -> Tuple[str, List[str]]:
        """Вернуть выбранный видеокодек и его параметры для FFmpeg."""
        codec_name = "libx264"

        preset = os.getenv("FFMPEG_PRESET", "ultrafast")
        tune = os.getenv("FFMPEG_TUNE", "zerolatency")

        crf = os.getenv("FFMPEG_CRF", "23")
        maxrate = os.getenv("FFMPEG_MAXRATE", "12M")
        bufsize = os.getenv("FFMPEG_BUFSIZE", "12M")

        gop_size = os.getenv("FFMPEG_GOP", "60")
        keyint_min = os.getenv("FFMPEG_KEYINT_MIN", gop_size)
        scenecut = os.getenv("FFMPEG_SC_THRESHOLD", "0")
        max_b_frames = os.getenv("FFMPEG_MAX_BFRAMES", "0")

        extra_params = os.getenv(
            "FFMPEG_X264_PARAMS",
            "nal-hrd=none:rc-lookahead=0:sync-lookahead=0:b-adapt=0",
        )

        codec_opts: List[str] = [
            '-preset', preset,
            '-tune', tune,
            '-crf', crf,
            '-maxrate', maxrate,
            '-bufsize', bufsize,
            '-g', gop_size,
            '-keyint_min', keyint_min,
            '-sc_threshold', scenecut,
            '-bf', max_b_frames,
            '-movflags', '+faststart',
        ]

        if extra_params:
            codec_opts += ['-x264-params', extra_params]

        threads = os.getenv("FFMPEG_THREADS")
        if threads:
            codec_opts += ['-threads', threads]

        return (codec_name, codec_opts)
