from __future__ import annotations

from typing import Optional, List, Tuple
import subprocess
import numpy as np
import fcntl


class FfmpegIOMixin:
    def _start_writer(self, output_path: Optional[str], fps: float, frame_w: int, frame_h: int):
        """Запуск FFmpeg-пайпа для записи видео. Возвращает процесс или None."""
        if not output_path:
            return None

        dimension = f"{frame_w}x{frame_h}"
        codec, codec_opts = self._select_ffmpeg_codec()
        threads_value = getattr(self, 'ffmpeg_threads', 16)
        try:
            threads_int = int(threads_value)
        except (TypeError, ValueError):
            threads_int = 16

        blocksize = frame_w * frame_h * 3
        vf_filters: List[str] = []
        if frame_w % 2 != 0 or frame_h % 2 != 0:
            vf_filters.append('scale=w=trunc(iw/2)*2:h=trunc(ih/2)*2:flags=fast_bilinear')

        filter_threads = getattr(self, 'ffmpeg_filter_threads', 0)

        command: List[str] = ['ffmpeg', '-y', '-loglevel', 'error']
        if threads_int >= 0:
            command.extend(['-threads', str(threads_int)])

        command += [
            '-f', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-s', dimension,
            '-r', str(fps),
            '-blocksize', str(blocksize),
            '-i', 'pipe:0',
            '-i', self.temp_wav,
        ]

        if vf_filters:
            command += ['-vf', ','.join(vf_filters)]

        command += ['-c:v', codec] + codec_opts

        if filter_threads > 0:
            command += ['-filter_threads', str(filter_threads)]

        command += [
            '-pix_fmt', 'yuv420p',
            '-c:a', 'aac',
            '-ar', '16000',
            '-b:a', '128k',
            '-shortest',
            '-movflags', '+faststart',
            output_path,
        ]

        proc = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

        if proc.stdin is not None:
            try:
                target_pipe_size = max(blocksize * 4, 64 * 1024 * 1024)
                fcntl.fcntl(proc.stdin.fileno(), fcntl.F_SETPIPE_SZ, target_pipe_size)
            except (AttributeError, OSError):
                pass

        return proc

    def _write_frame(self, proc, frame) -> None:
        """Записать один кадр в пайп FFmpeg."""
        if proc is None:
            return
        try:
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8, copy=False)
            if not frame.flags.c_contiguous:
                frame = np.ascontiguousarray(frame)
            proc.stdin.write(memoryview(frame))
        except BrokenPipeError:
            error_output = proc.stderr.read().decode('utf-8', errors='ignore')
            raise RuntimeError(f"FFmpeg pipe broken: {error_output}") from None
        except TypeError:
            proc.stdin.write(frame.tobytes())

    def _finish_writer(self, proc) -> Tuple[int, str]:
        """Закрыть пайп и вернуть (код возврата, stderr)."""
        if proc is None:
            return (0, "")
        proc.stdin.close()
        returncode = proc.wait()
        error_output = proc.stderr.read().decode('utf-8', errors='ignore')
        return returncode, error_output
