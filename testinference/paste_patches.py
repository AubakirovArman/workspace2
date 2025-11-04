#!/usr/bin/env python3
import argparse
import time
import json
from pathlib import Path
import numpy as np
import cv2
import sys
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

# Для единообразия логики, пути не требуются, но оставим заготовку
ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description='Paste predicted mouth patches into frames')
    parser.add_argument('--frames', type=str, required=True, help='Директория исходных кадров')
    parser.add_argument('--patches', type=str, required=True, help='Директория патчей от Wav2Lip')
    parser.add_argument('--coords', type=str, required=True, help='coords.npy')
    parser.add_argument('--out', type=str, required=True, help='Директория для сохранения объединённых кадров')
    parser.add_argument('--workers', type=int, default=max(os.cpu_count() or 1, 1), help='Количество потоков для вставки патчей')
    parser.add_argument('--png_level', type=int, default=3, help='Уровень сжатия PNG (0=без сжатия, 9=максимум)')
    # Потоковый режим: писать кадры сразу в ffmpeg без PNG
    parser.add_argument('--pipe_out', type=str, default=None, help='Выходной mp4 через ffmpeg pipe (стриминг кадров)')
    parser.add_argument('--audio', type=str, default=None, help='Путь к аудио (для pipe режима)')
    parser.add_argument('--fps', type=float, default=25.0, help='FPS исходного видео/статической картинки')
    parser.add_argument('--fast', action='store_true', help='Ускоренное кодирование в pipe режиме (низкая задержка)')
    parser.add_argument('--stream_workers', type=int, default=4, help='Количество потоков подготовки кадров в pipe режиме')
    parser.add_argument('--buffer', type=int, default=64, help='Размер буфера кадров между подготовкой и ffmpeg')
    parser.add_argument('--pipe_format', type=str, default='raw', choices=['raw', 'mjpeg', 'png'], help='Формат стриминга кадров в pipe: raw|mjpeg|png')
    parser.add_argument('--jpeg_q', type=int, default=85, help='Качество JPEG при pipe_format=mjpeg (0-100)')
    parser.add_argument('--log', type=str, default=None, help='Путь к JSONL лог-файлу')
    args = parser.parse_args()

    frames_dir = Path(args.frames)
    patches_dir = Path(args.patches)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    frame_files = sorted(frames_dir.glob('*.png'))
    patch_files = sorted(patches_dir.glob('*.png'))
    coords_arr = np.load(args.coords)

    # Автоматическая поддержка статики: если есть один кадр, повторяем его для всех патчей
    is_static = len(frame_files) == 1
    n = len(patch_files) if is_static else min(len(frame_files), len(patch_files), len(coords_arr))

    # Если координаты одни (статичный аватар), повторяем их
    if is_static and len(coords_arr) == 1:
        coords = [coords_arr[0] for _ in range(n)]
    else:
        coords = coords_arr

    # Загрузим первый кадр заранее для статики
    static_frame = None
    if is_static:
        static_frame = cv2.imread(str(frame_files[0]))

    # Предзагрузим патчи в память, чтобы избежать повторного чтения с диска
    patch_imgs = [cv2.imread(str(p)) for p in patch_files]

    # Если статика и координаты постоянные — заранее ресайзим все патчи к одному ROI
    if is_static and len(coords_arr) == 1 and n > 0 and patch_imgs and patch_imgs[0] is not None:
        y1, y2, x1, x2 = (coords_arr[0][0], coords_arr[0][1], coords_arr[0][2], coords_arr[0][3])
        roi_w, roi_h = (x2 - x1), (y2 - y1)
        if roi_w > 0 and roi_h > 0:
            resized = []
            for img in patch_imgs:
                # Защитимся от повреждённых PNG
                if img is None:
                    resized.append(None)
                else:
                    resized.append(cv2.resize(img, (roi_w, roi_h), interpolation=(cv2.INTER_NEAREST if args.fast else cv2.INTER_LINEAR)))
            patch_imgs = resized

    def paste_one(i: int):
        try:
            f = static_frame.copy() if is_static else cv2.imread(str(frame_files[i]))
            p = patch_imgs[i]
            y1, y2, x1, x2 = coords[i]
            if p is None:
                return False
            if is_static:
                # Для статики патч уже пред-ресайзен к ROI
                f[y1:y2, x1:x2] = p
            else:
                p_resized = cv2.resize(p, (x2 - x1, y2 - y1), interpolation=(cv2.INTER_NEAREST if args.fast else cv2.INTER_LINEAR))
                f[y1:y2, x1:x2] = p_resized
            cv2.imwrite(str(out_dir / f'{i:06d}.png'), f, [cv2.IMWRITE_PNG_COMPRESSION, int(args.png_level)])
            return True
        except Exception:
            return False

    def select_codec(fast: bool = False) -> tuple[str, list[str]]:
        """Выбор кодека без NVENC: используем только libx264."""
        if fast:
            # libx264: ультрабыстрый пресет и низкая задержка, авто-потоки
            return 'libx264', ['-preset', 'ultrafast', '-tune', 'zerolatency', '-crf', '28', '-threads', '0']
        return 'libx264', ['-crf', '23', '-preset', 'veryfast', '-threads', '0']

    def stream_to_ffmpeg():
        """Вставлять патчи и посылать кадры в ffmpeg через stdin (без PNG)."""
        nonlocal static_frame
        # Определим размер кадра
        sample = static_frame if is_static else cv2.imread(str(frame_files[0]))
        if sample is None:
            raise RuntimeError('Не удалось загрузить образец кадра для определения размеров')
        # Паддинг до чётных размеров, чтобы избежать лишнего scale в ffmpeg
        h, w = sample.shape[:2]
        pad_right = 1 if (w % 2) != 0 else 0
        pad_bottom = 1 if (h % 2) != 0 else 0
        if pad_right or pad_bottom:
            sample = cv2.copyMakeBorder(sample, 0, pad_bottom, 0, pad_right, borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
            if is_static:
                static_frame = sample
            h, w = sample.shape[:2]
        dimension = f'{w}x{h}'
        # Без масштабирования, только конвертация формата
        vf = 'format=yuv420p'
        codec, codec_opts = select_codec(fast=bool(args.fast))

        if not args.audio:
            raise RuntimeError('Для pipe режима требуется --audio')

        # Определим длительность аудио и ограничим число кадриков по ней
        audio_duration_sec = None
        try:
            probe_cmd = [
                'ffprobe', '-v', 'error', '-hide_banner',
                '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                str(args.audio)
            ]
            pr = subprocess.run(probe_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5)
            if pr.returncode == 0:
                out = (pr.stdout or b'').decode('utf-8', errors='ignore').strip()
                try:
                    audio_duration_sec = float(out)
                except Exception:
                    audio_duration_sec = None
        except Exception:
            audio_duration_sec = None
        effective_n = n
        if audio_duration_sec is not None and audio_duration_sec > 0:
            try:
                max_frames = int(audio_duration_sec * float(args.fps))
                if max_frames > 0:
                    effective_n = min(n, max_frames)
            except Exception:
                pass

        # Конструируем вход по формату pipe
        input_args = []
        if args.pipe_format == 'raw':
            input_args = ['-f', 'rawvideo', '-vcodec', 'rawvideo', '-s', dimension, '-pix_fmt', 'bgr24', '-r', str(args.fps), '-i', '-']
        elif args.pipe_format == 'mjpeg':
            input_args = ['-f', 'image2pipe', '-vcodec', 'mjpeg', '-r', str(args.fps), '-i', '-']
        else:  # png
            input_args = ['-f', 'image2pipe', '-vcodec', 'png', '-r', str(args.fps), '-i', '-']

        cmd = [
            'ffmpeg', '-y',
        ] + input_args + [
            '-i', args.audio,
            '-vf', vf,
            '-c:v', codec,
        ] + codec_opts + [
            '-pix_fmt', 'yuv420p',
            '-c:a', 'aac', '-ar', '16000', '-b:a', '128k',
            '-shortest', '-movflags', '+faststart',
            '-fflags', '+nobuffer', '-probesize', '32', '-analyzeduration', '0',
            str(args.pipe_out),
            '-loglevel', 'error'
        ]

        t0 = time.time()
        # Открываем ffmpeg и готовим параллельную подготовку кадров
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

        from queue import Queue
        from threading import Thread, Event

        q: Queue = Queue(maxsize=int(args.buffer))
        ok = 0
        stop_evt = Event()

        # Поток записи, сохраняет порядок кадров
        def writer_thread():
            nonlocal ok
            next_idx = 0
            pending = {}
            while True:
                item = q.get()
                try:
                    if item is None:
                        break
                    i, frame = item
                    pending[i] = frame
                    # Пишем по порядку, если доступно
                    while next_idx in pending:
                        f = pending.pop(next_idx)
                        if args.pipe_format == 'raw':
                            frame_contig = np.ascontiguousarray(f)
                            try:
                                proc.stdin.write(memoryview(frame_contig))
                            except TypeError:
                                proc.stdin.write(frame_contig.tobytes())
                        elif args.pipe_format == 'mjpeg':
                            ok_, buf = cv2.imencode('.jpg', f, [cv2.IMWRITE_JPEG_QUALITY, int(args.jpeg_q)])
                            if not ok_:
                                continue
                            proc.stdin.write(buf.tobytes())
                        else:  # png
                            ok_, buf = cv2.imencode('.png', f, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                            if not ok_:
                                continue
                            proc.stdin.write(buf.tobytes())
                        try:
                            proc.stdin.flush()
                        except Exception:
                            pass
                        ok += 1
                        next_idx += 1
                except BrokenPipeError:
                    # ffmpeg завершился, прекращаем запись
                    stop_evt.set()
                    break
                finally:
                    q.task_done()

        wt = Thread(target=writer_thread, daemon=True)
        wt.start()

        # Подготовка кадров в пуле потоков
        def prepare_frame(i: int):
            f = static_frame.copy() if is_static else cv2.imread(str(frame_files[i]))
            if f is None:
                return i, None
            p = patch_imgs[i]
            y1, y2, x1, x2 = coords[i]
            if p is None:
                return i, None
            if is_static:
                f[y1:y2, x1:x2] = p
            else:
                p_resized = cv2.resize(p, (x2 - x1, y2 - y1), interpolation=(cv2.INTER_NEAREST if args.fast else cv2.INTER_LINEAR))
                f[y1:y2, x1:x2] = p_resized
            # Паддинг до чётных размеров для каждого кадра
            hh, ww = f.shape[:2]
            pr = 1 if (ww % 2) != 0 else 0
            pb = 1 if (hh % 2) != 0 else 0
            if pr or pb:
                f = cv2.copyMakeBorder(f, 0, pb, 0, pr, borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
            return i, f

        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=int(args.stream_workers)) as ex:
            for i, f in ex.map(prepare_frame, range(effective_n)):
                if f is None:
                    continue
                if stop_evt.is_set():
                    break
                q.put((i, f))

        # Завершаем запись
        q.put(None)
        q.join()
        try:
            if proc.stdin:
                proc.stdin.close()
        except Exception:
            pass

        ret = proc.wait()
        dt = time.time() - t0
        # Сохраним stderr ffmpeg в файл для диагностики
        try:
            err_log_dir = (Path(args.pipe_out).parent if args.pipe_out else Path('.'))
            err_log_dir.mkdir(parents=True, exist_ok=True)
            err_log_path = err_log_dir / 'ffmpeg_stream_err.txt'
            err_bytes = b''
            try:
                err_bytes = proc.stderr.read() or b''
            except Exception:
                err_bytes = b''
            with open(err_log_path, 'wb') as ef:
                ef.write(err_bytes)
        except Exception:
            err_log_path = None
        if ret != 0:
            err_msg = ''
            try:
                if err_log_path and err_log_path.exists():
                    err_msg = f' (stderr saved to {err_log_path})'
            except Exception:
                pass
            raise RuntimeError(f'ffmpeg pipe encoding failed (code={ret}){err_msg}')
        print(f'✅ Стриминг завершён: {ok}/{effective_n} кадров → {args.pipe_out}, fps={args.fps}, fast={bool(args.fast)}, время={dt:.2f}s')

        # Логирование результатов в JSONL
        if args.log:
            try:
                log_path = Path(args.log)
                log_path.parent.mkdir(parents=True, exist_ok=True)
                payload = {
                    'step': 'paste_patches_stream',
                    'frames': len(frame_files),
                    'patches': len(patch_files),
                    'coords_len': len(coords_arr),
                    'is_static': is_static,
                    'frames_streamed': ok,
                    'fps': float(args.fps),
                    'fast': bool(args.fast),
                    'codec': codec,
                    'time_sec': round(dt, 4),
                    'video_out': str(args.pipe_out)
                }
                with open(log_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(payload, ensure_ascii=False) + '\n')
            except Exception as e:
                print(f"⚠️  Не удалось записать лог: {e}")

    if args.pipe_out:
        # Потоковый режим: кадры → ffmpeg pipe
        stream_to_ffmpeg()
    else:
        # Обычный режим: записываем PNG кадры на диск
        t0 = time.time()
        ok = 0
        with ThreadPoolExecutor(max_workers=int(args.workers)) as ex:
            futures = [ex.submit(paste_one, i) for i in range(n)]
            for fut in as_completed(futures):
                if fut.result():
                    ok += 1

        dt = time.time() - t0
        print(f'✅ Патчи вставлены: {ok}/{n} кадров, dir={out_dir}, workers={args.workers}, png_level={int(args.png_level)}, время={dt:.2f}s')

        # Логирование результатов в JSONL
        if args.log:
            try:
                log_path = Path(args.log)
                log_path.parent.mkdir(parents=True, exist_ok=True)
                payload = {
                    'step': 'paste_patches',
                    'frames': len(frame_files),
                    'patches': len(patch_files),
                    'coords_len': len(coords_arr),
                    'is_static': is_static,
                    'frames_pasted': ok,
                    'workers': int(args.workers),
                    'png_level': int(args.png_level),
                    'time_sec': round(dt, 4),
                    'out_dir': str(out_dir)
                }
                with open(log_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(payload, ensure_ascii=False) + '\n')
            except Exception as e:
                print(f"⚠️  Не удалось записать лог: {e}")


if __name__ == '__main__':
    main()
