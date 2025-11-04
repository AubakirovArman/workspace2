#!/usr/bin/env python3
import argparse
import time
import json
from pathlib import Path
import numpy as np
import cv2
import torch
import sys

# Добавляем путь к modern-lipsync для импорта моделей
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'modern-lipsync'))

from models import Wav2Lip


def load_model(checkpoint_path: str, device: str):
    # TorchScript сначала
    try:
        m = torch.jit.load(checkpoint_path, map_location=device)
        m.eval()
        return m.to(device), True
    except Exception:
        ckpt = torch.load(checkpoint_path, map_location=device)
        state_dict = None
        if isinstance(ckpt, dict):
            if 'state_dict' in ckpt and isinstance(ckpt['state_dict'], dict):
                state_dict = ckpt['state_dict']
            else:
                # Популярные альтернативные ключи
                for key in ('model', 'model_state', 'net', 'generator'):
                    if key in ckpt and isinstance(ckpt[key], dict):
                        state_dict = ckpt[key]
                        break
        # Если всё ещё None, возьмём только тензоры из корневого словаря
        if state_dict is None and isinstance(ckpt, dict):
            state_dict = {k: v for k, v in ckpt.items() if hasattr(v, 'shape')}

        cleaned = {k.replace('module.', ''): v for k, v in state_dict.items()}
        m = Wav2Lip()
        # Разрешаем неполные совпадения ключей
        m.load_state_dict(cleaned, strict=False)
        return m.to(device).eval(), False


def main():
    parser = argparse.ArgumentParser(description='Run Wav2Lip forward per mel-chunk and emit patches')
    parser.add_argument('--checkpoint', type=str, required=True, help='Путь к весам Wav2Lip')
    parser.add_argument('--frames', type=str, required=True, help='Директория исходных кадров')
    parser.add_argument('--mel_chunks', type=str, required=True, help='mel_chunks.npy')
    parser.add_argument('--coords', type=str, required=True, help='coords.npy')
    parser.add_argument('--img_size', type=int, default=96, help='Размер квадрата лица для Wav2Lip')
    parser.add_argument('--batch_size', type=int, default=64, help='Размер батча инференса')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--out', type=str, required=True, help='Директория для сохранения патчей PNG')
    parser.add_argument('--static', action='store_true', help='Статичный аватар (использовать первый кадр для всех чанков)')
    parser.add_argument('--fp16', action='store_true', help='Использовать FP16/AMP для ускорения на GPU')
    parser.add_argument('--log', type=str, default=None, help='Путь к JSONL лог-файлу')
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(Path(args.frames).glob('*.png'))
    frames = [cv2.imread(str(p)) for p in files]
    coords = np.load(args.coords)
    mel_chunks = np.load(args.mel_chunks)

    # Ограничить кадры по количеству чанков
    # Для статики не обрезаем: будем использовать idx=0 для всех чанков

    # Подготовка батчей в стиле сервиса
    img_size = args.img_size
    t_load = time.time()
    model, is_ts = load_model(args.checkpoint, args.device)
    # Перевод модели в half при запросе FP16
    use_fp16 = bool(args.fp16) and str(args.device).startswith('cuda')
    if use_fp16:
        try:
            model = model.half()
        except Exception:
            # Если модель TorchScript не поддерживает .half(), продолжим без перевода весов
            use_fp16 = False
    dt_load = time.time() - t_load

    print(f'✅ Модель загружена ({"TorchScript" if is_ts else "state_dict"}): {dt_load:.2f}s')

    def make_batches():
        img_batch, mel_batch, frames_batch, coords_batch = [], [], [], []
        for i, mel_win in enumerate(mel_chunks):
            idx = 0 if args.static else (i % len(frames))
            f = frames[idx].copy()
            y1, y2, x1, x2 = coords[idx]
            face = f[y1:y2, x1:x2]
            face = cv2.resize(face, (img_size, img_size))
            img_batch.append(face)
            mel_batch.append(mel_win)
            frames_batch.append(f)
            coords_batch.append((y1, y2, x1, x2))
            if len(img_batch) >= args.batch_size:
                yield img_batch, mel_batch, frames_batch, coords_batch
                img_batch, mel_batch, frames_batch, coords_batch = [], [], [], []
        if img_batch:
            yield img_batch, mel_batch, frames_batch, coords_batch

    total_forward = 0.0
    total_tensor = 0.0
    total_post = 0.0
    total_prep = 0.0

    idx_out = 0
    for img_batch, mel_batch, frames_batch, coords_batch in make_batches():
        t_prep = time.time()
        img_batch_np = np.asarray(img_batch)
        mel_batch_np = np.asarray(mel_batch)

        img_masked = img_batch_np.copy()
        img_masked[:, img_size // 2 :] = 0
        img_batch_np = np.concatenate((img_masked, img_batch_np), axis=3) / 255.0
        mel_batch_np = np.reshape(
            mel_batch_np, [len(mel_batch_np), mel_batch_np.shape[1], mel_batch_np.shape[2], 1]
        )
        total_prep += time.time() - t_prep

        t_tensor = time.time()
        img_t = torch.from_numpy(np.transpose(img_batch_np, (0, 3, 1, 2)))
        mel_t = torch.from_numpy(np.transpose(mel_batch_np, (0, 3, 1, 2)))
        # Приведение типов и отправка на устройство
        if use_fp16:
            img_t = img_t.to(dtype=torch.float16)
            mel_t = mel_t.to(dtype=torch.float16)
        else:
            img_t = img_t.to(dtype=torch.float32)
            mel_t = mel_t.to(dtype=torch.float32)
        img_t = img_t.to(args.device)
        mel_t = mel_t.to(args.device)
        total_tensor += time.time() - t_tensor

        t_fwd = time.time()
        with torch.no_grad():
            if use_fp16:
                # Автокаст для смешанной точности
                with torch.cuda.amp.autocast():
                    pred = model(mel_t, img_t)
            else:
                pred = model(mel_t, img_t)
        total_forward += time.time() - t_fwd

        t_post = time.time()
        pred = pred.detach().float().cpu().numpy().transpose(0, 2, 3, 1) * 255.0
        total_post += time.time() - t_post

        # Сохранить патчи
        for p in pred:
            patch = p.astype(np.uint8)
            cv2.imwrite(str(out_dir / f'{idx_out:06d}.png'), patch)
            idx_out += 1

    print(
        f'✅ Патчи сохранены: {out_dir}, '
        f'prep={total_prep:.2f}s tensor={total_tensor:.2f}s forward={total_forward:.2f}s post={total_post:.2f}s'
    )

    # Логирование результатов в JSONL
    if args.log:
        try:
            log_path = Path(args.log)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                'step': 'run_wav2lip',
                'device': args.device,
                'img_size': args.img_size,
                'batch_size': args.batch_size,
                'static': bool(args.static),
                'fp16': bool(use_fp16),
                'time_total_forward_sec': round(total_forward, 4),
                'time_total_tensor_sec': round(total_tensor, 4),
                'time_total_prep_sec': round(total_prep, 4),
                'time_total_post_sec': round(total_post, 4),
                'model_load_sec': round(dt_load, 4),
                'out_dir': str(out_dir),
                'checkpoint': args.checkpoint
            }
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(payload, ensure_ascii=False) + '\n')
        except Exception as e:
            print(f"⚠️  Не удалось записать лог: {e}")


if __name__ == '__main__':
    main()
