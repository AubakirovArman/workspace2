#!/usr/bin/env python3
"""Utility script to fine-tune Wav2Lip on a single video clip."""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

# Ensure we can import modern-lipsync modules
REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = REPO_ROOT / "modern-lipsync"
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))

from models.wav2lip import Wav2Lip  # type: ignore  # noqa: E402
from utils.audio import ModernAudioProcessor  # type: ignore  # noqa: E402

try:
    import face_detection  # type: ignore  # noqa: E402
except ImportError as import_err:  # pragma: no cover
    raise SystemExit("face_detection package is required for finetuning") from import_err


@dataclass
class DetectionResult:
    bbox: Tuple[int, int, int, int]
    face: np.ndarray


class SingleVideoDataset(Dataset):
    """Dataset that pairs mel chunks with corresponding face crops."""

    def __init__(
        self,
        faces: Sequence[np.ndarray],
        mel_chunks: Sequence[np.ndarray],
        loop_frames: bool = True,
        img_size: int = 96,
    ) -> None:
        if not faces:
            raise ValueError("Dataset requires at least one face crop")
        if not mel_chunks:
            raise ValueError("Dataset requires at least one mel chunk")

        self.faces = faces
        self.mel_chunks = mel_chunks
        self.loop_frames = loop_frames
        self.img_size = img_size
        self.length = len(mel_chunks)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        mel_idx = idx
        if self.loop_frames:
            face_idx = idx % len(self.faces)
        else:
            face_idx = min(idx, len(self.faces) - 1)

        face = self.faces[face_idx]
        mel = self.mel_chunks[mel_idx]

        face_resized = cv2.resize(face, (self.img_size, self.img_size))
        face_float = face_resized.astype(np.float32)
        face_masked = face_float.copy()
        half = face_masked.shape[1] // 2
        face_masked[:, half:, :] = 0.0

        concat = np.concatenate((face_masked, face_float), axis=2)
        face_input = torch.from_numpy(concat.transpose(2, 0, 1)) / 255.0
        target = torch.from_numpy(face_float.transpose(2, 0, 1)) / 255.0
        mel_tensor = torch.from_numpy(mel.astype(np.float32)).unsqueeze(0)

        return mel_tensor, face_input, target


def extract_audio(video_path: Path, output_wav: Path, sample_rate: int = 16000) -> None:
    """Use ffmpeg to extract mono PCM audio to WAV."""
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        str(output_wav),
        "-loglevel",
        "error",
    ]
    subprocess.run(cmd, check=True)


def read_video_frames(video_path: Path) -> Tuple[List[np.ndarray], float]:
    """Load video frames with OpenCV and return frames + FPS."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if math.isclose(fps, 0.0):
        fps = 25.0

    frames: List[np.ndarray] = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    if not frames:
        raise RuntimeError("Video contains no frames")
    return frames, fps


def detect_faces(
    frames: Sequence[np.ndarray],
    detector: face_detection.FaceAlignment,
    pads: Tuple[int, int, int, int] = (0, 40, 0, 0),
    smooth: bool = True,
) -> List[DetectionResult]:
    """Run batched face detection with optional smoothing."""
    if not frames:
        return []

    batch_size = 16
    predictions: List[np.ndarray] = []

    while True:
        predictions.clear()
        try:
            for start in range(0, len(frames), batch_size):
                batch = np.array(frames[start:start + batch_size])
                batch_preds = detector.get_detections_for_batch(batch)
                predictions.extend(batch_preds)
        except RuntimeError as runtime_err:
            if batch_size == 1:
                raise RuntimeError("Face detection failed even at batch_size=1") from runtime_err
            batch_size //= 2
            print(f"⚠️ Face detection OOM, reducing batch_size to {batch_size}")
            continue
        break

    pady1, pady2, padx1, padx2 = pads
    boxes: List[List[int]] = []
    results: List[DetectionResult] = []

    for rect, frame in zip(predictions, frames):
        if rect is None:
            raise RuntimeError("Face not detected in a frame")
        y1 = max(0, int(rect[1]) - pady1)
        y2 = min(frame.shape[0], int(rect[3]) + pady2)
        x1 = max(0, int(rect[0]) - padx1)
        x2 = min(frame.shape[1], int(rect[2]) + padx2)
        boxes.append([x1, y1, x2, y2])

    boxes_np = np.array(boxes, dtype=np.float32)
    if smooth:
        boxes_np = smooth_boxes(boxes_np, window=5)

    for idx, (frame, (x1, y1, x2, y2)) in enumerate(zip(frames, boxes_np.astype(int))):
        face = frame[y1:y2, x1:x2]
        if face.size == 0:
            raise RuntimeError(f"Empty face crop detected at frame {idx}")
        results.append(DetectionResult(bbox=(y1, y2, x1, x2), face=face))

    return results


def smooth_boxes(boxes: np.ndarray, window: int = 5) -> np.ndarray:
    """Simple moving-average smoothing for bounding boxes."""
    smoothed = boxes.copy()
    for idx in range(len(boxes)):
        if idx + window > len(boxes):
            window_slice = boxes[len(boxes) - window:]
        else:
            window_slice = boxes[idx:idx + window]
        smoothed[idx] = window_slice.mean(axis=0)
    return smoothed


def compute_mel_chunks(audio_path: Path, fps: float, audio_processor: ModernAudioProcessor, mel_step: int = 16) -> List[np.ndarray]:
    """Compute mel spectrogram chunks aligned with video FPS."""
    mel = audio_processor.extract_audio_features(str(audio_path))
    mel_chunks: List[np.ndarray] = []
    mel_idx_multiplier = 80.0 / fps
    idx = 0
    while True:
        start_idx = int(idx * mel_idx_multiplier)
        if start_idx + mel_step > mel.shape[1]:
            mel_chunks.append(mel[:, mel.shape[1] - mel_step:])
            break
        mel_chunks.append(mel[:, start_idx:start_idx + mel_step])
        idx += 1
    return mel_chunks


def load_checkpoint(model: nn.Module, checkpoint_path: Path) -> None:
    ckpt = torch.load(str(checkpoint_path), map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    clean_state = {k.replace("module.", ""): v for k, v in ckpt.items() if hasattr(v, "shape")}
    missing, unexpected = model.load_state_dict(clean_state, strict=False)
    if missing:
        print(f"⚠️ Missing keys: {len(missing)}")
    if unexpected:
        print(f"⚠️ Unexpected keys: {len(unexpected)}")


def export_torchscript(model: nn.Module, export_path: Path) -> None:
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        dummy_audio = torch.randn(1, 1, 80, 16, device=device)
        dummy_face = torch.randn(1, 6, 96, 96, device=device)
        traced = torch.jit.trace(model, (dummy_audio, dummy_face), strict=False)
    traced.save(str(export_path))
    print(f"✅ TorchScript saved to {export_path}")


def train(args: argparse.Namespace) -> None:
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    work_dir = Path(args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    video_path = Path(args.video).resolve()
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    print("🎞️ Reading video frames...")
    frames, fps = read_video_frames(video_path)
    print(f"   Frames: {len(frames)} | FPS: {fps:.2f}")

    print("🔊 Extracting audio track...")
    audio_path = work_dir / "finetune_audio.wav"
    extract_audio(video_path, audio_path, sample_rate=16000)

    print("🧠 Initialising face detector...")
    detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D, flip_input=False, device=args.detector_device)

    print("🙂 Running face detection...")
    detections = detect_faces(frames, detector, pads=(0, args.pad_bottom, 0, 0), smooth=not args.no_smooth)

    faces = [det.face for det in detections]
    print(f"   Detected faces: {len(faces)}")

    audio_processor = ModernAudioProcessor()
    print("🎼 Computing mel chunks...")
    mel_chunks = compute_mel_chunks(audio_path, fps, audio_processor, mel_step=16)
    print(f"   Mel chunks: {len(mel_chunks)}")

    dataset = SingleVideoDataset(faces=faces, mel_chunks=mel_chunks, loop_frames=not args.no_loop)

    if args.val_split > 0.0 and len(dataset) > 1:
        val_len = max(1, int(len(dataset) * args.val_split))
        train_len = len(dataset) - val_len
        train_ds, val_ds = random_split(dataset, [train_len, val_len])
    else:
        train_ds = dataset
        val_ds = None

    if len(train_ds) < 2:
        raise RuntimeError("Training set needs at least 2 samples for batch-norm layers. Add more footage or reduce val_split.")

    drop_last = len(train_ds) > args.batch_size
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=drop_last,
    )
    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    model = Wav2Lip().to(device)
    load_checkpoint(model, Path(args.checkpoint))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))
    criterion = nn.L1Loss()

    best_val = float("inf")
    history = []

    print("🚀 Starting fine-tune...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        num_batches = 0
        for mel, face_inp, target in train_loader:
            mel = mel.to(device)
            face_inp = face_inp.to(device)
            target = target.to(device)

            optimizer.zero_grad(set_to_none=True)
            pred = model(mel, face_inp)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(1, num_batches)
        log_entry = {"epoch": epoch, "train_loss": avg_loss}

        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            count = 0
            with torch.no_grad():
                for mel, face_inp, target in val_loader:
                    mel = mel.to(device)
                    face_inp = face_inp.to(device)
                    target = target.to(device)
                    pred = model(mel, face_inp)
                    loss = criterion(pred, target)
                    val_loss += loss.item()
                    count += 1
            avg_val = val_loss / max(1, count)
            log_entry["val_loss"] = avg_val
            if avg_val < best_val:
                best_val = avg_val
                save_checkpoint(model, Path(args.output_checkpoint), suffix="best")
        print(f"Epoch {epoch:03d}: train_loss={avg_loss:.6f}" + (f" val_loss={log_entry.get('val_loss', 0):.6f}" if 'val_loss' in log_entry else ""))
        history.append(log_entry)

        if args.save_every and (epoch % args.save_every == 0):
            save_checkpoint(model, Path(args.output_checkpoint), suffix=f"epoch{epoch}")

    final_ckpt = save_checkpoint(model, Path(args.output_checkpoint), suffix="final")

    history_path = Path(args.output_checkpoint).with_suffix(".history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    print(f"📝 Training history saved to {history_path}")

    if args.export_ts:
        export_torchscript(model, Path(args.export_ts))

    print("✅ Fine-tune complete")
    print(f"   Latest checkpoint: {final_ckpt}")


def save_checkpoint(model: nn.Module, output_path: Path, suffix: str = "final") -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_path.with_name(f"{output_path.stem}_{suffix}{output_path.suffix}")
    state = {"state_dict": model.state_dict()}
    torch.save(state, ckpt_path)
    print(f"💾 Saved checkpoint: {ckpt_path}")
    return ckpt_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Wav2Lip on a single video clip")
    parser.add_argument("--video", required=True, help="Path to input video with audio track")
    parser.add_argument("--checkpoint", default=str(REPO_ROOT / "checkpoints" / "wav2lip_gan.pth"), help="Base checkpoint to start from")
    parser.add_argument("--output-checkpoint", default=str(REPO_ROOT / "checkpoints" / "wav2lip_gan_finetuned.pth"), help="Where to write the fine-tuned checkpoint")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--save-every", type=int, default=0, help="Save intermediate checkpoints every N epochs (0 = only best/final)")
    parser.add_argument("--val-split", type=float, default=0.05, help="Fraction of samples for validation set (0 disables validation)")
    parser.add_argument("--no-loop", action="store_true", help="Disable looping frames when audio is longer than video")
    parser.add_argument("--no-smooth", action="store_true", help="Disable bounding box smoothing")
    parser.add_argument("--pad-bottom", type=int, default=50, help="Extra pixels to include below the detected box")
    parser.add_argument("--device", default="cuda", help="Torch device for training")
    parser.add_argument("--detector-device", default="cuda", help="Device for face detector (cuda or cpu)")
    parser.add_argument("--workers", type=int, default=2, help="Dataloader worker threads")
    parser.add_argument("--work-dir", default=str(REPO_ROOT / "temp" / "finetune"), help="Working directory for intermediate files")
    parser.add_argument("--export-ts", default="", help="Optional path to export TorchScript model after training")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
