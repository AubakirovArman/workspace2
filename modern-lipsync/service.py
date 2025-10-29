"""
Modern Lipsync Service - Fast inference with preloaded models
Keeps models in memory for fast repeated processing
"""
import os
import sys
import argparse
from pathlib import Path
from typing import Optional, Tuple, List
import time
import shutil

import numpy as np
import cv2
import torch
import torchaudio
from tqdm import tqdm

from models import Wav2Lip
from utils.audio import ModernAudioProcessor
import face_detection


class LipsyncService:
    """
    Preloaded lipsync service for fast repeated inference
    Models stay in memory between requests
    """
    
    def __init__(
        self, 
        checkpoint_path: str,
        device: str = 'cuda',
        face_det_batch_size: int = 16,
        wav2lip_batch_size: int = 32
    ):
        """
        Initialize service with preloaded models
        
        Args:
            checkpoint_path: Path to Wav2Lip checkpoint
            device: 'cuda' or 'cpu'
            face_det_batch_size: Batch size for face detection
            wav2lip_batch_size: Batch size for Wav2Lip inference
        """
        self.device = device
        self.face_det_batch_size = face_det_batch_size
        self.wav2lip_batch_size = wav2lip_batch_size
        self.img_size = 96
        self.mel_step_size = 16
        self._static_cache = {}
        
        print(f"🚀 Starting Lipsync Service on {device}...")
        
        # Preload models
        start = time.time()
        self._load_models(checkpoint_path)
        load_time = time.time() - start
        
        print(f"✅ Service ready! Models loaded in {load_time:.2f}s")
        print(f"   - Face Detector: Ready")
        print(f"   - Wav2Lip Model: Ready")
        print(f"   - Audio Processor: Ready")
    
    def _load_models(self, checkpoint_path: str):
        """Load all models into memory"""
        # Load Wav2Lip model
        print("Loading Wav2Lip model...")
        self.model = torch.jit.load(checkpoint_path, map_location=self.device)
        self.model.eval()
        
        # Load face detector
        print("Loading face detector...")
        self.face_detector = face_detection.FaceAlignment(
            face_detection.LandmarksType._2D,
            flip_input=False,
            device=self.device
        )
        
        # Initialize audio processor
        print("Loading audio processor...")
        self.audio_processor = ModernAudioProcessor()
    
    def get_smoothened_boxes(self, boxes: np.ndarray, T: int = 5) -> np.ndarray:
        """Smooth face detection boxes"""
        for i in range(len(boxes)):
            if i + T > len(boxes):
                window = boxes[len(boxes) - T:]
            else:
                window = boxes[i : i + T]
            boxes[i] = np.mean(window, axis=0)
        return boxes
    
    def detect_faces(
        self, 
        images: List[np.ndarray],
        pads: Tuple[int, int, int, int] = (0, 10, 0, 0),
        nosmooth: bool = False
    ) -> List[Tuple]:
        """
        Detect faces in images (uses preloaded detector)
        
        Args:
            images: List of image frames
            pads: Padding (top, bottom, left, right)
            nosmooth: Disable smoothing
            
        Returns:
            List of (cropped_face, coords) tuples
        """
        batch_size = self.face_det_batch_size
        
        while True:
            predictions = []
            try:
                for i in range(0, len(images), batch_size):
                    batch = np.array(images[i:i + batch_size])
                    predictions.extend(self.face_detector.get_detections_for_batch(batch))
            except RuntimeError as e:
                if batch_size == 1:
                    raise RuntimeError('Image too big for GPU')
                batch_size //= 2
                print(f'OOM recovery: reducing batch to {batch_size}')
                continue
            break
        
        results = []
        pady1, pady2, padx1, padx2 = pads
        
        for rect, image in zip(predictions, images):
            if rect is None:
                raise ValueError('Face not detected!')
            
            y1 = max(0, rect[1] - pady1)
            y2 = min(image.shape[0], rect[3] + pady2)
            x1 = max(0, rect[0] - padx1)
            x2 = min(image.shape[1], rect[2] + padx2)
            
            results.append([x1, y1, x2, y2])
        
        boxes = np.array(results)
        if not nosmooth:
            boxes = self.get_smoothened_boxes(boxes, T=5)
        
        results = [
            [image[y1:y2, x1:x2], (y1, y2, x1, x2)]
            for image, (x1, y1, x2, y2) in zip(images, boxes)
        ]
        
        return results
    
    def process(
        self,
        face_path: str,
        audio_path: str,
        output_path: str,
        static: bool = False,
        fps: float = 25.0,
        pads: Tuple[int, int, int, int] = (0, 10, 0, 0),
        resize_factor: int = 1,
        crop: Tuple[int, int, int, int] = (0, -1, 0, -1),
        box: Tuple[int, int, int, int] = (-1, -1, -1, -1),
        rotate: bool = False,
        nosmooth: bool = False,
        audio_waveform: Optional[torch.Tensor] = None,
        audio_sample_rate: int = 16000
    ) -> dict:
        """
        Process single request (fast - models already loaded!)
        
        Returns:
            dict with timing info and stats
        """
        stats = {}
        total_start = time.time()

        # Static cache can be used only when we control the crop box
        can_use_cache = static and box == (-1, -1, -1, -1)
        cache_key = None
        cache_entry = None
        frames_for_cache = None

        if can_use_cache:
            cache_key = self._static_cache_key(
                face_path, fps, pads, resize_factor, crop, rotate, nosmooth
            )
            cache_entry = self._static_cache.get(cache_key)

        static_face_resized = None
        if cache_entry:
            full_frames = list(cache_entry['frames'])
            video_fps = cache_entry['fps']
            stats['load_video_time'] = 0.0
            static_face_resized = cache_entry.get('resized_face')
        else:
            start = time.time()
            full_frames, video_fps = self._load_video(
                face_path, static, fps, resize_factor, crop, rotate
            )
            load_time = time.time() - start
            stats['load_video_time'] = load_time
            if can_use_cache:
                frames_for_cache = tuple(frame.copy() for frame in full_frames)
        stats['num_frames'] = len(full_frames)
        stats['fps'] = video_fps
        
        # Process audio
        start = time.time()
        mel, mel_chunks = self._process_audio(
            audio_path, video_fps, audio_waveform, audio_sample_rate
        )
        stats['process_audio_time'] = time.time() - start
        stats['num_mel_chunks'] = len(mel_chunks)
        
        # Adjust frames
        full_frames = full_frames[:len(mel_chunks)]
        
        # Face detection (using preloaded detector!)
        if cache_entry:
            face_det_results = [[cache_entry['face'].copy(), cache_entry['coords']]]
            stats['face_detection_time'] = 0.0
            if static_face_resized is None and 'resized_face' in cache_entry:
                static_face_resized = cache_entry['resized_face']
        else:
            start = time.time()
            if box[0] == -1:
                if not static:
                    face_det_results = self.detect_faces(full_frames, pads, nosmooth)
                else:
                    face_det_results = self.detect_faces([full_frames[0]], pads, nosmooth)
            else:
                y1, y2, x1, x2 = box
                face_det_results = [
                    [f[y1:y2, x1:x2], (y1, y2, x1, x2)]
                    for f in full_frames
                ]
            detect_time = time.time() - start
            stats['face_detection_time'] = detect_time

            if can_use_cache and frames_for_cache is not None:
                cached_face, cached_coords = face_det_results[0]
                self._static_cache[cache_key] = {
                    'frames': frames_for_cache,
                    'fps': video_fps,
                    'face': cached_face.copy(),
                    'coords': cached_coords,
                    'load_video_time': stats['load_video_time'],
                    'face_detection_time': detect_time,
                    'resized_face': cv2.resize(cached_face, (self.img_size, self.img_size))
                }
                static_face_resized = self._static_cache[cache_key]['resized_face']
        
        # Inference (using preloaded model!)
        start = time.time()
        self._run_inference(
            full_frames, face_det_results, mel_chunks, 
            output_path, video_fps, static, static_face_resized
        )
        stats['inference_time'] = time.time() - start
        
        stats['total_time'] = time.time() - total_start
        
        return stats
    
    def _load_video(
        self, face_path, static, fps, resize_factor, crop, rotate
    ) -> Tuple[List[np.ndarray], float]:
        """Load video frames"""
        ext = Path(face_path).suffix.lower()
        
        if ext in ['.jpg', '.png', '.jpeg']:
            full_frames = [cv2.imread(face_path)]
            video_fps = fps
        else:
            video_stream = cv2.VideoCapture(face_path)
            video_fps = video_stream.get(cv2.CAP_PROP_FPS)
            
            full_frames = []
            while True:
                still_reading, frame = video_stream.read()
                if not still_reading:
                    video_stream.release()
                    break
                
                if resize_factor > 1:
                    frame = cv2.resize(
                        frame,
                        (frame.shape[1] // resize_factor,
                         frame.shape[0] // resize_factor)
                    )
                
                if rotate:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                
                y1, y2, x1, x2 = crop
                if x2 == -1: x2 = frame.shape[1]
                if y2 == -1: y2 = frame.shape[0]
                
                frame = frame[y1:y2, x1:x2]
                full_frames.append(frame)
        
        return full_frames, video_fps
    
    def _process_audio(
        self,
        audio_path: str,
        fps: float,
        waveform: Optional[torch.Tensor] = None,
        sample_rate: int = 16000
    ) -> Tuple[np.ndarray, List]:
        """Process audio to mel spectrogram"""
        # Prepare temp directory and file
        temp_dir = os.path.join(os.path.dirname(__file__), 'temp')
        os.makedirs(temp_dir, exist_ok=True)
        self.temp_wav = os.path.join(temp_dir, 'temp.wav')
        
        target_sr = self.audio_processor.config.sample_rate
        
        if waveform is not None:
            waveform = waveform.detach()
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            waveform = waveform.to(self.audio_processor.device).float()
            
            if sample_rate != target_sr:
                resampler = torchaudio.transforms.Resample(
                    sample_rate, target_sr
                ).to(self.audio_processor.device)
                waveform = resampler(waveform)
                sample_rate = target_sr
            
            torchaudio.save(self.temp_wav, waveform.detach().cpu(), sample_rate)
            mel = self.audio_processor.melspectrogram(waveform)
        else:
            # Extract/copy audio to WAV
            if not audio_path.endswith('.wav'):
                cmd = f'ffmpeg -y -i "{audio_path}" -acodec pcm_s16le -ar {target_sr} "{self.temp_wav}" -loglevel quiet'
                os.system(cmd)
            else:
                shutil.copy2(audio_path, self.temp_wav)
            
            # Get mel spectrogram
            mel = self.audio_processor.extract_audio_features(self.temp_wav)
        
        # Create chunks
        mel_chunks = []
        mel_idx_multiplier = 80.0 / fps
        i = 0
        
        while True:
            start_idx = int(i * mel_idx_multiplier)
            if start_idx + self.mel_step_size > mel.shape[1]:
                mel_chunks.append(mel[:, mel.shape[1] - self.mel_step_size:])
                break
            mel_chunks.append(mel[:, start_idx:start_idx + self.mel_step_size])
            i += 1
        
        return mel, mel_chunks
    
    def _run_inference(
        self, full_frames, face_det_results, mel_chunks,
        output_path, fps, static, static_face_resized: Optional[np.ndarray]
    ):
        """Run model inference"""
        frame_h, frame_w = full_frames[0].shape[:-1]
        temp_dir = os.path.join(os.path.dirname(__file__), 'temp')
        os.makedirs(temp_dir, exist_ok=True)
        temp_result = os.path.join(temp_dir, 'result.avi')
        out = cv2.VideoWriter(
            temp_result,
            cv2.VideoWriter_fourcc(*'DIVX'),
            fps,
            (frame_w, frame_h)
        )
        
        batch_size = self.wav2lip_batch_size
        
        for i in range(0, len(mel_chunks), batch_size):
            batch_mel = mel_chunks[i:i + batch_size]
            img_batch, mel_batch, frames_batch, coords_batch = [], [], [], []
            
            for j, m in enumerate(batch_mel):
                idx = 0 if static else (i + j) % len(full_frames)
                frame_to_save = full_frames[idx].copy()
                face, coords = face_det_results[idx].copy() if not static else face_det_results[0].copy()
                
                if static and static_face_resized is not None:
                    face_resized = static_face_resized
                else:
                    face_resized = cv2.resize(face, (self.img_size, self.img_size))
                img_batch.append(face_resized.copy())
                mel_batch.append(m)
                frames_batch.append(frame_to_save)
                coords_batch.append(coords)
            
            img_batch = np.asarray(img_batch)
            mel_batch = np.asarray(mel_batch)
            
            img_masked = img_batch.copy()
            img_masked[:, self.img_size//2:] = 0
            
            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
            mel_batch = np.reshape(
                mel_batch,
                [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1]
            )
            
            img_batch = torch.FloatTensor(
                np.transpose(img_batch, (0, 3, 1, 2))
            ).to(self.device)
            
            mel_batch = torch.FloatTensor(
                np.transpose(mel_batch, (0, 3, 1, 2))
            ).to(self.device)
            
            with torch.no_grad():
                pred = self.model(mel_batch, img_batch)
            
            pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.
            
            for p, f, c in zip(pred, frames_batch, coords_batch):
                y1, y2, x1, x2 = c
                p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
                f[y1:y2, x1:x2] = p
                out.write(f)
        
        out.release()
        
        # Combine with audio (предпочтительно через NVENC)
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        nvenc_cmd = (
            f'ffmpeg -y -i "{temp_result}" -i "{self.temp_wav}" '
            '-c:v h264_nvenc -preset p4 -tune ll '
            '-rc constqp -qp 23 -pix_fmt yuv420p '
            '-c:a aac -ar 16000 -b:a 128k '
            '-shortest -movflags +faststart '
            f'"{output_path}" -loglevel quiet'
        )
        result = os.system(nvenc_cmd)
        if result != 0:
            fallback_cmd = (
                f'ffmpeg -y -i "{temp_result}" -i "{self.temp_wav}" '
                '-c:v libx264 -preset veryfast -pix_fmt yuv420p '
                '-c:a aac -ar 16000 -b:a 128k '
                '-shortest -movflags +faststart '
                f'"{output_path}" -loglevel quiet'
            )
            result = os.system(fallback_cmd)
        if result != 0:
            raise RuntimeError(f"ffmpeg failed with code {result}")

    def preload_static_face(
        self,
        face_path: str,
        fps: float = 25.0,
        pads: Tuple[int, int, int, int] = (0, 10, 0, 0),
        resize_factor: int = 1,
        crop: Tuple[int, int, int, int] = (0, -1, 0, -1),
        rotate: bool = False,
        nosmooth: bool = False
    ):
        """
        Preload cached data for a static face so realtime requests skip repeated setup.
        """
        cache_key = self._static_cache_key(
            face_path, fps, pads, resize_factor, crop, rotate, nosmooth
        )
        if cache_key in self._static_cache:
            return

        full_frames, video_fps = self._load_video(
            face_path, True, fps, resize_factor, crop, rotate
        )
        frames_for_cache = tuple(frame.copy() for frame in full_frames)
        face_det_results = self.detect_faces([full_frames[0]], pads, nosmooth)
        cached_face, cached_coords = face_det_results[0]
        self._static_cache[cache_key] = {
            'frames': frames_for_cache,
            'fps': video_fps,
            'face': cached_face.copy(),
            'coords': cached_coords,
            'load_video_time': 0.0,
            'face_detection_time': 0.0,
            'resized_face': cv2.resize(cached_face, (self.img_size, self.img_size))
        }

    def _static_cache_key(
        self,
        face_path: str,
        fps: float,
        pads: Tuple[int, int, int, int],
        resize_factor: int,
        crop: Tuple[int, int, int, int],
        rotate: bool,
        nosmooth: bool
    ) -> Tuple:
        resolved_path = str(Path(face_path).resolve())
        return (
            resolved_path,
            float(fps),
            tuple(int(p) for p in pads),
            int(resize_factor),
            tuple(int(c) for c in crop),
            bool(rotate),
            bool(nosmooth),
        )


def main():
    parser = argparse.ArgumentParser(description='Lipsync Service - Fast inference')
    
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--face_det_batch_size', type=int, default=16)
    parser.add_argument('--wav2lip_batch_size', type=int, default=128)
    
    # Request parameters
    parser.add_argument('--face', type=str, required=True)
    parser.add_argument('--audio', type=str, required=True)
    parser.add_argument('--outfile', type=str, default='results/output.mp4')
    parser.add_argument('--static', action='store_true')
    parser.add_argument('--fps', type=float, default=25.0)
    parser.add_argument('--pads', nargs='+', type=int, default=[0, 10, 0, 0])
    parser.add_argument('--resize_factor', type=int, default=1)
    parser.add_argument('--crop', nargs='+', type=int, default=[0, -1, 0, -1])
    parser.add_argument('--box', nargs='+', type=int, default=[-1, -1, -1, -1])
    parser.add_argument('--rotate', action='store_true')
    parser.add_argument('--nosmooth', action='store_true')
    
    args = parser.parse_args()
    
    # Initialize service (loads models once)
    service = LipsyncService(
        checkpoint_path=args.checkpoint_path,
        device=args.device,
        face_det_batch_size=args.face_det_batch_size,
        wav2lip_batch_size=args.wav2lip_batch_size
    )
    
    # Process request (fast!)
    print(f"\n🎬 Processing: {args.face} + {args.audio}")
    stats = service.process(
        face_path=args.face,
        audio_path=args.audio,
        output_path=args.outfile,
        static=args.static,
        fps=args.fps,
        pads=tuple(args.pads),
        resize_factor=args.resize_factor,
        crop=tuple(args.crop),
        box=tuple(args.box),
        rotate=args.rotate,
        nosmooth=args.nosmooth
    )
    
    # Print stats
    print(f"\n✅ Done! Saved to: {args.outfile}")
    print(f"\n📊 Performance:")
    print(f"   Load video:       {stats['load_video_time']:.2f}s")
    print(f"   Process audio:    {stats['process_audio_time']:.2f}s")
    print(f"   Face detection:   {stats['face_detection_time']:.2f}s")
    print(f"   Model inference:  {stats['inference_time']:.2f}s")
    print(f"   ─────────────────────────")
    print(f"   Total:            {stats['total_time']:.2f}s")
    print(f"\n   Frames: {stats['num_frames']}, FPS: {stats['fps']:.1f}")


if __name__ == '__main__':
    main()
