"""
Modern Lipsync Service - Fast inference with preloaded models
Keeps models in memory for fast repeated processing
"""
import os
import sys
import argparse
from pathlib import Path
from typing import Optional, Tuple, List, Callable
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
        wav2lip_batch_size: int = 960,
        segmentation_path: Optional[str] = None,
        sr_path: Optional[str] = None,
        modules_root: Optional[str] = None,
        realesrgan_path: Optional[str] = None,
        realesrgan_outscale: float = 1.0
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
        self._video_cache = {}
        self.segmentation_model = None
        self.segmentation_enabled = False
        self._swap_regions_fn: Optional[Callable] = None
        self.sr_model = None
        self.sr_enabled = False
        self._sr_enhance_fn: Optional[Callable] = None
        self._modules_root = modules_root
        self._segmentation_path = segmentation_path
        self._sr_path = sr_path
        self._realesrgan_path = realesrgan_path
        self._realesrgan_outscale = max(1.0, float(realesrgan_outscale))
        self._sys_path_added = False
        self.realesrgan_enabled = False
        self._realesrgan_enhancer = None
        
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
        self.model = None
        self._is_torchscript = False

        try:
            scripted_model = torch.jit.load(checkpoint_path, map_location=self.device)
            scripted_model.eval()
            self.model = scripted_model.to(self.device)
            self._is_torchscript = True
            print("   ✓ TorchScript checkpoint loaded")
        except Exception as load_err:
            print(f"   ⚠️ TorchScript load failed ({load_err}); falling back to state dict")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            state_dict = checkpoint.get('state_dict', checkpoint)
            cleaned_state = {k.replace('module.', ''): v for k, v in state_dict.items()}
            model = Wav2Lip()
            model.load_state_dict(cleaned_state)
            self.model = model.to(self.device).eval()

        
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

        # Optional HD post-processing modules
        self._load_optional_modules()

    def _ensure_modules_root(self):
        if self._modules_root and not self._sys_path_added:
            if self._modules_root not in sys.path:
                sys.path.insert(0, self._modules_root)
            self._sys_path_added = True

    def _load_optional_modules(self):
        real_sr_loaded = False

        if self._segmentation_path and Path(self._segmentation_path).is_file():
            if self.device != 'cuda':
                print("   ⚠️ Segmentation requires CUDA; skipping")
            else:
                self._ensure_modules_root()
                try:
                    from face_parsing import init_parser, swap_regions
                    print("Loading segmentation model...")
                    self.segmentation_model = init_parser(self._segmentation_path)
                    self._swap_regions_fn = swap_regions
                    self.segmentation_enabled = True
                    print("   ✓ Segmentation model ready")
                except Exception as seg_err:
                    print(f"   ⚠️ Failed to load segmentation model ({seg_err})")
                    self.segmentation_model = None
                    self._swap_regions_fn = None
                    self.segmentation_enabled = False

        if self._realesrgan_path and Path(self._realesrgan_path).is_file():
            if self.device != 'cuda':
                print("   ⚠️ Real-ESRGAN requires CUDA; skipping")
            else:
                try:
                    self._ensure_modules_root()
                    from realesrgan import RealESRGANer  # type: ignore
                    from basicsr.archs.rrdbnet_arch import RRDBNet
                    print("Loading Real-ESRGAN model...")
                    rrdbnet = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32)
                    self._realesrgan_enhancer = RealESRGANer(
                        scale=4,
                        model_path=self._realesrgan_path,
                        model=rrdbnet,
                        tile=0,
                        tile_pad=10,
                        pre_pad=0,
                        half=(self.device == 'cuda')
                    )
                    self.realesrgan_enabled = True
                    real_sr_loaded = True
                    print("   ✓ Real-ESRGAN model ready")
                    # При наличии Real-ESRGAN классический ESRGAN больше не нужен
                    if self.sr_enabled:
                        self.sr_enabled = False
                        self.sr_model = None
                        self._sr_enhance_fn = None
                except ImportError:
                    print("   ⚠️ Real-ESRGAN package is not installed; skipping")
                except Exception as real_err:
                    print(f"   ⚠️ Failed to load Real-ESRGAN model ({real_err})")
                    self._realesrgan_enhancer = None
                    self.realesrgan_enabled = False

        if (not real_sr_loaded) and self._sr_path and Path(self._sr_path).is_file():
            if self.device != 'cuda':
                print("   ⚠️ Super-resolution requires CUDA; skipping")
            else:
                self._ensure_modules_root()
                try:
                    from basicsr.apply_sr import init_sr_model, enhance
                    print("Loading super-resolution model (ESRGAN)...")
                    self.sr_model = init_sr_model(self._sr_path)
                    self._sr_enhance_fn = enhance
                    self.sr_enabled = True
                    print("   ✓ Super-resolution model ready")
                except Exception as sr_err:
                    print(f"   ⚠️ Failed to load super-resolution model ({sr_err})")
                    self.sr_model = None
                    self._sr_enhance_fn = None
                    self.sr_enabled = False
    
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
                    print(batch_size)
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
        audio_path: str = "",
        output_path: Optional[str] = None,
        static: bool = False,
        fps: float = 25.0,
        pads: Tuple[int, int, int, int] = (0, 10, 0, 0),
        resize_factor: int = 1,
        crop: Tuple[int, int, int, int] = (0, -1, 0, -1),
        box: Tuple[int, int, int, int] = (-1, -1, -1, -1),
        rotate: bool = False,
        nosmooth: bool = False,
        audio_waveform: Optional[torch.Tensor] = None,
        audio_sample_rate: int = 16000,
        frame_sink: Optional[Callable[[np.ndarray], None]] = None
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
        video_cache_key = None
        video_cache_entry = None

        if can_use_cache:
            cache_key = self._static_cache_key(
                face_path, fps, pads, resize_factor, crop, rotate, nosmooth
            )
            cache_entry = self._static_cache.get(cache_key)
        elif box == (-1, -1, -1, -1):
            video_cache_key = self._video_cache_key(
                face_path, fps, pads, resize_factor, crop, rotate, nosmooth
            )
            video_cache_entry = self._video_cache.get(video_cache_key)

        static_face_resized = None
        if cache_entry:
            full_frames = list(cache_entry['frames'])
            video_fps = cache_entry['fps']
            stats['load_video_time'] = 0.0
            static_face_resized = cache_entry.get('resized_face')
        elif video_cache_entry and 'frames' in video_cache_entry:
            full_frames = [frame.copy() for frame in video_cache_entry['frames']]
            video_fps = video_cache_entry.get('fps', fps)
            stats['load_video_time'] = 0.0
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
        elif video_cache_entry and not static and box == (-1, -1, -1, -1):
            coords_list = video_cache_entry['coords']
            if len(coords_list) >= len(full_frames):
                coords_list = coords_list[:len(full_frames)]
                face_det_results = [
                    [frame[y1:y2, x1:x2], (y1, y2, x1, x2)]
                    for frame, (y1, y2, x1, x2) in zip(full_frames, coords_list)
                ]
                stats['face_detection_time'] = 0.0
            else:
                video_cache_entry = None  # Fallback to fresh detection
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
            elif not static and video_cache_key and box == (-1, -1, -1, -1):
                coords_list = [coords for _, coords in face_det_results]
                self._video_cache[video_cache_key] = {
                    'coords': coords_list,
                    'fps': video_fps,
                    'num_frames': len(coords_list),
                    'face_detection_time': detect_time,
                }
        
        # Inference (using preloaded model!)
        start = time.time()
        self._run_inference(
            full_frames,
            face_det_results,
            mel_chunks,
            output_path,
            video_fps,
            static,
            static_face_resized,
            frame_sink=frame_sink,
        )
        stats['inference_time'] = time.time() - start
        
        stats['total_time'] = time.time() - total_start
        stats['segmentation'] = bool(self.segmentation_enabled and self.segmentation_model is not None)
        stats['super_resolution'] = bool(self.sr_enabled and self.sr_model is not None)
        stats['real_esrgan'] = bool(self.realesrgan_enabled and self._realesrgan_enhancer is not None)

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
        self,
        full_frames: List[np.ndarray],
        face_det_results: List[Tuple[np.ndarray, Tuple[int, int, int, int]]],
        mel_chunks: List[np.ndarray],
        output_path: Optional[str],
        fps: float,
        static: bool,
        static_face_resized: Optional[np.ndarray],
        frame_sink: Optional[Callable[[np.ndarray], None]] = None,
    ) -> None:
        """Run Wav2Lip inference and emit frames to disk and/or a sink."""
        if not full_frames:
            raise ValueError("No video frames provided for inference")

        frame_h, frame_w = full_frames[0].shape[:-1]
        temp_dir = os.path.join(os.path.dirname(__file__), 'temp')
        os.makedirs(temp_dir, exist_ok=True)

        temp_result: Optional[str] = None
        video_writer: Optional[cv2.VideoWriter] = None

        if output_path:
            temp_result = os.path.join(temp_dir, 'result.avi')
            video_writer = cv2.VideoWriter(
                temp_result,
                cv2.VideoWriter_fourcc(*'DIVX'),
                fps,
                (frame_w, frame_h),
            )

        batch_size = self.wav2lip_batch_size

        for i in range(0, len(mel_chunks), batch_size):
            batch_mel = mel_chunks[i:i + batch_size]
            img_batch, mel_batch, frames_batch, coords_batch = [], [], [], []

            for j, mel_window in enumerate(batch_mel):
                idx = 0 if static else (i + j) % len(full_frames)
                frame_to_save = full_frames[idx].copy()
                face, coords = (
                    face_det_results[idx].copy()
                    if not static
                    else face_det_results[0].copy()
                )

                if static and static_face_resized is not None:
                    face_resized = static_face_resized
                else:
                    face_resized = cv2.resize(face, (self.img_size, self.img_size))

                img_batch.append(face_resized.copy())
                mel_batch.append(mel_window)
                frames_batch.append(frame_to_save)
                coords_batch.append(coords)

            img_batch_np = np.asarray(img_batch)
            mel_batch_np = np.asarray(mel_batch)

            img_masked = img_batch_np.copy()
            img_masked[:, self.img_size // 2 :] = 0

            img_batch_np = np.concatenate((img_masked, img_batch_np), axis=3) / 255.0
            mel_batch_np = np.reshape(
                mel_batch_np, [len(mel_batch_np), mel_batch_np.shape[1], mel_batch_np.shape[2], 1]
            )

            img_batch_tensor = torch.from_numpy(
                np.transpose(img_batch_np, (0, 3, 1, 2))
            ).float().to(self.device)

            mel_batch_tensor = torch.from_numpy(
                np.transpose(mel_batch_np, (0, 3, 1, 2))
            ).float().to(self.device)

            with torch.no_grad():
                pred = self.model(mel_batch_tensor, img_batch_tensor)

            pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.0

            for predicted_patch, frame, coords in zip(pred, frames_batch, coords_batch):
                y1, y2, x1, x2 = coords
                face_crop = frame[y1:y2, x1:x2]
                frame_patch = predicted_patch.astype(np.uint8)

                if self.realesrgan_enabled and self._realesrgan_enhancer is not None:
                    try:
                        enhanced_patch, _ = self._realesrgan_enhancer.enhance(
                            frame_patch,
                            outscale=self._realesrgan_outscale,
                        )
                        frame_patch = enhanced_patch.astype(np.uint8)
                    except Exception as real_err:
                        print(f"   ⚠️ Real-ESRGAN failed ({real_err}); disabling Real-ESRGAN")
                        self.realesrgan_enabled = False
                        self._realesrgan_enhancer = None

                if self.sr_enabled and self.sr_model is not None and self._sr_enhance_fn:
                    try:
                        frame_patch = self._sr_enhance_fn(self.sr_model, frame_patch)
                    except Exception as sr_err:
                        print(f"   ⚠️ Super-resolution failed ({sr_err}); disabling SR")
                        self.sr_enabled = False
                        self.sr_model = None
                        self._sr_enhance_fn = None

                frame_patch = cv2.resize(frame_patch, (x2 - x1, y2 - y1))

                if self.segmentation_enabled and self.segmentation_model is not None and self._swap_regions_fn:
                    try:
                        blended = self._swap_regions_fn(face_crop, frame_patch, self.segmentation_model)
                        frame_patch = blended.astype(np.uint8)
                    except Exception as seg_err:
                        print(f"   ⚠️ Segmentation blend failed ({seg_err}); disabling segmentation")
                        self.segmentation_enabled = False
                        self.segmentation_model = None
                        self._swap_regions_fn = None

                frame[y1:y2, x1:x2] = frame_patch

                if video_writer is not None:
                    video_writer.write(frame)

                if frame_sink is not None:
                    # Send a copy to avoid downstream mutation issues
                    frame_sink(frame.copy())

        if video_writer is not None:
            video_writer.release()

        if output_path and temp_result:
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

    def process_with_preloaded(
        self,
        audio_path: str,
        output_path: Optional[str] = None,
        fps: float = 25.0,
        audio_waveform: Optional[torch.Tensor] = None,
        audio_sample_rate: int = 16000,
        frame_sink: Optional[Callable[[np.ndarray], None]] = None
    ) -> dict:
        """
        Быстрая обработка с использованием предзагруженного статического лица.
        Пропускает загрузку видео и детекцию лица - использует кэш.
        
        Args:
            audio_path: Путь к аудио файлу
            output_path: Путь для сохранения видео (если None, то не сохраняет)
            fps: FPS для видео
            audio_waveform: Опциональный готовый аудио тензор
            audio_sample_rate: Частота дискретизации аудио
            frame_sink: Опциональная функция для потоковой обработки кадров
            
        Returns:
            dict с информацией о времени обработки
        """
        # Проверяем наличие кэша
        if not self._static_cache:
            raise RuntimeError(
                "Статическое лицо не предзагружено. "
                "Сначала вызовите preload_static_face() или используйте process() с static=True"
            )
        
        # Берём первую запись из кэша (обычно она одна)
        cache_entry = next(iter(self._static_cache.values()))
        
        stats = {}
        total_start = time.time()
        
        # Используем предзагруженные данные
        full_frames = list(cache_entry['frames'])
        video_fps = cache_entry.get('fps', fps)
        cached_face = cache_entry['face'].copy()
        cached_coords = cache_entry['coords']
        static_face_resized = cache_entry.get('resized_face')
        
        stats['load_video_time'] = 0.0  # Из кэша!
        stats['face_detection_time'] = 0.0  # Из кэша!
        
        # Обработка аудио
        start = time.time()
        mel, mel_chunks = self._process_audio(
            audio_path, video_fps, audio_waveform, audio_sample_rate
        )
        stats['process_audio_time'] = time.time() - start
        stats['num_mel_chunks'] = len(mel_chunks)
        
        # Подгоняем количество кадров под аудио
        full_frames = full_frames[:len(mel_chunks)]
        stats['num_frames'] = len(full_frames)
        stats['fps'] = video_fps
        
        # Используем предзагруженное лицо
        face_det_results = [[cached_face, cached_coords]]
        
        # Инференс модели
        start = time.time()
        self._run_inference(
            full_frames,
            face_det_results,
            mel_chunks,
            output_path,
            video_fps,
            True,  # static=True
            static_face_resized,
            frame_sink=frame_sink
        )
        stats['inference_time'] = time.time() - start
        
        # Постобработка выполняется внутри _run_inference
        stats['postprocess_time'] = 0.0
        stats['write_time'] = 0.0
        

        
        stats['total_time'] = time.time() - total_start
        stats['segmentation'] = self.segmentation_enabled
        stats['super_resolution'] = self.sr_enabled
        stats['real_esrgan'] = self.realesrgan_enabled
        
        return stats

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

    def _video_cache_key(
        self,
        face_path: str,
        fps: float,
        pads: Tuple[int, int, int, int],
        resize_factor: int,
        crop: Tuple[int, int, int, int],
        rotate: bool,
        nosmooth: bool
    ) -> Tuple:
        # Prefix with "video" to avoid collision with static cache entries
        return ("video",) + self._static_cache_key(
            face_path, fps, pads, resize_factor, crop, rotate, nosmooth
        )

    def preload_video_cache(
        self,
        face_path: str,
        fps: float = 25.0,
        pads: Tuple[int, int, int, int] = (0, 10, 0, 0),
        resize_factor: int = 1,
        crop: Tuple[int, int, int, int] = (0, -1, 0, -1),
        rotate: bool = False,
        nosmooth: bool = False
    ) -> None:
        """
        Pre-compute face detection for dynamic video sources so repeated requests
        can reuse crop coordinates instead of re-running the detector.
        """
        cache_key = self._video_cache_key(
            face_path, fps, pads, resize_factor, crop, rotate, nosmooth
        )
        if cache_key in self._video_cache:
            return

        full_frames, video_fps = self._load_video(
            face_path, False, fps, resize_factor, crop, rotate
        )
        if not full_frames:
            raise ValueError("Video does not contain any frames")

        usable_frames: List[np.ndarray] = []
        coords: List[Tuple[int, int, int, int]] = []
        detect_time = 0.0

        scales_to_try = [1, 2, 3, 4]
        detection_success = False

        for scale in scales_to_try:
            frames_for_detection = full_frames
            scale_start = time.time()
            if scale > 1:
                frames_for_detection = []
                for frame in full_frames:
                    h, w = frame.shape[:2]
                    new_w = max(1, w // scale)
                    new_h = max(1, h // scale)
                    frames_for_detection.append(
                        cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
                    )
            try:
                detect_start = time.time()
                face_det_results = self.detect_faces(frames_for_detection, pads, nosmooth)
                detect_duration = time.time() - detect_start
                detect_time += detect_duration + (time.time() - scale_start - detect_duration)

                for original_frame, (_, coord_set) in zip(full_frames, face_det_results):
                    if scale > 1:
                        y1, y2, x1, x2 = coord_set
                        y1, y2 = int(y1 * scale), int(y2 * scale)
                        x1, x2 = int(x1 * scale), int(x2 * scale)
                        coord_set = (y1, y2, x1, x2)
                    usable_frames.append(original_frame.copy())
                    coords.append(coord_set)
                detection_success = True
                break
            except RuntimeError as runtime_err:
                if "Image too big for GPU" in str(runtime_err) and scale < scales_to_try[-1]:
                    print(f"⚠️ Face detection OOM at scale {scale}. Retrying with scale {scale + 1}...")
                    continue
                raise
            except ValueError:
                # Retry per frame below
                detect_time += time.time() - scale_start
                break

        if not detection_success and not coords:
            print("⚠️ Bulk face detection failed; retrying frame-by-frame with fallbacks")
            for idx, frame in enumerate(full_frames):
                success = False
                for scale in scales_to_try:
                    scaled_frame = frame
                    if scale > 1:
                        h, w = frame.shape[:2]
                        new_w = max(1, w // scale)
                        new_h = max(1, h // scale)
                        scaled_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
                    try:
                        frame_start = time.time()
                        result = self.detect_faces([scaled_frame], pads, nosmooth)[0]
                        detect_time += time.time() - frame_start
                        y1, y2, x1, x2 = result[1]
                        if scale > 1:
                            y1, y2 = int(y1 * scale), int(y2 * scale)
                            x1, x2 = int(x1 * scale), int(x2 * scale)
                        usable_frames.append(frame.copy())
                        coords.append((y1, y2, x1, x2))
                        success = True
                        break
                    except (ValueError, RuntimeError):
                        continue
                if not success:
                    print(f"   ⚠️ Пропущен кадр #{idx}: лицо не найдено ни при одном масштабировании")

        if not usable_frames:
            raise ValueError("Face not detected in any frame of the video avatar")

        self._video_cache[cache_key] = {
            "coords": coords,
            "frames": tuple(usable_frames),
            "fps": video_fps,
            "num_frames": len(coords),
            "face_detection_time": detect_time,
        }


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
