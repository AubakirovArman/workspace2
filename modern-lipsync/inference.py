"""
Modern Lipsync - Inference Script
Updated for PyTorch 2.8.0+ and modern libraries
"""
import os
import sys
import argparse
import subprocess
import platform
from pathlib import Path
from typing import List, Tuple

import numpy as np
import cv2
import torch
from tqdm import tqdm

# Import our modern modules
from models import Wav2Lip
from utils.audio import ModernAudioProcessor
import face_detection


# Argument parser
parser = argparse.ArgumentParser(
    description='Modern Lipsync - Lip-sync videos using Wav2Lip with PyTorch 2.8.0+'
)

parser.add_argument(
    '--checkpoint_path', type=str, required=True,
    help='Path to model checkpoint (e.g., Wav2Lip-SD-GAN.pt or Wav2Lip-SD-NOGAN.pt)'
)

parser.add_argument(
    '--face', type=str, required=True,
    help='Path to video/image containing faces'
)

parser.add_argument(
    '--audio', type=str, required=True,
    help='Path to audio/video file to use as audio source'
)

parser.add_argument(
    '--outfile', type=str, default='results/result_voice.mp4',
    help='Output video path'
)

parser.add_argument(
    '--static', action='store_true',
    help='Use only first video frame for inference'
)

parser.add_argument(
    '--fps', type=float, default=25.0,
    help='FPS for static image input'
)

parser.add_argument(
    '--pads', nargs='+', type=int, default=[0, 10, 0, 0],
    help='Padding (top, bottom, left, right) for face detection'
)

parser.add_argument(
    '--face_det_batch_size', type=int, default=16,
    help='Batch size for face detection'
)

parser.add_argument(
    '--wav2lip_batch_size', type=int, default=16,
    help='Batch size for Wav2Lip inference'
)

parser.add_argument(
    '--resize_factor', type=int, default=1,
    help='Reduce resolution by this factor'
)

parser.add_argument(
    '--crop', nargs='+', type=int, default=[0, -1, 0, -1],
    help='Crop video (top, bottom, left, right)'
)

parser.add_argument(
    '--box', nargs='+', type=int, default=[-1, -1, -1, -1],
    help='Constant bounding box for face (top, bottom, left, right)'
)

parser.add_argument(
    '--rotate', action='store_true',
    help='Rotate video 90 degrees clockwise'
)

parser.add_argument(
    '--nosmooth', action='store_true',
    help='Disable face detection smoothing'
)

parser.add_argument(
    '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
    help='Device to use (cuda or cpu)'
)


class ModernLipsyncInference:
    """Modern Lipsync inference engine"""
    
    def __init__(self, args):
        self.args = args
        self.args.img_size = 96
        self.device = args.device
        self.mel_step_size = 20
        
        # Initialize audio processor
        self.audio_processor = ModernAudioProcessor()
        
        # Check if face is static image
        if os.path.isfile(args.face):
            ext = Path(args.face).suffix.lower()
            if ext in ['.jpg', '.png', '.jpeg']:
                self.args.static = True
        
        print(f'Using {self.device} for inference')
    
    def get_smoothened_boxes(self, boxes: np.ndarray, T: int = 5) -> np.ndarray:
        """Smooth face detection boxes over temporal window"""
        for i in range(len(boxes)):
            if i + T > len(boxes):
                window = boxes[len(boxes) - T:]
            else:
                window = boxes[i : i + T]
            boxes[i] = np.mean(window, axis=0)
        return boxes
    
    def face_detect(self, images: List[np.ndarray]) -> List[Tuple]:
        """Detect faces in image batch"""
        detector = face_detection.FaceAlignment(
            face_detection.LandmarksType._2D,
            flip_input=False,
            device=self.device
        )
        
        batch_size = self.args.face_det_batch_size
        
        while True:
            predictions = []
            try:
                for i in tqdm(range(0, len(images), batch_size), desc='Face detection'):
                    batch = np.array(images[i:i + batch_size])
                    predictions.extend(detector.get_detections_for_batch(batch))
            except RuntimeError as e:
                if batch_size == 1:
                    raise RuntimeError(
                        'Image too big for GPU. Use --resize_factor to reduce resolution'
                    )
                batch_size //= 2
                print(f'Recovering from OOM error; New batch size: {batch_size}')
                continue
            break
        
        results = []
        pady1, pady2, padx1, padx2 = self.args.pads
        
        for rect, image in zip(predictions, images):
            if rect is None:
                os.makedirs('temp', exist_ok=True)
                cv2.imwrite('temp/faulty_frame.jpg', image)
                raise ValueError(
                    'Face not detected! Check temp/faulty_frame.jpg'
                )
            
            y1 = max(0, rect[1] - pady1)
            y2 = min(image.shape[0], rect[3] + pady2)
            x1 = max(0, rect[0] - padx1)
            x2 = min(image.shape[1], rect[2] + padx2)
            
            results.append([x1, y1, x2, y2])
        
        boxes = np.array(results)
        if not self.args.nosmooth:
            boxes = self.get_smoothened_boxes(boxes, T=5)
        
        results = [
            [image[y1:y2, x1:x2], (y1, y2, x1, x2)]
            for image, (x1, y1, x2, y2) in zip(images, boxes)
        ]
        
        del detector
        return results
    
    def datagen(self, frames: List[np.ndarray], mels: List[np.ndarray]):
        """Generate batches for inference"""
        img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []
        
        if self.args.box[0] == -1:
            if not self.args.static:
                face_det_results = self.face_detect(frames)
            else:
                face_det_results = self.face_detect([frames[0]])
        else:
            print('Using specified bounding box instead of face detection...')
            y1, y2, x1, x2 = self.args.box
            face_det_results = [
                [f[y1:y2, x1:x2], (y1, y2, x1, x2)]
                for f in frames
            ]
        
        for i, m in enumerate(mels):
            idx = 0 if self.args.static else i % len(frames)
            frame_to_save = frames[idx].copy()
            face, coords = face_det_results[idx].copy()
            
            face = cv2.resize(face, (self.args.img_size, self.args.img_size))
            
            img_batch.append(face)
            mel_batch.append(m)
            frame_batch.append(frame_to_save)
            coords_batch.append(coords)
            
            if len(img_batch) >= self.args.wav2lip_batch_size:
                img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)
                
                img_masked = img_batch.copy()
                img_masked[:, self.args.img_size//2:] = 0
                
                img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
                mel_batch = np.reshape(
                    mel_batch,
                    [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1]
                )
                
                yield img_batch, mel_batch, frame_batch, coords_batch
                img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []
        
        if len(img_batch) > 0:
            img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)
            
            img_masked = img_batch.copy()
            img_masked[:, self.args.img_size//2:] = 0
            
            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
            mel_batch = np.reshape(
                mel_batch,
                [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1]
            )
            
            yield img_batch, mel_batch, frame_batch, coords_batch
    
    def load_model(self, checkpoint_path: str) -> Wav2Lip:
        """Load Wav2Lip model from checkpoint"""
        print(f"Loading checkpoint from: {checkpoint_path}")
        
        try:
            # Try loading as TorchScript first
            model = torch.jit.load(checkpoint_path, map_location=self.device)
            print("Loaded as TorchScript model")
            return model.eval()
        except:
            # Fall back to regular checkpoint loading
            if str(self.device).startswith('cuda'):
                checkpoint = torch.load(checkpoint_path, weights_only=False)
            else:
                checkpoint = torch.load(
                    checkpoint_path,
                    map_location=lambda storage, loc: storage,
                    weights_only=False
                )
            
            model = Wav2Lip()
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if "state_dict" in checkpoint:
                    state_dict = checkpoint["state_dict"]
                else:
                    state_dict = checkpoint
                
                # Remove 'module.' prefix if present
                new_state_dict = {}
                for k, v in state_dict.items():
                    new_state_dict[k.replace('module.', '')] = v
                
                model.load_state_dict(new_state_dict)
            
            model = model.to(self.device)
            return model.eval()
    
    def load_video_frames(self) -> Tuple[List[np.ndarray], float]:
        """Load frames from video or image"""
        if not os.path.isfile(self.args.face):
            raise ValueError('--face must be a valid file path')
        
        ext = Path(self.args.face).suffix.lower()
        
        if ext in ['.jpg', '.png', '.jpeg']:
            full_frames = [cv2.imread(self.args.face)]
            fps = self.args.fps
        else:
            video_stream = cv2.VideoCapture(self.args.face)
            fps = video_stream.get(cv2.CAP_PROP_FPS)
            
            print('Reading video frames...')
            full_frames = []
            
            while True:
                still_reading, frame = video_stream.read()
                if not still_reading:
                    video_stream.release()
                    break
                
                if self.args.resize_factor > 1:
                    frame = cv2.resize(
                        frame,
                        (frame.shape[1] // self.args.resize_factor,
                         frame.shape[0] // self.args.resize_factor)
                    )
                
                if self.args.rotate:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                
                y1, y2, x1, x2 = self.args.crop
                if x2 == -1:
                    x2 = frame.shape[1]
                if y2 == -1:
                    y2 = frame.shape[0]
                
                frame = frame[y1:y2, x1:x2]
                full_frames.append(frame)
        
        print(f"Number of frames: {len(full_frames)}")
        return full_frames, fps
    
    def prepare_audio(self) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Extract and process audio"""
        # Extract audio to WAV if needed
        if not self.args.audio.endswith('.wav'):
            print('Extracting audio to WAV...')
            os.makedirs('temp', exist_ok=True)
            command = f'ffmpeg -y -i "{self.args.audio}" -acodec pcm_s16le -ar 16000 temp/temp.wav'
            subprocess.call(command, shell=True)
            audio_path = 'temp/temp.wav'
        else:
            audio_path = self.args.audio
        
        # Process audio
        print('Processing audio...')
        mel = self.audio_processor.extract_audio_features(audio_path)
        print(f"Mel spectrogram shape: {mel.shape}")
        
        if np.isnan(mel).any():
            raise ValueError(
                'Mel spectrogram contains NaN values! '
                'If using TTS, add small noise to WAV file'
            )
        
        return mel, audio_path
    
    def run(self):
        """Run inference"""
        # Load video frames
        full_frames, fps = self.load_video_frames()
        
        # Process audio
        mel, _ = self.prepare_audio()
        
        # Create mel chunks
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
        
        print(f"Number of mel chunks: {len(mel_chunks)}")
        
        # Adjust frames to match mel chunks
        full_frames = full_frames[:len(mel_chunks)]
        
        # Run inference
        batch_size = self.args.wav2lip_batch_size
        gen = self.datagen(full_frames.copy(), mel_chunks)
        
        model = None
        out = None
        
        for i, (img_batch, mel_batch, frames, coords) in enumerate(
            tqdm(gen, total=int(np.ceil(len(mel_chunks) / batch_size)), desc='Inference')
        ):
            if i == 0:
                model = self.load_model(self.args.checkpoint_path)
                print("Model loaded")
                
                frame_h, frame_w = full_frames[0].shape[:-1]
                os.makedirs('temp', exist_ok=True)
                out = cv2.VideoWriter(
                    'temp/result.avi',
                    cv2.VideoWriter_fourcc(*'DIVX'),
                    fps,
                    (frame_w, frame_h)
                )
            
            # Convert to tensors
            img_batch = torch.FloatTensor(
                np.transpose(img_batch, (0, 3, 1, 2))
            ).to(self.device)
            
            mel_batch = torch.FloatTensor(
                np.transpose(mel_batch, (0, 3, 1, 2))
            ).to(self.device)
            
            # Inference
            with torch.no_grad():
                pred = model(mel_batch, img_batch)
            
            pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.
            
            # Write frames
            for p, f, c in zip(pred, frames, coords):
                y1, y2, x1, x2 = c
                p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
                f[y1:y2, x1:x2] = p
                out.write(f)
        
        out.release()
        
        # Combine audio and video
        print('Combining audio and video...')
        os.makedirs(os.path.dirname(self.args.outfile) or '.', exist_ok=True)
        command = f'ffmpeg -y -i "{self.args.audio}" -i temp/result.avi -strict -2 -q:v 1 "{self.args.outfile}"'
        subprocess.call(command, shell=platform.system() != 'Windows')
        
        print(f'Result saved to: {self.args.outfile}')


def main():
    args = parser.parse_args()
    inference = ModernLipsyncInference(args)
    inference.run()


if __name__ == '__main__':
    main()
