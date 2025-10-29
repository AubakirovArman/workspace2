"""
Modern Lipsync - Audio Processing with torchaudio 2.8.0
"""
import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np
from scipy import signal
from typing import Union, Tuple


class AudioConfig:
    """Audio processing configuration"""
    def __init__(self):
        self.sample_rate = 16000
        self.n_fft = 800
        self.hop_length = 200
        self.win_length = 800
        self.n_mels = 80
        self.f_min = 55
        self.f_max = 7600
        self.preemphasis = 0.97
        self.ref_level_db = 20
        self.min_level_db = -100
        self.max_abs_value = 4.0
        self.symmetric_mels = True
        self.allow_clipping = True


class ModernAudioProcessor:
    """
    Modern audio processor using torchaudio 2.8.0
    Replaces librosa-based processing with native PyTorch operations
    """
    def __init__(self, config: AudioConfig = None):
        self.config = config or AudioConfig()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Initialize mel spectrogram transform
        self.mel_transform = T.MelSpectrogram(
            sample_rate=self.config.sample_rate,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.win_length,
            n_mels=self.config.n_mels,
            f_min=self.config.f_min,
            f_max=self.config.f_max,
            power=1.0,  # magnitude spectrogram
        ).to(self.device)
        
        # Amplitude to DB conversion
        self.amplitude_to_db = T.AmplitudeToDB(
            stype='magnitude',
            top_db=80
        ).to(self.device)
    
    def load_audio(self, path: str, target_sr: int = None) -> Tuple[torch.Tensor, int]:
        """
        Load audio file using torchaudio
        Args:
            path: Path to audio file
            target_sr: Target sample rate (defaults to config sample rate)
        Returns:
            waveform: Audio tensor (1, num_samples)
            sample_rate: Sample rate
        """
        target_sr = target_sr or self.config.sample_rate
        
        # Load audio
        waveform, sr = torchaudio.load(path)
        
        # Resample if needed
        if sr != target_sr:
            resampler = T.Resample(sr, target_sr).to(self.device)
            waveform = resampler(waveform.to(self.device))
        else:
            waveform = waveform.to(self.device)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        return waveform, target_sr
    
    def save_audio(self, waveform: torch.Tensor, path: str, sample_rate: int = None):
        """
        Save audio file using torchaudio
        Args:
            waveform: Audio tensor
            path: Output path
            sample_rate: Sample rate
        """
        sample_rate = sample_rate or self.config.sample_rate
        
        # Ensure waveform is on CPU
        if waveform.is_cuda:
            waveform = waveform.cpu()
        
        # Normalize to int16 range
        waveform = waveform / torch.max(torch.abs(waveform).max(), torch.tensor(0.01))
        
        torchaudio.save(path, waveform, sample_rate)
    
    def preemphasis(self, wav: torch.Tensor, coeff: float = None) -> torch.Tensor:
        """
        Apply preemphasis filter
        Args:
            wav: Input waveform (1, num_samples)
            coeff: Preemphasis coefficient
        Returns:
            Filtered waveform
        """
        coeff = coeff or self.config.preemphasis
        
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        
        # Apply preemphasis filter: y[n] = x[n] - coeff * x[n-1]
        return torch.cat([
            wav[:, :1],
            wav[:, 1:] - coeff * wav[:, :-1]
        ], dim=1)
    
    def melspectrogram(self, wav: Union[torch.Tensor, str], normalize: bool = True) -> np.ndarray:
        """
        Compute mel spectrogram
        Args:
            wav: Input waveform tensor or path to audio file
            normalize: Whether to normalize the spectrogram
        Returns:
            Mel spectrogram as numpy array
        """
        # Load audio if path provided
        if isinstance(wav, str):
            wav, _ = self.load_audio(wav)
        
        # Ensure tensor is on correct device
        wav = wav.to(self.device)
        
        # Apply preemphasis
        wav = self.preemphasis(wav)
        
        # Compute mel spectrogram
        mel_spec = self.mel_transform(wav)
        
        # Convert to dB scale
        mel_spec_db = self.amplitude_to_db(mel_spec)
        
        # Adjust reference level
        mel_spec_db = mel_spec_db - self.config.ref_level_db
        
        # Normalize if requested
        if normalize:
            mel_spec_db = self._normalize(mel_spec_db)
        
        # Convert to numpy and remove batch dimension
        return mel_spec_db.squeeze(0).cpu().numpy()
    
    def _normalize(self, S: torch.Tensor) -> torch.Tensor:
        """
        Normalize spectrogram
        Args:
            S: Input spectrogram
        Returns:
            Normalized spectrogram
        """
        if self.config.symmetric_mels:
            # Scale to [-max_abs_value, max_abs_value]
            normalized = (2 * self.config.max_abs_value) * \
                        ((S - self.config.min_level_db) / (-self.config.min_level_db)) - \
                        self.config.max_abs_value
            
            if self.config.allow_clipping:
                normalized = torch.clamp(normalized, 
                                       -self.config.max_abs_value, 
                                       self.config.max_abs_value)
        else:
            # Scale to [0, max_abs_value]
            normalized = self.config.max_abs_value * \
                        ((S - self.config.min_level_db) / (-self.config.min_level_db))
            
            if self.config.allow_clipping:
                normalized = torch.clamp(normalized, 0, self.config.max_abs_value)
        
        return normalized
    
    def extract_audio_features(self, audio_path: str) -> np.ndarray:
        """
        Extract mel spectrogram features from audio file
        This is the main function used for inference
        Args:
            audio_path: Path to audio file
        Returns:
            Mel spectrogram features (num_mels, num_frames)
        """
        return self.melspectrogram(audio_path, normalize=True)


# Convenience functions for backward compatibility
def load_wav(path: str, sr: int = 16000) -> np.ndarray:
    """Load audio file (backward compatible with librosa)"""
    processor = ModernAudioProcessor()
    waveform, _ = processor.load_audio(path, sr)
    return waveform.squeeze(0).cpu().numpy()


def save_wav(wav: np.ndarray, path: str, sr: int = 16000):
    """Save audio file (backward compatible with librosa)"""
    processor = ModernAudioProcessor()
    if isinstance(wav, np.ndarray):
        wav = torch.from_numpy(wav).unsqueeze(0)
    processor.save_audio(wav, path, sr)


def melspectrogram(wav: Union[torch.Tensor, np.ndarray, str]) -> np.ndarray:
    """Compute mel spectrogram (backward compatible)"""
    processor = ModernAudioProcessor()
    if isinstance(wav, np.ndarray):
        wav = torch.from_numpy(wav).unsqueeze(0)
    return processor.melspectrogram(wav, normalize=True)
