from __future__ import annotations

import os
import shutil
import wave
from typing import List, Tuple, Optional

import numpy as np
import torch
import torchaudio


class AudioMixin:
    mel_step_size: int

    def _process_audio(
        self,
        audio_path: str,
        fps: float,
        waveform: Optional[torch.Tensor] = None,
        sample_rate: int = 16000,
        write_temp_wav: bool = True,
    ) -> Tuple[np.ndarray, List]:
        """Преобразование аудио в mel-спектрограмму и нарезка чанков."""
        temp_dir = os.path.join(os.path.dirname(__file__), '..', 'temp')
        temp_dir = os.path.abspath(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)
        self.temp_wav = os.path.join(temp_dir, 'temp.wav')

        target_sr = self.audio_processor.config.sample_rate

        if waveform is not None:
            waveform = waveform.detach()
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            waveform = waveform.to(self.audio_processor.device).float()

            if sample_rate != target_sr:
                resampler = torchaudio.transforms.Resample(sample_rate, target_sr).to(self.audio_processor.device)
                waveform = resampler(waveform)
                sample_rate = target_sr
            # В режиме in-memory избегаем записи временного WAV
            if write_temp_wav:
                cpu_wave = waveform.detach().cpu()
                try:
                    torchaudio.save(self.temp_wav, cpu_wave, sample_rate)
                except Exception:
                    self._write_wave_manual(self.temp_wav, cpu_wave, sample_rate)
            mel = self.audio_processor.melspectrogram(waveform)
        else:
            if write_temp_wav:
                # Поддержка существующего пути: создаём temp_wav, чтобы ffmpeg-объединение могло использовать аудио
                if not audio_path.endswith('.wav'):
                    cmd = f'ffmpeg -y -i "{audio_path}" -acodec pcm_s16le -ar {target_sr} "{self.temp_wav}" -loglevel quiet'
                    os.system(cmd)
                else:
                    try:
                        shutil.copy2(audio_path, self.temp_wav)
                    except Exception:
                        pass
                mel = self.audio_processor.extract_audio_features(self.temp_wav)
            else:
                # Извлекаем мел-спектрограмму напрямую из файла без записи копии
                source_path = audio_path
                mel = self.audio_processor.extract_audio_features(source_path)

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

    @staticmethod
    def _write_wave_manual(path: str, tensor: torch.Tensor, sample_rate: int) -> None:
        array = tensor.detach().cpu().numpy()
        if array.ndim == 1:
            array = array[None, :]
        clipped = np.clip(array, -1.0, 1.0)
        pcm16 = (clipped * 32767.0).round().astype(np.int16)
        interleaved = pcm16.transpose(1, 0).reshape(-1)
        with wave.open(path, "wb") as wav_file:
            wav_file.setnchannels(pcm16.shape[0])
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(interleaved.tobytes())
