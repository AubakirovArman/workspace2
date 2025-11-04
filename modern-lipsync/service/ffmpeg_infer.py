from __future__ import annotations

from typing import List, Tuple, Optional, Callable
import os
import numpy as np
import cv2
import torch
import time


class FfmpegInferMixin:
    wav2lip_batch_size: int
    img_size: int
    use_fp16: bool
    device: str

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
        batch_size_override: Optional[int] = None,
        return_frames: bool = False,
        frame_offset: int = 0,
    ) -> Optional[List[np.ndarray]]:
        """Инференс Wav2Lip и запись кадров в видео через FFmpeg."""
        if not full_frames:
            raise ValueError("No video frames provided for inference")

        frame_h, frame_w = full_frames[0].shape[:-1]
        temp_dir = os.path.join(os.path.dirname(__file__), '..', 'temp')
        temp_dir = os.path.abspath(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)

        proc = self._start_writer(output_path, fps, frame_w, frame_h)

        batch_size = batch_size_override or self.wav2lip_batch_size

        total_chunks = len(mel_chunks)
        total_batches = (total_chunks + batch_size - 1) // batch_size
        print(f"ℹ️ Wav2Lip inference: {total_chunks} mel chunks, batch_size={batch_size}, batches={total_batches}")
        print("   ⏳ Подготовка входных тензоров...")

        device = torch.device(self.device)
        use_amp = self.use_fp16 and device.type == 'cuda'
        amp_dtype = getattr(self, 'amp_dtype', torch.float16)
        target_dtype = amp_dtype if use_amp else torch.float32
        non_blocking = device.type == 'cuda'

        precompute_start = time.time()

        if total_chunks == 0:
            raise ValueError("No mel chunks provided for inference")

        mel_np = np.asarray(mel_chunks, dtype=np.float32)
        if mel_np.ndim != 3:
            raise ValueError(f"Unexpected mel_chunks shape: {mel_np.shape}")
        mel_tensor = torch.from_numpy(mel_np).unsqueeze(1).to(
            device=device,
            dtype=target_dtype if use_amp else torch.float32,
            non_blocking=non_blocking,
        )

        prepared_face_results = face_det_results if static else face_det_results[:len(full_frames)]
        if not prepared_face_results:
            raise ValueError("No face detection results available for inference")

        face_coords_list: List[Tuple[int, int, int, int]]
        if static:
            face_coords_list = [prepared_face_results[0][1]]
        else:
            face_coords_list = [coords for _, coords in prepared_face_results]

        face_tensors: List[torch.Tensor] = []
        faces_iterable = prepared_face_results if not static else prepared_face_results[:1]
        for face_img, _coords in faces_iterable:
            if static and static_face_resized is not None:
                face_resized = static_face_resized
            else:
                face_resized = cv2.resize(face_img, (self.img_size, self.img_size))
            face_arr = face_resized.astype(np.float32) / 255.0
            face_tensor_cpu = torch.from_numpy(face_arr).permute(2, 0, 1).contiguous()
            mask_tensor = face_tensor_cpu.clone()
            mask_tensor[:, self.img_size // 2 :, :] = 0
            combined = torch.cat((mask_tensor, face_tensor_cpu), dim=0)
            face_tensors.append(combined)
        face_stack = torch.stack(face_tensors, dim=0)
        face_tensor = face_stack.to(
            device=device,
            dtype=target_dtype if use_amp else torch.float32,
            non_blocking=non_blocking,
        )

        precompute_time = time.time() - precompute_start
        print(
            f"   ✅ Тензоры готовы: frames={len(full_frames)}, faces={face_tensor.shape[0]}, mel={mel_tensor.shape[0]}"
        )

        batch_prep_time = 0.0
        tensor_time = 0.0
        forward_time = 0.0
        postprocess_time = 0.0
        paste_time = 0.0
        write_time = 0.0
        ffmpeg_time = 0.0

        collected_frames: List[np.ndarray] = []

        frame_count = len(full_frames)
        face_count = face_tensor.shape[0]

        frame_mod = max(1, frame_count)
        face_mod = max(1, face_count)
        offset_base = 0 if static else int(frame_offset) % frame_mod

        print(f"   ▶️ Запуск батчей: {total_batches} (max size {batch_size})")

        for batch_idx, i in enumerate(range(0, total_chunks, batch_size), start=1):
            end_idx = min(i + batch_size, total_chunks)
            current_batch = end_idx - i

            start_batch = time.time()
            frame_indices: List[int] = []
            face_indices: List[int] = []

            print(f"   ▶️ Батч {batch_idx}/{total_batches}: кадры {current_batch}")

            for offset in range(current_batch):
                frame_idx = 0 if static else (offset_base + i + offset) % frame_mod
                frame_indices.append(frame_idx)
                if static:
                    face_idx = 0
                else:
                    if face_count == frame_count:
                        face_idx = frame_idx
                    else:
                        face_idx = frame_idx % face_mod
                face_indices.append(face_idx)

            frames_batch = [full_frames[idx].copy() for idx in frame_indices]
            coords_batch = [face_coords_list[idx] for idx in face_indices]

            batch_prep_time += time.time() - start_batch

            start_tensor = time.time()
            mel_batch_tensor = mel_tensor[i:end_idx]
            index_tensor = torch.as_tensor(face_indices, device=face_tensor.device, dtype=torch.long)
            img_batch_tensor = face_tensor.index_select(0, index_tensor)
            tensor_time += time.time() - start_tensor

            start_forward = time.time()
            with torch.no_grad():
                if use_amp:
                    with torch.amp.autocast('cuda', dtype=amp_dtype):
                        pred = self.model(mel_batch_tensor, img_batch_tensor)
                else:
                    pred = self.model(mel_batch_tensor, img_batch_tensor)
            forward_time += time.time() - start_forward

            start_post = time.time()
            pred = pred.detach().cpu().numpy().transpose(0, 2, 3, 1) * 255.0
            postprocess_time += time.time() - start_post

            for predicted_patch, frame, coords in zip(pred, frames_batch, coords_batch):
                start_paste = time.time()
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
                paste_time += time.time() - start_paste

                start_write = time.time()
                self._write_frame(proc, frame)

                if frame_sink is not None:
                    frame_sink(frame)
                if return_frames:
                    collected_frames.append(frame.copy())
                write_time += time.time() - start_write

        if proc is not None:
            start_ffmpeg = time.time()
            returncode, error_output = self._finish_writer(proc)
            if returncode != 0:
                raise RuntimeError(f"FFmpeg failed with code {returncode}: {error_output}")
            ffmpeg_time += time.time() - start_ffmpeg

        print("\n📊 Inference Breakdown:")
        print(f"   Precompute Tensors: {precompute_time:.2f}s")
        print(f"   Batch Preparation: {batch_prep_time:.2f}s")
        print(f"   Tensor Conversion: {tensor_time:.2f}s")
        print(f"   Model Forward: {forward_time:.2f}s")
        print(f"   Postprocessing: {postprocess_time:.2f}s")
        print(f"   Pasting Patches: {paste_time:.2f}s")
        print(f"   Writing Frames: {write_time:.2f}s")
        print(f"   FFmpeg Encoding: {ffmpeg_time:.2f}s")

        self._last_inference_breakdown = {
            'precompute_tensors_time': precompute_time,
            'batch_preparation_time': batch_prep_time,
            'tensor_conversion_time': tensor_time,
            'model_forward_time': forward_time,
            'postprocessing_time': postprocess_time,
            'pasting_patches_time': paste_time,
            'writing_frames_time': write_time,
            'ffmpeg_encoding_time': ffmpeg_time,
        }

        if return_frames:
            return collected_frames
        return None
