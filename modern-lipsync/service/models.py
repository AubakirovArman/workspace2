from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Optional

import torch
import face_detection

from models import Wav2Lip
from utils.audio import ModernAudioProcessor


class ModelsMixin:
    """Загрузка основных и дополнительных моделей."""

    def _ensure_functional_tensor_stub(self) -> None:
        """Provide compatibility shim for older basicsr imports."""
        target_name = 'torchvision.transforms.functional_tensor'
        if target_name in sys.modules:
            return
        try:
            from torchvision.transforms import functional as _F
        except Exception:
            return

        module = types.ModuleType(target_name)
        if hasattr(_F, 'rgb_to_grayscale'):
            module.rgb_to_grayscale = _F.rgb_to_grayscale  # type: ignore[attr-defined]
        sys.modules[target_name] = module

    def _ensure_modules_root(self):
        if getattr(self, "_modules_root", None) and not getattr(self, "_sys_path_added", False):
            if self._modules_root not in sys.path:
                sys.path.insert(0, self._modules_root)
            self._sys_path_added = True

    def _load_models(self, checkpoint_path: str):
        """Load Wav2Lip, face detector и аудио процессор."""
        print("Loading Wav2Lip model...")
        self.model = None
        self._is_torchscript = False
        is_cuda = getattr(self, "is_cuda", str(self.device).startswith('cuda'))

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

        # Mixed precision (FP16/BF16)
        if getattr(self, "use_fp16", False) and is_cuda:
            try:
                target_dtype = getattr(self, 'amp_dtype', torch.float16)
                self.model = self.model.to(dtype=target_dtype)
                amp_name = 'FP16' if target_dtype == torch.float16 else 'BF16'
                print(f"   ✓ Model converted to {amp_name}")
            except Exception as fp16_err:
                print(f"   ⚠️ AMP conversion failed ({fp16_err}), using FP32")
                self.use_fp16 = False

        # torch.compile
        if getattr(self, "use_compile", False) and not self._is_torchscript:
            try:
                print("   ⏳ Compiling model with torch.compile...")
                self.model = torch.compile(self.model, mode='max-autotune')
                print("   ✓ Model compiled with torch.compile")
            except Exception as compile_err:
                print(f"   ⚠️ torch.compile failed ({compile_err}), using eager mode")
                self.use_compile = False

        # Face detector
        print("Loading face detector...")
        self.face_detector = face_detection.FaceAlignment(
            face_detection.LandmarksType._2D,
            flip_input=False,
            device=self.device
        )

        # Audio processor
        print("Loading audio processor...")
        self.audio_processor = ModernAudioProcessor()

        # Optional modules
        self._load_optional_modules()

    def _load_optional_modules(self):
        self._ensure_functional_tensor_stub()
        real_sr_loaded = False
        is_cuda = getattr(self, "is_cuda", str(self.device).startswith('cuda'))
        segmentation_path = getattr(self, "_segmentation_path", None)
        sr_path = getattr(self, "_sr_path", None)
        realesrgan_path = getattr(self, "_realesrgan_path", None)

        if segmentation_path and Path(segmentation_path).is_file():
            if not is_cuda:
                print("   ⚠️ Segmentation requires CUDA; skipping")
            else:
                self._ensure_modules_root()
                try:
                    from face_parsing import init_parser, swap_regions
                    print("Loading segmentation model...")
                    self.segmentation_model = init_parser(segmentation_path)
                    self._swap_regions_fn = swap_regions
                    self.segmentation_enabled = True
                    print("   ✓ Segmentation model ready")
                except Exception as seg_err:
                    print(f"   ⚠️ Failed to load segmentation model ({seg_err})")
                    self.segmentation_model = None
                    self._swap_regions_fn = None
                    self.segmentation_enabled = False

        if realesrgan_path and Path(realesrgan_path).is_file():
            if not is_cuda:
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
                        model_path=realesrgan_path,
                        model=rrdbnet,
                        tile=0,
                        tile_pad=10,
                        pre_pad=0,
                        half=is_cuda
                    )
                    self.realesrgan_enabled = True
                    real_sr_loaded = True
                    print("   ✓ Real-ESRGAN model ready")
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

        if (not real_sr_loaded) and sr_path and Path(sr_path).is_file():
            if not is_cuda:
                print("   ⚠️ Super-resolution requires CUDA; skipping")
            else:
                self._ensure_modules_root()
                self._ensure_functional_tensor_stub()
                try:
                    from basicsr.apply_sr import init_sr_model, enhance  # type: ignore
                except ImportError:
                    from .sr_compat import init_sr_model, enhance
                    print("Loading super-resolution model (ESRGAN)...")
                    self.sr_model = init_sr_model(sr_path)
                    self._sr_enhance_fn = enhance
                    self.sr_enabled = True
                    print("   ✓ Super-resolution model ready")
                except Exception as sr_err:
                    print(f"   ⚠️ Failed to load super-resolution model ({sr_err})")
                    self.sr_model = None
                    self._sr_enhance_fn = None
                    self.sr_enabled = False
