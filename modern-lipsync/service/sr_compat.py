"""Fallback helpers for ESRGAN super-resolution when basicsr.apply_sr is unavailable."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch

try:
    from basicsr.archs.rrdbnet_arch import RRDBNet  # type: ignore
except ImportError as exc:  # pragma: no cover - should be installed alongside basicsr
    raise RuntimeError("RRDBNet architecture not available; install basicsr.") from exc


def _select_state_dict(raw_state: object) -> dict:
    if not isinstance(raw_state, dict):
        raise TypeError("Unexpected ESRGAN checkpoint format")
    for key in ("params_ema", "params", "state_dict", "model", "net" ):
        if key in raw_state:
            candidate = raw_state[key]
            if isinstance(candidate, dict):
                return candidate
    return raw_state  # assume raw dict with tensor weights


def init_sr_model(model_path: str | Path, device: Optional[str] = None) -> torch.nn.Module:
    resolved = Path(model_path)
    if not resolved.is_file():
        raise FileNotFoundError(f"ESRGAN checkpoint not found: {resolved}")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_device = torch.device(device)

    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32)

    raw_state = torch.load(str(resolved), map_location=torch_device)
    state = _select_state_dict(raw_state)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"   ⚠️ ESRGAN checkpoint loaded with missing keys: {missing}, unexpected: {unexpected}")

    model.to(torch_device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def enhance(model: torch.nn.Module, image: np.ndarray) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected image with shape (H, W, 3)")

    device = next(model.parameters()).device
    tensor = torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0)
    tensor = tensor.to(device) / 255.0

    with torch.inference_mode():
        output = model(tensor)

    output = output.squeeze(0).clamp_(0.0, 1.0)
    result = (output.permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)
    return result
