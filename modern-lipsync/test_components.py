#!/usr/bin/env python3
"""
Test script for Modern Lipsync
Checks if all components are working correctly
"""
import sys
import torch
import torchaudio

def test_imports():
    """Test if all required modules can be imported"""
    print("🧪 Testing imports...")
    
    try:
        import numpy as np
        print("✓ numpy")
    except ImportError as e:
        print(f"✗ numpy: {e}")
        return False
    
    try:
        import cv2
        print("✓ opencv")
    except ImportError as e:
        print(f"✗ opencv: {e}")
        return False
    
    try:
        from models import Wav2Lip, Conv2d
        print("✓ models")
    except ImportError as e:
        print(f"✗ models: {e}")
        return False
    
    try:
        from utils.audio import ModernAudioProcessor
        print("✓ audio utils")
    except ImportError as e:
        print(f"✗ audio utils: {e}")
        return False
    
    try:
        import face_detection
        print("✓ face_detection")
    except ImportError as e:
        print(f"✗ face_detection: {e}")
        return False
    
    return True


def test_pytorch():
    """Test PyTorch and CUDA"""
    print("\n🔥 Testing PyTorch...")
    
    print(f"✓ PyTorch version: {torch.__version__}")
    print(f"✓ torchaudio version: {torchaudio.__version__}")
    
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"✓ CUDA version: {torch.version.cuda}")
    else:
        print("⚠ CUDA not available, will use CPU")
    
    return True


def test_model():
    """Test if Wav2Lip model can be created"""
    print("\n🤖 Testing model creation...")
    
    try:
        from models import Wav2Lip
        model = Wav2Lip()
        print(f"✓ Wav2Lip model created")
        
        # Count parameters
        params = sum(p.numel() for p in model.parameters())
        print(f"✓ Model parameters: {params:,}")
        
        return True
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return False


def test_audio_processor():
    """Test audio processor"""
    print("\n🎵 Testing audio processor...")
    
    try:
        from utils.audio import ModernAudioProcessor, AudioConfig
        
        config = AudioConfig()
        processor = ModernAudioProcessor(config)
        print("✓ Audio processor created")
        print(f"✓ Sample rate: {config.sample_rate}")
        print(f"✓ N_mels: {config.n_mels}")
        
        return True
    except Exception as e:
        print(f"✗ Audio processor failed: {e}")
        return False


def main():
    """Run all tests"""
    print("=" * 50)
    print("Modern Lipsync - Component Test")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_pytorch,
        test_model,
        test_audio_processor,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"\n✗ Test failed with exception: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    if all(results):
        print("✅ All tests passed!")
        print("=" * 50)
        return 0
    else:
        print("❌ Some tests failed")
        print("=" * 50)
        return 1


if __name__ == '__main__':
    sys.exit(main())
