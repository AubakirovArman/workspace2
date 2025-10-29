"""
Modern Lipsync - Hyperparameters
Updated for modern PyTorch ecosystem
"""


class HParams:
    """Hyperparameters container"""
    def __init__(self, **kwargs):
        self.data = {}
        for key, value in kwargs.items():
            self.data[key] = value

    def __getattr__(self, key):
        if key not in self.data:
            raise AttributeError(f"'HParams' object has no attribute '{key}'")
        return self.data[key]

    def set_hparam(self, key, value):
        self.data[key] = value


# Default hyperparameters for Modern Lipsync
hparams = HParams(
    # Audio parameters
    num_mels=80,
    rescale=True,
    rescaling_max=0.9,
    
    # STFT parameters
    use_lws=False,
    n_fft=800,
    hop_size=200,
    win_size=800,
    sample_rate=16000,
    frame_shift_ms=None,
    
    # Mel spectrogram normalization
    signal_normalization=True,
    allow_clipping_in_normalization=True,
    symmetric_mels=True,
    max_abs_value=4.0,
    
    # Preemphasis
    preemphasize=True,
    preemphasis=0.97,
    
    # Audio limits
    min_level_db=-100,
    ref_level_db=20,
    fmin=55,
    fmax=7600,
    
    # Video parameters
    img_size=96,
    fps=25,
    
    # Training parameters (for reference, not used in inference)
    batch_size=16,
    initial_learning_rate=1e-4,
    num_workers=4,  # Reduced for modern systems
    
    # Checkpoint parameters
    checkpoint_interval=3000,
    eval_interval=3000,
    save_optimizer_state=True,
    
    # SyncNet parameters
    syncnet_wt=0.03,
    syncnet_batch_size=64,
    syncnet_lr=1e-4,
    
    # Discriminator parameters
    disc_wt=0.07,
    disc_initial_learning_rate=1e-4,
)


def hparams_debug_string():
    """Return a string with all hyperparameters"""
    values = hparams.data
    hp = [f"  {name}: {values[name]}" for name in sorted(values)]
    return "Hyperparameters:\n" + "\n".join(hp)
