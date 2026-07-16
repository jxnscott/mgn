import random
import numpy as np
import torch
import tensorflow as tf


def seed_everything(*, seed: int) -> None:
    """Seeds every RNG this project touches, for reproducible runs.

    Covers Python's ``random``, NumPy, PyTorch (CPU and, if available,
    CUDA), and TensorFlow. When CUDA is available, also disables cuDNN's
    autotuning (``benchmark = False``) and forces deterministic algorithm
    selection (``deterministic = True``), since cuDNN convolutions can be
    non-deterministic even with a fixed seed otherwise — this costs some
    performance in exchange for reproducible GPU training.

    Args:
        seed: The seed value to apply to every RNG.
    """
    # Python
    random.seed(seed)

    # numpy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # TensorFlow
    tf.random.set_seed(seed)
