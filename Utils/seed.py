import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """固定随机种子，提升实验复现性。

    同时设置 Python random、NumPy、PyTorch CPU 和 PyTorch CUDA 的随机种子。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
