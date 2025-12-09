"""Utility functions for seeding all relevant random number generators.

This helps to make experiments more reproducible by setting seeds for:
- Python's built-in `random`
- NumPy
- PyTorch CPU & CUDA (if available)
- Optional CuDNN deterministic behavior
"""
from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True, cuda: Optional[bool] = None) -> None:
    """Seed Python, NumPy and PyTorch (CPU & CUDA) for reproducibility.

    Parameters
    ----------
    seed : int
        The random seed to use.
    deterministic : bool, default True
        If True, configures CuDNN for deterministic behavior
        (slower but reproducible). If False, leaves CuDNN settings unchanged.
    cuda : Optional[bool]
        If True, force-enable CUDA seeding (if torch.cuda.is_available()).
        If False, skip CUDA seeding entirely. If None (default), only seed
        CUDA when it is available.
    """

    # Basic environment-level seeds
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Python & NumPy
    random.seed(seed)
    np.random.seed(seed)

    # PyTorch CPU
    torch.manual_seed(seed)

    # PyTorch CUDA (if requested / available)
    if cuda is None:
        use_cuda = torch.cuda.is_available()
    else:
        use_cuda = cuda and torch.cuda.is_available()

    if use_cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # CuDNN deterministic behavior (optional)
    if deterministic and torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
