import torch
from typing import Callable

from model_runner.util import gpipe_split_spec  # noqa: F401 — re-exported for baseline.py


def generate_batch(rand_inputs: Callable[[], torch.Tensor], batch_size: int, seed: int) -> torch.Tensor:
    """Generate a batch of inputs from a single seed.

    Sets the seed once, then calls rand_inputs() batch_size times and
    concatenates the results. Each sample is different because the RNG
    state advances between calls, but the full batch is deterministically
    reproducible from the seed.
    """
    torch.manual_seed(seed)
    return torch.cat([rand_inputs() for _ in range(batch_size)], dim=0)
