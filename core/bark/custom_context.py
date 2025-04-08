import contextlib
import torch
import funcy


"""
Custom context managers for PyTorch inference operations.

This module provides context managers for controlling:
- CUDA benchmarking settings
- Inference mode and gradient calculation
- Automatic mixed precision (AMP) casting

The main context manager `inference_mode()` combines all these settings
for optimal inference performance.
"""


class InferenceContext:
    """
    Context manager for controlling CUDA benchmarking settings.

    Args:
        benchmark (bool): Whether to enable cudnn benchmarking. Defaults to False
            since input lengths may vary in inference scenarios.

    This context manager saves and restores the original cudnn.benchmark setting
    when entering/exiting the context.
    """

    def __init__(self, benchmark=False):
        # we can't expect inputs to be the same length, so disable benchmarking by default
        self._chosen_cudnn_benchmark = benchmark
        self._cudnn_benchmark = None

    def __enter__(self):
        self._cudnn_benchmark = torch.backends.cudnn.benchmark
        torch.backends.cudnn.benchmark = self._chosen_cudnn_benchmark

    def __exit__(self, exc_type, exc_value, exc_traceback):
        torch.backends.cudnn.benchmark = self._cudnn_benchmark


if (
    torch.cuda.is_available()
    and hasattr(torch.cuda, "amp")
    and hasattr(torch.cuda.amp, "autocast")
    and hasattr(torch.cuda, "is_bf16_supported")
    and torch.cuda.is_bf16_supported()
):
    autocast = funcy.partial(
        torch.amp.autocast, dtype=torch.bfloat16, device_type="cuda"
    )
    """Context manager for automatic mixed precision (AMP) using bfloat16 where supported."""
else:

    @contextlib.contextmanager
    def autocast():
        """No-op autocast context manager when bfloat16 is not supported."""
        yield


@contextlib.contextmanager
def inference_mode():
    """
    Combined context manager for optimal inference performance.

    Combines:
    - CUDA benchmarking control
    - PyTorch inference mode
    - Disabled gradient calculation
    - Automatic mixed precision casting (where supported)

    Usage:
        with inference_mode():
            # inference operations here
    """
    with InferenceContext(), torch.inference_mode(), torch.no_grad(), autocast():
        yield
