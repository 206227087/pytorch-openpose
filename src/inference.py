"""Inference optimization utilities for OpenPose and HRNet models.

Provides TensorRT compilation, mixed-precision inference, and
model profiling tools.
"""

import time

import torch
import torch.nn as nn


# ─── TensorRT Compilation ─────────────────────────────────────────────────────

def compile_tensorrt(model, input_shape=(1, 3, 368, 368),
                     enabled_precisions=None, dynamic_batch=False):
    """Compile a PyTorch model to TensorRT for accelerated inference.

    Requires torch_tensorrt to be installed. Falls back to the original
    model if TensorRT is not available.

    Args:
        model: nn.Module in eval mode.
        input_shape: expected input tensor shape (default (1, 3, 368, 368)).
        enabled_precisions: set of torch dtypes (default {torch.float, torch.half}).
        dynamic_batch: enable dynamic batch size (default False).

    Returns:
        Compiled TensorRT model, or original model if TensorRT unavailable.
    """
    if enabled_precisions is None:
        enabled_precisions = {torch.float, torch.half}

    try:
        import torch_tensorrt

        if dynamic_batch:
            inputs = [torch_tensorrt.Input(
                min_shape=[1, *input_shape[1:]],
                opt_shape=[4, *input_shape[1:]],
                max_shape=[16, *input_shape[1:]],
            )]
        else:
            inputs = [torch_tensorrt.Input(input_shape)]

        trt_model = torch_tensorrt.compile(
            model,
            inputs=inputs,
            enabled_precisions=enabled_precisions,
        )
        print(f"TensorRT compilation successful (input: {input_shape})")
        return trt_model

    except ImportError:
        print("torch_tensorrt not installed, returning original model")
        return model
    except Exception as e:
        print(f"TensorRT compilation failed: {e}, returning original model")
        return model


# ─── Mixed Precision Inference ────────────────────────────────────────────────

class MixedPrecisionInference:
    """Wrapper for automatic mixed-precision (AMP) inference.

    Uses torch.amp.autocast to run the model in float16 where possible,
    reducing memory usage and increasing throughput on compatible GPUs.

    Args:
        model: nn.Module in eval mode.
        device: compute device (default 'cuda').
    """

    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()

    @torch.inference_mode()
    def __call__(self, x):
        """Run inference with automatic mixed precision.

        Args:
            x: input tensor (already on device).

        Returns:
            Model output in float32.
        """
        with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
            return self.model(x)


# ─── Model Profiling ─────────────────────────────────────────────────────────

def profile_model(model, input_shape=(1, 3, 368, 368), device='cuda',
                  warmup=10, repeats=100):
    """Profile model inference speed and memory usage.

    Args:
        model: nn.Module in eval mode.
        input_shape: input tensor shape.
        device: compute device.
        warmup: number of warmup iterations (default 10).
        repeats: number of timed iterations (default 100).

    Returns:
        Dict with 'avg_ms', 'fps', 'params_M', 'flops_estimate' keys.
    """
    model.to(device)
    model.eval()

    # Parameter count
    params = sum(p.numel() for p in model.parameters()) / 1e6  # millions

    # Create dummy input
    x = torch.randn(*input_shape, device=device)

    # Warmup
    for _ in range(warmup):
        with torch.inference_mode():
            _ = model(x)

    # Timed runs
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(repeats):
        with torch.inference_mode():
            _ = model(x)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    avg_ms = (elapsed / repeats) * 1000
    fps = repeats / elapsed

    # Memory usage
    mem_mb = 0
    if torch.cuda.is_available():
        mem_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

    return {
        'avg_ms': avg_ms,
        'fps': fps,
        'params_M': params,
        'gpu_mem_MB': mem_mb,
    }


def optimize_for_inference(model):
    """Apply PyTorch inference optimizations to a model.

    Sets eval mode and applies torch.compile if available (PyTorch 2.0+).

    Args:
        model: nn.Module.

    Returns:
        Optimized model.
    """
    model.eval()

    # Try torch.compile (PyTorch 2.0+)
    try:
        model = torch.compile(model, mode='reduce-overhead')
        print("torch.compile optimization applied")
    except Exception:
        print("torch.compile not available, using standard eval mode")

    return model
