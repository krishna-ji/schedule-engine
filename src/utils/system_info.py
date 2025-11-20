"""System information utilities for resource detection."""

import os
import logging

logger = logging.getLogger(__name__)

DEFAULT_CPU_COUNT = 8  # Fallback if detection fails


def get_cpu_count() -> int:
    """Get number of available CPU cores with proper fallback.

    Returns:
        Number of logical CPU cores (including hyperthreading)
    """
    try:
        count = os.cpu_count()
        if count is None or count < 1:
            logger.warning(
                f"CPU detection returned invalid count: {count}, using default {DEFAULT_CPU_COUNT}"
            )
            return DEFAULT_CPU_COUNT
        return count
    except Exception as e:
        logger.error(
            f"Failed to detect CPU count: {e}, using default {DEFAULT_CPU_COUNT}"
        )
        return DEFAULT_CPU_COUNT


def get_gpu_info():
    """Get GPU availability and information.

    Returns:
        Tuple of (is_available, device_name, memory_gb)
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return False, "No CUDA GPU", 0

        device_name = torch.cuda.get_device_name(0)
        memory_bytes = torch.cuda.get_device_properties(0).total_memory
        memory_gb = memory_bytes // (1024**3)

        return True, device_name, memory_gb
    except Exception as e:
        logger.error(f"Failed to detect GPU: {e}")
        return False, f"Error: {e}", 0


def diagnose_system():
    """Get comprehensive system information for diagnostics.

    Returns:
        Dictionary with system info
    """
    cpu_count = get_cpu_count()
    gpu_available, gpu_name, gpu_memory = get_gpu_info()

    try:
        import torch

        pytorch_version = torch.__version__
        cuda_version = torch.version.cuda if torch.cuda.is_available() else "N/A"
    except ImportError:
        pytorch_version = "Not installed"
        cuda_version = "N/A"

    info = {
        "cpu_cores": cpu_count,
        "gpu_available": gpu_available,
        "gpu_name": gpu_name,
        "gpu_memory_gb": gpu_memory,
        "pytorch_version": pytorch_version,
        "cuda_version": cuda_version,
    }

    return info


def print_system_diagnostics():
    """Print formatted system diagnostics."""
    info = diagnose_system()

    print("\n=== System Diagnostics ===")
    print(f"CPU Cores: {info['cpu_cores']}")
    print(f"PyTorch Version: {info['pytorch_version']}")
    print(f"\nGPU Status:")
    print(f"  Available: {info['gpu_available']}")
    print(f"  Device: {info['gpu_name']}")
    if info["gpu_available"]:
        print(f"  Memory: {info['gpu_memory_gb']} GB")
        print(f"  CUDA Version: {info['cuda_version']}")
    print("=" * 30 + "\n")
