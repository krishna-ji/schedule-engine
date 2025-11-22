"""System information utilities for resource detection."""

from rich.console import Console
from rich.text import Text
import os
import logging

console = Console()
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


def print_system_diagnostics(sep: str = " . "):
    """Print formatted system diagnostics in a single line using Rich.

    The output is printed on a single line and fields are separated using
    the provided `sep` string (default is `" . "`). This makes the
    diagnostics compact and easier to scan in logs.

    Args:
        sep: separator string inserted between fields (default: " . ")
    """

    # keep `sep` configurable so callers can pick '.' or a different divider
    def _print_single_line(sep: str = " . ") -> None:
        info = diagnose_system()

        parts = [
            f"CPU Cores: {info['cpu_cores']}",
            f"PyTorch: {info['pytorch_version']}",
            f"GPU Available: {info['gpu_available']}",
            f"GPU: {info['gpu_name']}",
        ]

        if info["gpu_available"]:
            parts.append(f"GPU Memory: {info['gpu_memory_gb']} GB")
            parts.append(f"CUDA: {info['cuda_version']}")

        # Build a single Text object and print via the rich Console
        line = Text(sep).join(Text(p, style="bold cyan") for p in parts)

        # prepend a short heading for clarity
        console.print(Text("System Diagnostics:", style="bold magenta"), line)

    # print with the supplied separator
    _print_single_line(sep)
