"""System information utilities for resource detection."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import TypedDict

from rich.console import Console
from rich.text import Text

console: Console = Console()
logger = logging.getLogger(__name__)

DEFAULT_CPU_COUNT = 8  # Fallback if detection fails


@dataclass(frozen=True)
class GPUInfo:
    """Simple structured GPU information container."""

    available: bool
    name: str
    memory_gb: int


class SystemDiagnostics(TypedDict):
    """Dictionary layout returned by :func:`diagnose_system`."""

    cpu_cores: int
    gpu_available: bool
    gpu_name: str
    gpu_memory_gb: int
    pytorch_version: str
    cuda_version: str


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


def get_gpu_info() -> GPUInfo:
    """Get GPU availability and information."""
    try:
        import torch

        if not torch.cuda.is_available():
            return GPUInfo(False, "No CUDA GPU", 0)

        device_name = torch.cuda.get_device_name(0)
        memory_bytes = torch.cuda.get_device_properties(0).total_memory
        memory_gb = memory_bytes // (1024**3)

        return GPUInfo(True, device_name, memory_gb)
    except Exception as e:
        logger.error(f"Failed to detect GPU: {e}")
        return GPUInfo(False, f"Error: {e}", 0)


def diagnose_system() -> SystemDiagnostics:
    """Get comprehensive system information for diagnostics."""
    cpu_count = get_cpu_count()
    gpu_info = get_gpu_info()

    try:
        import torch

        pytorch_version: str = torch.__version__
        cuda_attr = torch.version.cuda if torch.cuda.is_available() else "N/A"
        cuda_version = cuda_attr or "N/A"
    except ImportError:
        pytorch_version = "Not installed"
        cuda_version = "N/A"

    info: SystemDiagnostics = {
        "cpu_cores": cpu_count,
        "gpu_available": gpu_info.available,
        "gpu_name": gpu_info.name,
        "gpu_memory_gb": gpu_info.memory_gb,
        "pytorch_version": pytorch_version,
        "cuda_version": cuda_version,
    }

    return info


def print_system_diagnostics(sep: str = " . ") -> None:
    """Print formatted system diagnostics in a single line using Rich.

    The output is printed on a single line and fields are separated using
    the provided `sep` string (default is `" . "`). This makes the
    diagnostics compact and easier to scan in logs.

    Args:
        sep: separator string inserted between fields (default: " . ")
    """

    # keep `sep` configurable so callers can pick '.' or a different divider
    def _print_single_line(inner_sep: str = " . ") -> None:
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
        line = Text(inner_sep).join(Text(p, style="bold cyan") for p in parts)

        # prepend a short heading for clarity
        console.print(Text("System Diagnostics:", style="bold magenta"), line)

    # print with the supplied separator
    _print_single_line(sep)
