import os
import sys
import re

# Ensure repository root is on PYTHONPATH for test discovery/imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.utils import system_info


def _strip_ansi(text: str) -> str:
    # remove common ANSI escape sequences so tests can assert on plain text
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def test_print_single_line_default_separator(monkeypatch, capsys):
    monkeypatch.setattr(
        system_info,
        "diagnose_system",
        lambda: {
            "cpu_cores": 4,
            "gpu_available": False,
            "gpu_name": "No CUDA GPU",
            "gpu_memory_gb": 0,
            "pytorch_version": "test-pytorch-1.0",
            "cuda_version": "N/A",
        },
    )

    system_info.print_system_diagnostics()
    captured = capsys.readouterr().out
    text = _strip_ansi(captured).replace("\n", " ").strip()
    # collapse multiple whitespace sequences (ANSI/Console can add extra spacing)
    text = re.sub(r"\s+", " ", text)

    assert "CPU Cores: 4" in text
    assert "PyTorch: test-pytorch-1.0" in text
    assert "GPU Available: False" in text
    # default separator uses ` . ` so check pieces are separated
    assert "CPU Cores: 4 . PyTorch" in text


def test_print_single_line_custom_separator(monkeypatch, capsys):
    monkeypatch.setattr(
        system_info,
        "diagnose_system",
        lambda: {
            "cpu_cores": 2,
            "gpu_available": True,
            "gpu_name": "Test GPU",
            "gpu_memory_gb": 16,
            "pytorch_version": "pt-2.4",
            "cuda_version": "12.1",
        },
    )

    # use a custom separator
    custom = " | "
    system_info.print_system_diagnostics(sep=custom)
    captured = capsys.readouterr().out
    text = _strip_ansi(captured).replace("\n", " ").strip()
    # collapse multiple whitespace sequences so tests are robust to formatting
    text = re.sub(r"\s+", " ", text)

    assert "CPU Cores: 2 | PyTorch: pt-2.4" in text
    assert "GPU: Test GPU | GPU Memory: 16 GB" in text
    assert "CUDA: 12.1" in text
