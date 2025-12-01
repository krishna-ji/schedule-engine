"""
GPU Diagnostics Script

Comprehensive GPU and CUDA setup verification for schedule-engine.

Usage:
    uv run python scripts/diagnose_gpu.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch


def print_section(title: str):
    """Print formatted section header."""
    print(f"\n{'=' * 60}")
    print(f"{title}")
    print(f"{'=' * 60}")


def check_cuda_availability():
    """Check if CUDA is available."""
    print_section("CUDA Availability")

    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")

    if cuda_available:
        print(" CUDA is properly configured")
        return True
    else:
        print(" CUDA is NOT available")
        print("\nPossible reasons:")
        print("1. No NVIDIA GPU detected")
        print("2. NVIDIA drivers not installed or outdated")
        print("3. PyTorch installed without CUDA support")
        print("\nTo fix:")
        print("  1. Install/update NVIDIA drivers from nvidia.com")
        print("  2. Reinstall PyTorch with CUDA:")
        print("     uv pip uninstall torch")
        print(
            "     uv pip install torch --index-url https://download.pytorch.org/whl/cu121"
        )
        return False


def check_pytorch_info():
    """Print PyTorch version and configuration."""
    print_section("PyTorch Information")

    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Version (PyTorch): {torch.version.cuda or 'Not available'}")
    print(
        f"cuDNN Version: {torch.backends.cudnn.version() if torch.cuda.is_available() else 'Not available'}"
    )
    print(
        f"cuDNN Enabled: {torch.backends.cudnn.enabled if torch.cuda.is_available() else 'Not available'}"
    )


def check_gpu_devices():
    """List all available GPU devices."""
    print_section("GPU Devices")

    if not torch.cuda.is_available():
        print("No GPU devices available (CUDA not available)")
        return False

    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs: {num_gpus}")

    for i in range(num_gpus):
        print(f"\n--- GPU {i} ---")
        props = torch.cuda.get_device_properties(i)

        print(f"Name: {torch.cuda.get_device_name(i)}")
        print(f"Compute Capability: {props.major}.{props.minor}")
        print(f"Total Memory: {props.total_memory / 1e9:.2f} GB")
        print(f"Multi-Processors: {props.multi_processor_count}")

        # Current memory usage
        allocated = torch.cuda.memory_allocated(i) / 1e9
        reserved = torch.cuda.memory_reserved(i) / 1e9
        print(f"Memory Allocated: {allocated:.2f} GB")
        print(f"Memory Reserved: {reserved:.2f} GB")
        print(f"Memory Available: {(props.total_memory / 1e9 - reserved):.2f} GB")

    return True


def test_gpu_operations():
    """Test basic GPU tensor operations."""
    print_section("GPU Operation Tests")

    if not torch.cuda.is_available():
        print("Skipped (CUDA not available)")
        return False

    try:
        print("Test 1: Creating tensors on GPU...")
        x = torch.randn(1000, 1000, device="cuda")
        y = torch.randn(1000, 1000, device="cuda")
        print(f"  ✓ Tensors created on: {x.device}")

        print("\nTest 2: Matrix multiplication on GPU...")
        z = torch.matmul(x, y)
        print(f"  ✓ Result shape: {z.shape}")
        print(f"  ✓ Result device: {z.device}")

        print("\nTest 3: Transferring tensors CPU  GPU...")
        x_cpu = x.cpu()
        x_gpu = x_cpu.cuda()
        print(f"  ✓ CPU device: {x_cpu.device}")
        print(f"  ✓ GPU device: {x_gpu.device}")

        print("\nTest 4: Gradient computation on GPU...")
        a = torch.randn(100, 100, device="cuda", requires_grad=True)
        b = torch.randn(100, 100, device="cuda", requires_grad=True)
        c = torch.matmul(a, b)
        loss = c.sum()
        loss.backward()
        print(f"  ✓ Gradient computed: {a.grad is not None}")
        print(f"  ✓ Gradient device: {a.grad.device}")

        print("\n All GPU operations successful!")
        return True

    except Exception as e:
        print(f"\n GPU operation failed: {e}")
        return False


def test_config_device():
    """Test device configuration from config."""
    print_section("Configuration Check")

    try:
        from src.config import get_config

        config = get_config()
        device = config.rl.agent.device

        print(f"Configured device: {device}")

        if device == "cuda" and not torch.cuda.is_available():
            print("️  WARNING: Config set to 'cuda' but CUDA not available!")
            print(
                "   Training will fail. Change to 'cpu' or 'auto' in Python presets (see src/config/presets/data.py)."
            )
            return False
        elif device == "cuda":
            print(" Config correctly set for GPU training")
            return True
        elif device == "auto":
            actual = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"️  'auto' will use: {actual}")
            if torch.cuda.is_available():
                print(" TIP: Set 'device: cuda' explicitly for guaranteed GPU usage")
            return True
        else:
            print(f"️  Using CPU (device: {device})")
            if torch.cuda.is_available():
                print(
                    " TIP: Change to 'device: cuda' in Python presets (src/config/presets/data.py) for GPU acceleration"
                )
            return True

    except Exception as e:
        print(f" Failed to load config: {e}")
        return False


def estimate_vram_usage():
    """Estimate VRAM usage for RL training."""
    print_section("VRAM Usage Estimation")

    if not torch.cuda.is_available():
        print("Skipped (CUDA not available)")
        return

    print("Estimated VRAM requirements for RL training:")
    print("")
    print(f"{'Component':<30} {'VRAM':<15} {'Notes'}")
    print(f"{'-' * 60}")
    print(f"{'PPO Policy Network (MLP)':<30} {'~50 MB':<15} {'Small network'}")
    print(f"{'PPO Value Network (MLP)':<30} {'~50 MB':<15} {'Small network'}")
    print(f"{'Experience Buffer':<30} {'~200 MB':<15} {'Default batch size'}")
    print(f"{'Batch Processing':<30} {'~100 MB':<15} {'Temporary tensors'}")
    print(f"{'Overhead & Fragmentation':<30} {'~100 MB':<15} {'Safety margin'}")
    print(f"{'-' * 60}")
    print(f"{'Total (typical)':<30} {'~500 MB':<15} {' Well within 8GB'}")
    print(f"{'Total (maximum)':<30} {'~1 GB':<15} {' Still plenty of room'}")
    print("")

    # Check current GPU
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"Your GPU: {gpu_mem:.2f} GB total VRAM")
    print(f"Available for training: ~{gpu_mem * 0.9:.2f} GB (after OS overhead)")
    print("")

    if gpu_mem >= 8:
        print(" Your GPU has sufficient VRAM for RL training")
        print(" You can even run 2-3 training sessions simultaneously!")
    elif gpu_mem >= 4:
        print(" Your GPU has sufficient VRAM for RL training")
    else:
        print("️  Your GPU has limited VRAM - may need to reduce batch sizes")


def print_recommendations():
    """Print final recommendations."""
    print_section("Recommendations")

    if torch.cuda.is_available():
        print(" GPU is ready for training!")
        print("")
        print("Next steps:")
        print("1. Enable GPU in config:")
        print(
            "   Update Python presets (src/config/presets/data.py) to set rl.agent.device: cuda"
        )
        print("")
        print("2. Run benchmark:")
        print("   uv run python scripts/benchmark_gpu_training.py")
        print("")
        print("3. Start training with GPU:")
        print("   uv run train")
        print("")
        print("4. Monitor GPU usage:")
        print("   nvidia-smi -l 1")
    else:
        print(" GPU not available")
        print("")
        print("To enable GPU:")
        print("1. Install NVIDIA drivers:")
        print("   https://www.nvidia.com/Download/index.aspx")
        print("")
        print("2. Install PyTorch with CUDA:")
        print("   uv pip uninstall torch")
        print(
            "   uv pip install torch --index-url https://download.pytorch.org/whl/cu121"
        )
        print("")
        print("3. Rerun this script to verify setup")


def main():
    print(f"\n{'=' * 60}")
    print("Schedule-Engine GPU Diagnostics")
    print(f"{'=' * 60}")

    # Run all checks
    cuda_ok = check_cuda_availability()
    check_pytorch_info()
    gpu_ok = check_gpu_devices()

    if cuda_ok and gpu_ok:
        test_gpu_operations()
        test_config_device()
        estimate_vram_usage()

    print_recommendations()

    # Exit code
    if cuda_ok and gpu_ok:
        print(f"\n{'=' * 60}")
        print(" GPU DIAGNOSTICS PASSED")
        print(f"{'=' * 60}\n")
        sys.exit(0)
    else:
        print(f"\n{'=' * 60}")
        print(" GPU DIAGNOSTICS FAILED")
        print(f"{'=' * 60}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
