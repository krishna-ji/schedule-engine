#!/usr/bin/env python3
"""Quick GPU detection test"""

import torch


print("GPU/CUDA Detection Test")


print(f"\nPyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"    Memory: {props.total_memory / 1024**3:.1f} GB")
        print(f"    Compute: {props.major}.{props.minor}")
else:
    print("\n⚠️  No CUDA GPU detected!")
    print("Reasons:")
    print("  1. No NVIDIA GPU installed")
    print("  2. CUDA drivers not installed")
    print("  3. Wrong PyTorch version (CPU-only)")

print("\n" + "=" * 60)
