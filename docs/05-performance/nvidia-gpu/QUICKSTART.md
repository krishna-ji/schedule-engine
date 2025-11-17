# GPU Quick Start - 5 Minute Setup

**Goal:** Enable GPU acceleration for 3-5× faster RL training  
**Time Required:** 5 minutes  
**Prerequisites:** NVIDIA GPU with 8GB VRAM

---

## Step 1: Verify GPU (30 seconds)

```powershell
# Check GPU is detected
nvidia-smi

# Should show your NVIDIA GPU and CUDA version
```

**If this fails:** Install/update NVIDIA drivers from nvidia.com

---

## Step 2: Verify PyTorch CUDA (30 seconds)

```powershell
uv run python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

**Expected Output:**
```
CUDA Available: True
GPU: NVIDIA GeForce RTX xxxx
```

**If CUDA is False:**
```powershell
# Reinstall PyTorch with CUDA support
uv pip uninstall torch
uv pip install torch --index-url https://download.pytorch.org/whl/cu121
```

---

## Step 3: Enable GPU in Config (1 minute)

Edit `configs/base.yaml` line 328:

```yaml
rl:
  agent:
    device: cuda  # ← Change from "auto" to "cuda"
```

Save the file.

---

## Step 4: Test Training (2 minutes)

```powershell
# Start training
uv run train --timesteps 5000

# In another terminal, verify GPU usage
nvidia-smi -l 1
```

**Look for:**
- GPU memory usage: 200-800 MB
- GPU utilization: 40-90%
- Process name: python.exe

---

## Step 5: Benchmark (Optional - 5 minutes)

```powershell
# Run full benchmark
uv run python scripts/benchmark_gpu_training.py
```

**Expected Result:**
```
CPU: 125s
GPU: 32s
Speedup: 3.9× faster ✅
```

---

## ✅ Done!

Your RL training is now 3-5× faster!

**Before GPU:**
- 300K training: ~90 minutes

**After GPU:**
- 300K training: ~20 minutes

**Time saved per full training run:** ~70 minutes ⏰

---

## Troubleshooting

### GPU not showing in nvidia-smi
→ Install NVIDIA drivers: https://www.nvidia.com/Download/index.aspx

### CUDA Available: False
→ Reinstall PyTorch with CUDA:
```powershell
uv pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Training still using CPU
→ Check config: `device: cuda` not `device: auto`

### Out of Memory error
→ Reduce batch size in config:
```yaml
rl:
  agent:
    ppo:
      batch_size: 64  # Reduce from 128
```

---

**For detailed information, see:** [GPU_ACCELERATION_GUIDE.md](./GPU_ACCELERATION_GUIDE.md)
