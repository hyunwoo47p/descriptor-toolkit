"""ChemDescriptorML (CDML) - GPU Utilities"""
import torch
from typing import Tuple

def get_optimal_device(prefer_gpu: bool = True) -> torch.device:
    if prefer_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA Version: {torch.version.cuda}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        return device
    else:
        if prefer_gpu:
            print("⚠ GPU requested but not available, using CPU")
        else:
            print("Using CPU")
        return torch.device('cpu')

def get_gpu_memory_info() -> Tuple[float, float, float]:
    if not torch.cuda.is_available():
        return 0.0, 0.0, 0.0
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    return allocated, reserved, total

def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
