"""
Hardware verification script for ground PC running NVIDIA GPU (e.g. RTX 2080 Ti 11 GB).
Checks CUDA availability, compute capability, memory allocation, and PyTorch build.
"""

import sys

def main():
    print("=" * 60)
    print(" C12 THERMAL DETECTION - GPU DIAGNOSTICS")
    print("=" * 60)
    
    try:
        import torch
    except ImportError:
        print("[ERROR] PyTorch is not installed in the current environment.", file=sys.stderr)
        sys.exit(1)

    print(f"[*] Python Version       : {sys.version.split()[0]}")
    print(f"[*] PyTorch Version      : {torch.__version__}")
    print(f"[*] CUDA Built in PyTorch: {torch.version.cuda}")
    print(f"[*] cuDNN Enabled        : {torch.backends.cudnn.enabled}")

    cuda_available = torch.cuda.is_available()
    print(f"[*] CUDA Available       : {cuda_available}")

    if not cuda_available:
        print("\n[!] WARNING: CUDA acceleration is not active.")
        print("    Inference and training will run on CPU, which may limit real-time FPS.")
        print("    Ensure the NVIDIA driver and CUDA-enabled PyTorch wheel are installed.")
        sys.exit(0)

    device_count = torch.cuda.device_count()
    print(f"[*] Detected GPU Count   : {device_count}")

    for idx in range(device_count):
        props = torch.cuda.get_device_properties(idx)
        vram_total_gb = props.total_memory / (1024 ** 3)
        print(f"\n--- GPU {idx}: {props.name} ---")
        print(f"    Compute Capability   : {props.major}.{props.minor}")
        print(f"    Total VRAM           : {vram_total_gb:.2f} GB")
        print(f"    Multi-Processors     : {props.multi_processor_count}")

        # Quick memory allocation smoke test
        try:
            test_tensor = torch.zeros((1000, 1000), device=f"cuda:{idx}")
            print(f"    Tensor Allocation    : SUCCESS")
            del test_tensor
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"    Tensor Allocation    : FAILED ({e})")

    print("\n" + "=" * 60)
    print("[OK] GPU Environment verification completed.")


if __name__ == "__main__":
    main()
