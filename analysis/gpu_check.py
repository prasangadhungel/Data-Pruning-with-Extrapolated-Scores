import psutil
import torch

# — Host RAM —
vm = psutil.virtual_memory()
print(f"Host total RAM:     {vm.total/1e9:.1f} GB")
print(f"Host available RAM: {vm.available/1e9:.1f} GB")
print(f"Host used RAM:      {vm.used/1e9:.1f} GB\n")

# — GPU status —
if torch.cuda.is_available():
    for idx in range(torch.cuda.device_count()):
        prop = torch.cuda.get_device_properties(idx)
        tot = prop.total_memory / 1e9
        alloc = torch.cuda.memory_allocated(idx) / 1e9
        reserved = torch.cuda.memory_reserved(idx) / 1e9
        free = tot - reserved

        print(f"GPU #{idx}: {prop.name}")
        print(f"  Total VRAM:  {tot:.1f} GB")
        print(f"  Allocated:   {alloc:.1f} GB")
        print(f"  Cached (PyTorch reserved): {reserved:.1f} GB")
        print(f"  Free for use:             {free:.1f} GB\n")
else:
    print("No CUDA device available")


# gtx_1080

# Name:            NVIDIA GeForce RTX 2080 Ti
# Total memory:    11.4 GB
# Compute device:  7.5

# Starting job 1357317
# SLURM assigned me these nodes:
# gpu18               
# Name:            NVIDIA A100-PCIE-40GB
# Total memory:    42.4 GB
# Compute device:  8.0
