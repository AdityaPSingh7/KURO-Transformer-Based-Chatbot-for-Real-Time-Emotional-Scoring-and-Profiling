import torch

if torch.cuda.is_available():
    print("✅ CUDA is available! Using GPU:", torch.cuda.get_device_name(0))
else:
    print("⚠️ CUDA not available — using CPU")