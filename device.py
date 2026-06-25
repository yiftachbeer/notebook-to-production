import torch

DEVICE = 0 if torch.cuda.is_available() else -1
DEVICE_T = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32
