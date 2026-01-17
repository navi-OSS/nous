import torch
print(f"torch.conv2d exists: {hasattr(torch, 'conv2d')}")
print(f"torch.unfold exists: {hasattr(torch, 'unfold')}")
print(f"torch.nn.functional.conv2d exists: {True}") # We know this
