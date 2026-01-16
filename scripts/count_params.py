import torch
from nous.engine import NousModel

def count_parameters():
    model = NousModel(max_terms=32)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
    
    # Break down
    for name, param in model.named_parameters():
        print(f"  {name}: {param.numel()} ('{param.dtype}', trainable={param.requires_grad})")

if __name__ == "__main__":
    count_parameters()
