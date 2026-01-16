"""
Export Nous model to a portable format.
"""
import torch
from nous.engine import NousModel

def export_model(output_path="nous_model.pt"):
    """
    Export the Nous model architecture and initialized weights.
    
    Args:
        output_path: Path to save the model file
    """
    model = NousModel()
    
    # Save model state dict and metadata
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_config': {
            'max_terms': model.hilbert.max_terms,
            'solver_iterations': model.algebra.iterations,
        },
        'version': '1.0.0'
    }
    
    torch.save(checkpoint, output_path)
    print(f"Model exported to {output_path}")
    print(f"Model size: {sum(p.numel() for p in model.parameters())} parameters")
    
    return output_path

def load_model(checkpoint_path="nous_model.pt"):
    """
    Load a saved Nous model.
    
    Args:
        checkpoint_path: Path to the saved model file
    
    Returns:
        Loaded NousModel instance
    """
    checkpoint = torch.load(checkpoint_path)
    model = NousModel()
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Model loaded from {checkpoint_path}")
    print(f"Version: {checkpoint['version']}")
    
    return model

if __name__ == "__main__":
    # Export the model
    export_model()
    
    # Test loading
    print("\nTesting model load...")
    loaded_model = load_model()
    
    # Verify with a simple test
    coeffs = torch.tensor([[1.0, -5.0, 6.0]], dtype=torch.float64)
    roots = loaded_model.forward(coeffs, op='solve')
    print(f"Test solve: x^2 - 5x + 6 = 0")
    print(f"Roots: {roots}")
