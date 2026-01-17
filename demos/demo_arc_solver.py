"""
Nous V8.4: System 5 ARC DSL Demo.

Demonstrates differentiable grid manipulations using the `SoftGrid` primitive.
Shows: Crop, Resize, Paste, and Pixel Mapping.
"""
from nous.workspace import NousWorkspace
import torch

def demo():
    print("=== System 5: Differentiable ARC DSL ===")
    ws = NousWorkspace()
    
    # Task: Processing a 10x10 Grid
    # 1. Crop the center 4x4
    # 2. Resize it to 8x8 (Upsample)
    # 3. Invert colors (1 - x)
    
    import textwrap
    code = textwrap.dedent("""
    # Input: 10x10 random grid
    # We simulate an input directly in DSL or pass via inputs
    data = torch.rand(10, 10)
    g = grid(data)
    
    # 1. Crop Center (approx coordinates (3,3) size 4x4)
    g_crop = g.crop(3, 3, 4, 4)
    
    # 2. Resize to 8x8
    g_up = g_crop.resize(8, 8)
    
    # 3. Pixel Logic: Invert Colors
    g_final = g_up.map_pixels(lambda x: 1.0 - x)
    
    return g_final.data
    """)
    
    print("Executing ARC Pipeline: Crop -> Resize -> Invert...")
    result = ws.run(code)
    
    print(f"Output Shape: {result.shape}")
    print(f"Output Sample:\n{result[:4, :4]}")
    
    assert result.shape == (8, 8)
    print("SUCCESS: Flexible Grid Manipulation verified.")

if __name__ == "__main__":
    demo()
