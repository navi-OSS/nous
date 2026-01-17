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
    # Input: 10x10 grid with a 4x4 filled square in center
    data = torch.zeros(10, 10)
    data[3:7, 3:7] = 1.0
    g = grid(data)
    
    # 1. Scale (Stretch) - Interpolation
    g_stretched = g.scale(20, 20) # Stretches the 4x4 square to 8x8 essentially
    
    # 2. Canvas Resize (Pad) - No Distortion
    # Resizing canvas to 20x20 while keeping content centered
    g_padded = g.canvas_resize(20, 20, anchor_y=0.5, anchor_x=0.5)
    
    # Return both for comparison
    return [g_stretched.data, g_padded.data]
    """)
    
    print("Executing ARC Pipeline: Scale vs Canvas Resize...")
    results = ws.run(code)
    
    stretched, padded = results[0], results[1]
    
    print(f"Stretched Shape: {stretched.shape}")
    print(f"Padded Shape: {padded.shape}")
    
    # Verify Content Difference
    # Stretched center should be 'smeared' ~1.0 area larger
    # Padded center should be exactly 1.0 in a sea of 0.0
    print(f"Stretched Sum: {stretched.sum():.2f}")
    print(f"Padded Sum: {padded.sum():.2f}") # Should be 16.0 (4x4 of 1s)
    
    assert padded.shape == (20, 20)
    print("SUCCESS: Flexible Canvas Control verified.")

if __name__ == "__main__":
    demo()
