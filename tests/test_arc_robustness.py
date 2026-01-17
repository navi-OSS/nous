"""
Nous V8.6: ARC Robustness / Differentiability Stress Test.

Checks if gradients flow through SoftGrid operations.
1. Pixel Value Gradients (Should PASS)
2. Spatial Parameter Gradients (Should FAIL currently)
"""
import torch
from nous.arc import SoftGrid

def test_robustness():
    print("=== ARC Robustness Audit ===")
    
    # Test 1: Pixel Differentiability
    print("\n[Test 1] Pixel Value Differentiability")
    data = torch.randn(5, 5, requires_grad=True)
    g = SoftGrid(data)
    
    # Op chain: Crop -> Resize -> Sum
    g_crop = g.crop(1, 1, 3, 3)
    g_out = g_crop.scale(6, 6)
    loss = g_out.data.sum()
    
    loss.backward()
    if data.grad is not None and data.grad.abs().sum() > 0:
        print("PASS: Gradients flow back to pixel values.")
    else:
        print("FAIL: Pixel gradients blocked.")
        
    # Test 2: Spatial Parameter Differentiability
    print("\n[Test 2] Spatial Parameter Differentiability")
    # Can we learn WHERE to crop?
    # Coordinates must be floats for grad, but crop() casts to int
    start_x = torch.tensor(1.0, requires_grad=True)
    
    data2 = torch.randn(10, 10)
    g2 = SoftGrid(data2)
    
    # We expect this to FAIL if using naive slicing, but PASS if using Soft Spatial Transformers
    try:
        # Note: Differentiable Grid Sample requires normalized coordinates (-1 to 1)
        # But our API takes pixel coordinates. The new crop() implementation handles this conversion.
        # However, grid_sample is only differentiable w.r.t the sampling grid (theta).
        # Our crop() builds theta from x, y. So dy/dx should flow!
        
        g_crop2 = g2.crop(start_x, 1, 3, 3)
        loss2 = g_crop2.data.sum()
        loss2.backward()
        
        if start_x.grad is not None and start_x.grad.abs().sum() > 0:
            print(f"PASS: Gradients flow to spatial coords. Grad: {start_x.grad}")
        else:
            print("FAIL: Gradients blocked for spatial coords.")
            print(f"      Grad value: {start_x.grad}")
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"FAIL: Execution error: {e}")

if __name__ == "__main__":
    test_robustness()
