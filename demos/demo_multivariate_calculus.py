import torch
import torch.nn as nn
from nous.workspace import NousWorkspace
from nous.interpreter import NeuralInterpreter
from nous.symbolic import ExprVar, ExprFunc

def demo_calculus():
    print("=== Nous: Multivariate Calculus & Optimization Demo ===")
    ws = NousWorkspace()
    
    # 1. Define the Function Symbolically
    # f(x,y) = x^3 + y^3 - 3xy
    print("[1] Defining f(x,y) = x^3 + y^3 - 3*x*y")
    
    # We use the interpreter to trace this naturally
    code_def = """
x = ExprVar('x')
y = ExprVar('y')
f = x**3 + y**3 - 3 * x * y
return f
"""
    f_node = ws.run(code_def)
    print(f"Symbolic Node: f = {f_node}")
    
    # 2. Compute Symbolic Gradients (Partial Derivatives)
    # The engine applies symbolic chain rule
    print("\n[2] Computing Symbolic Gradient \u2207f...")
    dx = f_node.diff('x')
    dy = f_node.diff('y')
    print(f"df/dx = {dx}")
    print(f"df/dy = {dy}")
    
    # 3. Numeric Optimization for Critical Points
    # We want to find (x,y) such that df/dx = 0 and df/dy = 0.
    # We will use a differentiable optimizer (Gradient Descent on the norm of the gradient).
    # Loss = (df/dx)^2 + (df/dy)^2
    
    print("\n[3] Optimizing for Critical Points (Roots of \u2207f)...")
    
    # We'll run 2 parallel searchers (batch size 2) initialized at different points
    # One near (0,0), one near (1,1) potentially, or just random
    # Let's try to discover them blindly from random seeds.
    batch_size = 8 # Increase batch to hit more basins
    # Ensure leaf tensor: create data then require grad
    params = {
        'coords': (torch.randn(batch_size, 2) * 2.5).requires_grad_(True)
    }
    
    # Use a slightly lower LR for stability near roots, but enough to move
    optimizer = torch.optim.Adam(params.values(), lr=0.02)
    
    # The optimization script
    # We effectively want to minimize the magnitude of the gradient vector.
    # Note: We are using Nous to EVALUATE the symbolic gradient at the current coords.
    
    for i in range(300):
        optimizer.zero_grad()
        
        # Current guesses
        points = params['coords'] # [B, 2]
        x_vals = points[:, 0]
        y_vals = points[:, 1]
        
        # Evaluate symbolic derivatives at current points
        # to_taylor returns [B, terms]. We need value (term 0).
        # We need to construct the 'center' dict for to_taylor properly.
        # But wait, to_taylor expects a single center usually?
        # No, ExprVar.to_taylor handles tensor centers.
        
        # Evaluate df/dx
        # Note: We must ensure vector support works here.
        dx_val = dx.to_taylor(center={'x': x_vals, 'y': y_vals}, max_terms=1, hilbert=ws.model.hilbert)
        dy_val = dy.to_taylor(center={'x': x_vals, 'y': y_vals}, max_terms=1, hilbert=ws.model.hilbert)
        
        dx_Scalar = dx_val[..., 0]
        dy_Scalar = dy_val[..., 0]
        
        # Loss = Norm of gradient vector at these points
        grad_norm = dx_Scalar**2 + dy_Scalar**2
        loss = grad_norm.mean()
        
        loss.backward()
        optimizer.step()
        
        if i % 20 == 0:
            print(f"Iter {i:02d} | Mean Grad Norm: {loss.item():.4e}")

    print("-" * 40)
    print("Optimization Complete.")
    
    # 4. Result Analysis
    final_points = params['coords'].detach()
    print("\n[4] Discovered Critical Points:")
    
    # Cluster/Round results
    for j in range(batch_size):
        pt = final_points[j]
        grad_x = dx_Scalar[j].item()
        grad_y = dy_Scalar[j].item()
        
        # Identify if it's a known solution
        is_00 = torch.norm(pt - torch.tensor([0.0, 0.0])) < 0.1
        is_11 = torch.norm(pt - torch.tensor([1.0, 1.0])) < 0.1
        
        label = "Unknown"
        if is_00: label = "Saddle Point (0,0)"
        if is_11: label = "Local Min (1,1)"
        
        print(f"Point {j}: ({pt[0]:.4f}, {pt[1]:.4f}) | Grad: ({grad_x:.4f}, {grad_y:.4f}) -> {label}")

    if loss.item() < 0.1:
        print("\nSUCCESS: Calculated derivatives and found roots successfully.")

if __name__ == "__main__":
    demo_calculus()
