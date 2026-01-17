import torch
import torch.nn as nn
from nous.engine import NousModel
from nous.interpreter import NeuralInterpreter

def demo_recursion():
    print("=== Demo: Recursive Differentiable Programming (STE Exact Logic) ===")
    model = NousModel(hard_logic=True)
    interpreter = NeuralInterpreter(model)
    
    # Recursive Factorial using soft_if
    # Note: Pure recursion needs a base case.
    # In a differentiable trace, we need to ensure it terminates.
    # We'll use a wrapper that limits depth.
    
    factorial_code = """
def fact(n):
    # Scale logit by 10 to make decisions sharper (more like hard recursion)
    return soft_if(10.0 * (n - 1), lambda: n * fact(n - 1), 1.0)

return fact(x)
"""
    x_val = torch.tensor(3.0, requires_grad=True)
    x = model.to_symbolic(x_val) # fact(3) = 6
    result = interpreter.execute(factorial_code, {'x': x})
    print(f"Result for fact(3): {result}")
    
    # Evaluate numerically (Forward Pass)
    val = result.to_taylor(center=0.0, max_terms=1, hilbert=model.hilbert)[0]
    print(f"Numerical value: {val.item()} (Expected exactly 6.0)")
    
    # Verify Differentiabiliy (Backward Pass)
    # Since val is a tensor from Taylor expansion, we can backprop
    val.backward()
    print(f"Gradient w.r.t input x (at 3.0): {x.value.grad}")
    
    if val.item() == 6.0:
        print("✓ SUCCESS: Exact result achieved.")
    if x.value.grad is not None:
        print(f"✓ SUCCESS: Differentiability preserved (grad={x.value.grad.item():.4f})")

if __name__ == "__main__":
    demo_recursion()
