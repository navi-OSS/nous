import torch
import math
from engine import NousModel
from expression import install_dsl

def test_symbolic_accuracy():
    print("=== Symbolic Accuracy Verification ===")
    model = NousModel(max_terms=32)
    model = install_dsl(model)
    
    # 1. Parse sin(x)
    expr = model.parse("sin(x)")
    print(f"Parsed expression: {expr}")
    
    # 2. Symbolic derivative: sin(x)' = cos(x)
    deriv = expr.diff('x')
    print(f"Symbolic derivative: {deriv}")
    
    # 3. Expansion at 0 (Maclaurin)
    cos_0_exact = 1.0
    cos_coeffs = model.expand(deriv, center=0.0)
    cos_0_approx = cos_coeffs[0].item()
    print(f"cos(0) expansion at 0: {cos_0_approx} (Error: {abs(cos_0_approx - cos_0_exact):.2e})")
    
    # 4. Expansion at pi (Non-zero center)
    # sin(x) expanded at pi: sin(pi) + cos(pi)(x-pi) - sin(pi)(x-pi)^2/2 - cos(pi)(x-pi)^3/6 ...
    # sin(pi) = 0, cos(pi) = -1
    # sin(x) ≈ -(x-pi) + (x-pi)^3/6
    sin_coeffs_pi = model.expand(expr, center=math.pi)
    print(f"sin(x) expansion at pi: coeffs[0]={sin_coeffs_pi[0].item():.4f}, coeffs[1]={sin_coeffs_pi[1].item():.4f}")
    
    # 5. Accuracy check for sin(pi + 0.1)
    x_val = 0.1 # This is (x - pi)
    approx = 0.0
    for i in range(len(sin_coeffs_pi)):
        approx += sin_coeffs_pi[i].item() * (x_val ** i)
    
    exact = math.sin(math.pi + x_val)
    print(f"sin(pi + 0.1) approx: {approx:.10f}")
    print(f"sin(pi + 0.1) exact:  {exact:.10f}")
    print(f"Error: {abs(approx - exact):.2e}")

    # 6. Differentiation of Power
    pow_expr = model.parse("x**3")
    pow_deriv = pow_expr.diff('x') # 3*x**2
    print(f"\nPower test: ({pow_expr})' = {pow_deriv}")
    
    # Expand at x=2
    # 3*(2)**2 = 12
    deriv_coeffs_2 = model.expand(pow_deriv, center=2.0)
    deriv_val_2 = deriv_coeffs_2[0].item()
    print(f"3*(2)**2 expansion at 2: {deriv_val_2} (Error: {abs(deriv_val_2 - 12.0):.2e})")

if __name__ == "__main__":
    test_symbolic_accuracy()
