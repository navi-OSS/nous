import torch
import math
from engine import NousModel
from interpreter import NeuralInterpreter
from symbolic import ExprVar

def test_transcendental():
    print("=== Transcendental Completeness Verification ===")
    model = NousModel(max_terms=16)
    interpreter = NeuralInterpreter(model)
    
    test_cases = [
        ("log(x)", 2.0, 0.1, math.log),
        ("sqrt(x)", 4.0, 0.1, math.sqrt),
        ("sinh(x)", 0.5, 0.1, math.sinh),
        ("cosh(x)", 0.5, 0.1, math.cosh),
        ("tanh(x)", 0.5, 0.1, math.tanh),
        ("tan(x)", 0.5, 0.1, math.tan),
    ]
    
    all_passed = True
    
    for code, center, delta, exact_func in test_cases:
        print(f"\nTesting: {code} at center={center}, delta={delta}")
        
        # 1. Execute
        x_sym = ExprVar('x')
        res_node = interpreter.execute(f"return {code}", {'x': x_sym})
        
        # 2. Expand
        coeffs = model.expand(res_node, center=center)
        
        # 3. Predict value at center + delta
        approx_val = 0.0
        for i in range(len(coeffs)):
            approx_val += coeffs[i].item() * (delta ** i)
            
        exact_val = exact_func(center + delta)
        error = abs(approx_val - exact_val)
        
        print(f"  Approx: {approx_val:.10f}")
        print(f"  Exact:  {exact_val:.10f}")
        print(f"  Error:  {error:.2e}")
        
        if error < 1e-6:
            print(f"  ✓ PASS")
        else:
            print(f"  ✗ FAIL")
            all_passed = False
            
        # 4. Check Gradient (diff)
        grad_node = res_node.diff('x')
        grad_coeffs = model.expand(grad_node, center=center)
        approx_grad = grad_coeffs[0].item()
        
        # Numerical gradient for verification
        eps = 1e-5
        exact_grad = (exact_func(center + eps) - exact_func(center - eps)) / (2 * eps)
        grad_error = abs(approx_grad - exact_grad)
        
        print(f"  Grad Approx: {approx_grad:.10f}")
        print(f"  Grad Exact:  {exact_grad:.10f}")
        print(f"  Grad Error:  {grad_error:.2e}")
        
        if grad_error < 1e-5:
            print(f"  ✓ GRAD PASS")
        else:
            print(f"  ✗ GRAD FAIL")
            all_passed = False
            
    if all_passed:
        print("\n🎉 ALL TRANSCENDENTAL TESTS PASSED!")
    else:
        print("\n❌ SOME TESTS FAILED")
        exit(1)

if __name__ == "__main__":
    test_transcendental()
