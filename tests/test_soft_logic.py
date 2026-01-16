from interpreter import NeuralInterpreter
from symbolic import ExprVar
from engine import NousModel

def test_soft_logic():
    print("=== Soft Logic Gradient Test ===")
    model = NousModel()
    interpreter = NeuralInterpreter(model)
    
    # y = soft_if(x, 10.0, 2.0)
    # y = sig(x)*10 + (1-sig(x))*2
    # At x=0, sig(0)=0.5, y = 5 + 1 = 6.
    # Gradient should exist.
    
    code = "return soft_if(x, 10.0, 2.0)"
    
    # Symbolic x
    x_sym = ExprVar('x')
    inputs = {'x': x_sym}
    
    res = interpreter.execute(code, inputs)
    print(f"Result Expression: {res}")
    
    # Differentiate
    grad = res.diff('x')
    print(f"Gradient Expression: {grad}")
    
    # Verification with Taylor expansion at x=0
    # res.to_taylor needs hilbert engine.
    # We can use model.expand() which handles this.
    expanded_res = model.expand(res, center=0.0)
    print(f"Taylor Expansion at x=0: {expanded_res}")
    # Expected value at x=0 is 6.0
    if abs(expanded_res[0].item() - 6.0) < 1e-4:
        print("✓ Value correct at x=0")
    else:
        print(f"✗ Value Incorrect: {expanded_res[0].item()}")
        
    # Check gradient value
    # diff of sig(x)*10 + (1-sig(x))*2
    # = sig'(x)*10 - sig'(x)*2 = 8*sig'(x)
    # sig'(0) = 0.25
    # Expected grad = 8 * 0.25 = 2.0
    # Convert gradient expression to Taylor
    expanded_grad = model.expand(grad, center=0.0)
    grad_val = expanded_grad[0].item()
    print(f"Gradient Value at x=0: {grad_val}")
    
    if abs(grad_val - 2.0) < 1e-4:
         print("✓ Gradient correct at x=0")
    else:
         print(f"✗ Gradient Incorrect: Expected 2.0, Got {grad_val}")

if __name__ == "__main__":
    test_soft_logic()
