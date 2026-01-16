from nous.interpreter import NeuralInterpreter
from nous.symbolic import ExprVar
from nous.engine import NousModel
import torch
import math

def run_optimization_demo():
    print("=== Differentiable Optimization Demo ===")
    print("Goal: Optimize 'x' to make python_prog(x) == 2.0")
    
    model = NousModel()
    interpreter = NeuralInterpreter(model)
    
    # A complex program with branching, loops, and math
    # Logic:
    #   if x > 0: return sin(x) + x
    #   else: return x^2
    # We use soft_if for differentiability.
    
    code = """
# x is input
term1 = sin(x) + x
term2 = x**2

# Soft switch
# cond = x. We treat x as logit for sigmoid.
y = soft_if(x*5, term1, term2) 

# Accumulate in loop (just to show off)
result = 0.0
for i in range(2):
    result = result + y

return result
"""
    # 2 * (sin(x)+x) roughly if x > 0.
    # Target = 2.0. So sin(x)+x = 1.0. 
    # x approx 0.51 works? sin(0.51)=0.48 + 0.51 = 0.99. 2*0.99=1.98.
    
    # Initialization
    x_val = torch.tensor([-2.0], requires_grad=True) # Start at -2 (wrong branch!)
    print(f"Initial x: {x_val.item()}")
    
    optimizer = torch.optim.Adam([x_val], lr=0.1)
    
    for step in range(50):
        optimizer.zero_grad()
        
        # 1. Execute Program Symbolicallly
        x_sym = ExprVar('x')
        prog_graph = interpreter.execute(code, {'x': x_sym})
        
        # 2. Bind current value of x and differentiate
        # We need a way to evaluate the graph at x_val AND get gradients w.r.t x_val.
        # Our engine produces Taylor coefficients.
        # We can evaluate the Taylor series at (x_val - center).
        
        # Center expansion at the current estimate 'x_val'
        # Note: x_val is a tensor. We need to treat it as 'center'.
        term_center = x_val.item()
        
        # Expand!
        coeffs = model.expand(prog_graph, center=term_center)
        # The 0-th term is the value f(c).
        y_pred = coeffs[0] 
        
        # But wait, 'coeffs' computation in 'model.expand' uses torch.no_grad normally?
        # No, our engine is implemented in PyTorch! 
        # SymbolicNode.to_taylor calls operations on tensors.
        # So gradients should flow from y_pred back to... wait.
        # 'to_taylor' uses 'center' (float) to compute values?
        # ExprVar.to_taylor:
        #   res[0] = center.
        # If 'center' is a float, no gradient flows to x_val (the source of center).
        
        # CRITICAL: We need ExprVar to generate a Taylor series where the 0-th term IS the tensor `x_val`.
        # Currently ExprVar takes 'center' as float (usually).
        # Let's check Symbolic.to_taylor. It takes 'center'.
        # If I pass 'x_val' (tensor) as center?
        
        try:
             coeffs = model.expand(prog_graph, center=x_val) # Pass tensor!
             y_pred = coeffs[0]
        except Exception as e:
             print(f"Graph execution failed: {e}")
             break
             
        loss = (y_pred - 2.0)**2
        loss.backward()
        optimizer.step()
        
        if step % 10 == 0:
            print(f"Step {step}: x={x_val.item():.4f}, Out={y_pred.item():.4f}, Loss={loss.item():.6f}")
            
    print(f"Final x: {x_val.item():.4f}")
    print(f"Final Output: {y_pred.item():.4f}")
    
    if loss < 1e-3:
        print("SUCCESS: Optimized x to reach target.")
    else:
        print("FAILED: Did not converge.")

if __name__ == "__main__":
    run_optimization_demo()
