import torch
from nous.engine import NousModel
from nous.interpreter import NeuralInterpreter

def demo_ste_loops():
    print("=== Demo: STE Exact Logic in Loops ===")
    # With hard_logic=True, the loop should terminate exactly and sum exactly
    model = NousModel(hard_logic=True)
    interpreter = NeuralInterpreter(model)
    
    # Sum 1 to 5 using soft_while
    loop_code = """
def sum_to(n):
    def cond(state):
        i, s = state
        return n - i
    def body(state):
        i, s = state
        i = i + 1
        s = s + i
        return i, s
    final_state = soft_while(cond, body, (0, 0))
    return final_state[1]

return sum_to(n)
"""
    n_val = torch.tensor(5.0, requires_grad=True)
    n = model.to_symbolic(n_val)
    
    result = interpreter.execute(loop_code, {'n': n})
    print(f"Result for sum_to(5): {result} (Type: {type(result)})")
    
    # Forward pass
    val = result.to_taylor(0.0, 1, model.hilbert)[0]
    print(f"Numerical value: {val.item()} (Expected exactly 15.0)")
    print(f"Val requires grad: {val.requires_grad}")
    
    # Verify Differentiabiliy (Backward Pass)
    if not val.requires_grad:
        print("!!! ERROR: val does not require grad. Tracing back...")
        # Check result itself
        t = result.to_taylor(0.0, 1, model.hilbert)
        print(f"Full Taylor requires grad: {t.requires_grad}")
    else:
        val.backward()
        print(f"Gradient w.r.t n: {n_val.grad}")
    
    if val.item() == 15.0:
        print("✓ SUCCESS: Exact loop sum achieved.")
    if n_val.grad is not None:
        print(f"✓ SUCCESS: Differentiability preserved through loop.")

if __name__ == "__main__":
    demo_ste_loops()
