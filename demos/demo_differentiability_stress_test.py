import torch
import time
from nous.workspace import NousWorkspace

def demo_everything_is_differentiable():
    print("=== Differentiability Stress Test: The 'End-to-End' Proof ===")
    ws = NousWorkspace(hard_logic=True)
    
    # 1. Setup Input: A differentiable observation
    x_raw = torch.tensor([10.0, 20.0, 30.0], requires_grad=True)
    ws.save('obs', x_raw)
    
    # 2. The Stress Test Code:
    # This script chains:
    # - Recursion (Factorial)
    # - Branching (soft_if)
    # - Vectorized reduction (soft_mean)
    # - Linear Regression (soft_linreg)
    # - Knowledge retrieval (soft_search)
    
    stress_code = """
# 1. Recursive Factorial on mean of observations
mu = soft_mean(obs)
n_idx = soft_if(mu > 15.0, lambda: 3.0, lambda: 2.0)

def fact(n):
    return soft_if(n <= 1, lambda: 1.0, lambda: n * fact(n-1))

f_val = fact(n_idx)

# 2. Linear Regression on observed trend
# We use x_raw as features and f_val as a potential target influencer
x_feat = [1.0, 2.0, 3.0]
y_target = [v * f_val for v in obs] # Chaining f_val into a new dataset
reg = soft_linreg(x_feat, y_target)

# 3. Knowledge Lookup influenced by slope
# If slope is high, search for "growth", else "decline"
q_text = soft_if(reg['slope'] > 0.0, lambda: "growth", lambda: "decline")

# Assume we have a dummy KB loaded
kb = soft_load_book("This is a story about growth. This is a story about decline.", chunk_size=5)
search_res = soft_search(q_text, kb, k=1)

# Result is a mix of the slope and the search relevance
return reg['slope'] * search_res['relevance'].sum()
"""
    
    print("\n[Step 1] Running complex multi-modal chain...")
    start = time.time()
    final_result = ws.run(stress_code)
    duration = (time.time() - start) * 1000
    print(f"Execution successful in {duration:.2f}ms.")
    
    # 3. The Backward Pass
    print("\n[Step 2] Performing Backward Pass (Final Result -> x_raw)...")
    if not torch.is_tensor(final_result):
        val = ws.to_taylor(final_result)[0]
    else:
        val = final_result
        
    val.backward()
    
    # 4. Verification
    grad = x_raw.grad
    print(f"Gradients found: {grad}")
    
    if grad is not None and grad.abs().sum() > 0:
        print("\n✓ SUCCESS: Everything is differentiable. The chain is intact.")
    else:
        print("\nFAIL: Gradient chain broken.")

if __name__ == "__main__":
    demo_everything_is_differentiable()
