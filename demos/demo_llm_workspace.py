import torch
from nous.workspace import NousWorkspace

def demo_workspace():
    print("=== Demo: Nous LLM Workspace & Data Analysis Scratchpad ===")
    # Initialize workspace with hard_logic=True for exact results
    ws = NousWorkspace(hard_logic=True)
    
    # 1. Store "observed" data as a differentiable tensor
    # Imagine an LLM observes these values from an external tool
    observations_val = torch.tensor([10.0, 20.0, 30.0], requires_grad=True)
    ws.save('obs', observations_val)
    
    print("\n[Step 1] Observed data stored in scratchpad: 'obs'")
    print(ws.summary())
    
    # 2. LLM performs analysis via run()
    # Codes stores results back into scratchpad by returning a dict
    analysis_code = """
m = soft_mean(obs)
s = soft_std(obs)
return {'mean': m, 'std': s}
"""
    results = ws.run(analysis_code)
    print("\n[Step 2] Performed Analysis:")
    print(f"Mean: {results['mean']}")
    print(f"Std: {results['std']}")
    print(ws.summary())
    
    # 3. refer to stored details in a new calculation
    # Standardizing a new data point
    x_new_val = torch.tensor(25.0, requires_grad=True)
    standardize_code = """
# Refer to previously stored 'mean' and 'std'
z = (x - mean) / std
return z
"""
    z_score = ws.run(standardize_code, inputs={'x': x_new_val})
    
    print("\n[Step 3] Standardized new point 25.0 using stored mean/std:")
    num_z = ws.to_taylor(z_score)[0]
    print(f"Z-Score: {num_z.item()}")
    
    # 4. Verify Differentiability
    # We should be able to backprop from the z-score all the way to the original observations
    print("\n[Step 4] Verifying Gradient flow (Backprop from Z-Score to Observations):")
    num_z.backward()
    
    print(f"Grad check: d(Z)/d(obs_0) = {observations_val.grad[0].item()}")
    print(f"Grad check: d(Z)/d(x_new) = {x_new_val.grad.item()}")
    
    if observations_val.grad[0] != 0:
        print("\n✓ SUCCESS: End-to-end differentiability through stateful workspace preserved.")
    else:
        print("\nFAIL: Gradients did not flow back to observations.")

if __name__ == "__main__":
    demo_workspace()
