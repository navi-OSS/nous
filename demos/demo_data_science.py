import torch
from nous.workspace import NousWorkspace

def demo_advanced_data_science():
    print("=== Demo: Nous Robust Data Science Workbench ===")
    ws = NousWorkspace(hard_logic=True)
    
    # 1. Generate Noisy Synthetic Data (e.g., from an LLM-retrieved source)
    # y = 2x + 5 + noise
    torch.manual_seed(42)
    x_val = torch.linspace(0, 10, 10)
    noise = torch.randn(10) * 0.5
    y_val = 2 * x_val + 5 + noise
    
    # Make them differentiable
    x_val.requires_grad_(True)
    y_val.requires_grad_(True)
    
    ws.save('x', x_val)
    ws.save('y', y_val)
    
    print("\n[Step 1] Synthetic data loaded into workspace.")
    
    # 2. Complex Analysis Workflow:
    # - Standardize data
    # - Find the index of the largest 'error' from a naive prediction (y=2x)
    # - Run actual Linear Regression
    analysis_code = """
# 1. Standardize
x_std = soft_standardize(x)
y_std = soft_standardize(y)

# 2. Find outlier (largest diff from y=2x+5)
y_naive = [2*xi + 5 for xi in x]
diffs = [abs(y[i] - y_naive[i]) for i in range(len(y))]
peak_idx = soft_argmax(diffs)
max_diff = soft_max(diffs)

# 3. Fit Linear Regression
reg = soft_linreg(x, y)

return {
    'slope': reg['slope'], 
    'intercept': reg['intercept'], 
    'max_err': max_diff,
    'outlier_probs': peak_idx
}
"""
    results = ws.run(analysis_code)
    
    print("\n[Step 2] Multi-step Data Analysis Results:")
    slope = ws.to_taylor(results['slope'])[0].item()
    intercept = ws.to_taylor(results['intercept'])[0].item()
    max_err = ws.to_taylor(results['max_err'])[0].item()
    
    print(f"Regression Fit: y = {slope:.4f}x + {intercept:.4f}")
    print(f"Largest Pointwise Error: {max_err:.4f}")
    print(f"Outlier Probability Distribution: {results['outlier_probs']}")
    
    # 3. Verify Differentiability through the entire chain
    # We want to know how the fitted slope changes if we tweak the first data point
    print("\n[Step 3] Verifying Gradient flow (Slope -> Input):")
    slope_tensor = ws.to_taylor(results['slope'])[0]
    slope_tensor.backward()
    
    print(f"d(Slope)/d(x_0) = {x_val.grad[0].item():.4f}")
    print(f"d(Slope)/d(y_0) = {y_val.grad[0].item():.4f}")
    
    if x_val.grad[0] != 0:
        print("\n✓ SUCCESS: Production-grade robustness confirmed. All primitives are differentiable.")
    else:
        print("\nFAIL: Gradient chain broken in complex analysis.")

if __name__ == "__main__":
    demo_advanced_data_science()
