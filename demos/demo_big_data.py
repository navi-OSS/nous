import torch
import time
from nous.workspace import NousWorkspace

def demo_big_data():
    print("=== Demo: Nous Differentiable Big Data (Vectorized CSV) ===")
    ws = NousWorkspace(hard_logic=True)
    
    # 1. Load the "Big" CSV (1000 rows)
    # The interpreter uses soft_read_csv to load this into differentiable tensors
    load_code = "return soft_read_csv('data.csv')"
    data = ws.run(load_code)
    
    # Enable gradients for features and labels
    # (Normally the LLM would do this or we can do it via save)
    data['feature'].requires_grad_(True)
    data['label'].requires_grad_(True)
    
    ws.save('df', data)
    print(f"\n[Step 1] Loaded CSV with {len(data['id'])} rows.")
    
    # 2. Perform Vectorized Differentiable Query
    # "Filter labels based on whether feature > 50"
    query_code = """
f = df['feature']
l = df['label']

# 1. Differentiable filtering (Vectorized)
# Apply predicate based on features to scale labels
probs = sigmoid(f - 50.0)
filtered_labels = probs * l

# 2. Compute mean (Vectorized)
m = soft_mean(filtered_labels)

# 3. Groupby analysis (Aggregating by 'id')
stats = soft_groupby_mean(df['id'], l)

return {'mean_filtered': m, 'stats': stats}
"""
    start_time = time.time()
    results = ws.run(query_code)
    duration = (time.time() - start_time) * 1000
    
    print(f"\n[Step 2] Executed complex query on 1000 rows in {duration:.2f}ms.")
    mean_val = ws.to_taylor(results['mean_filtered'])[0].item()
    print(f"Mean of filtered labels (feature-conditional): {mean_val:.4f}")
    
    # 3. Verify Differentiability
    print("\n[Step 3] Verifying Gradient flow (Aggregated Mean -> Raw CSV Cells):")
    m_tensor = results['mean_filtered']
    if not torch.is_tensor(m_tensor):
        m_tensor = ws.to_taylor(m_tensor)[0]
        
    m_tensor.backward()
    
    gf = data['feature'].grad
    gl = data['label'].grad
    
    print(f"d(Mean)/d(feature_0) = {gf[0].item() if gf is not None else 'None'}")
    print(f"d(Mean)/d(label_0) = {gl[0].item() if gl is not None else 'None'}")
    
    if gl is not None and gl[0] != 0:
        print("\n✓ SUCCESS: Differentiable Big Data confirmed. Vectorized ops scale efficiently.")
    else:
        print("\nFAIL: Gradient chain broken or zero.")

if __name__ == "__main__":
    demo_big_data()
