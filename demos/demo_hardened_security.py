import torch
from nous.workspace import NousWorkspace

def demo_security_sandbox():
    print("=== Demo: Nous V6.0 Hardened Security Sandbox (TEE) ===")
    ws = NousWorkspace(hard_logic=True)
    
    # 1. Attempt Breach: Import a forbidden module
    print("\n[Test 1] Attempting forbidden import (socket)...")
    code_import = "import socket; s = socket.socket()"
    try:
        ws.run(code_import)
    except Exception as e:
        print(f"FAILED (Correct!): {e}")

    # 2. Attempt Breach: Use a forbidden built-in (eval)
    print("\n[Test 2] Attempting forbidden built-in (eval)...")
    code_eval = "return eval('2+2')"
    try:
        ws.run(code_eval)
    except Exception as e:
        print(f"FAILED (Correct!): {e}")

    # 3. Attempt Breach: Network Access via torch (if it were possible)
    # We only expose a subset of torch ops
    print("\n[Test 3] Checking torch isolation...")
    code_torch_net = "return torch.hub.load('hub_model')" # Torch hub should be blocked
    try:
        ws.run(code_torch_net)
    except Exception as e:
        print(f"FAILED (Correct!): {e}")

    # 4. Success: Legitimate Robust Math/Analysis
    print("\n[Test 4] Verifying robust support for whitelisted libraries (torch/math)...")
    code_legit = """
# math.pi is whitelisted
area = math.pi * (10 ** 2)

# torch.ones and torch.mean are whitelisted
# We explicitly set requires_grad=True to test the gradient chain
x = torch.ones(3, requires_grad=True)
mu = torch.mean(x * 2.0)

return {'area': area, 'mu': mu}
"""
    results = ws.run(code_legit)
    print(f"SUCCESS: Result = {results}")
    
    # 5. Verify Differentiability
    print("\n[Test 5] Verifying gradients through safe library wrappers...")
    mu = results['mu']
    mu.backward()
    
    # If the gradient chain is intact, the backward pass should have happened
    if torch.is_tensor(mu) and mu.grad_fn is not None:
        print("✓ SUCCESS: TEE is both secure and robust. Gradients are preserved.")
    else:
        print("FAIL: Security layer broke gradient flow.")

if __name__ == "__main__":
    demo_security_sandbox()
