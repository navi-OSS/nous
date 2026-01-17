import torch
import time
from nous.workspace import NousWorkspace
from nous.symbolic import ExprVar

def demo_turbo():
    print("=== Nous V8.0: Hyper-Scaling Benchmark (Turbo vs Recursive) ===")
    ws = NousWorkspace()
    iters = 50
    
    # 1. Create a deep recursive graph
    depth = 800
    print(f"[Phase 1] Constructing a chain of {depth} operations...")
    
    # Use ws.run to build the node securely via the DSL context
    build_code = f"""
x = ExprVar('x')
res = x
for i in range({depth}):
    if i % 2 == 0:
        res = (res + 0.001) * 0.999
    else:
        res = exp(res * 0.0001)
return res
"""
    node = ws.run(build_code)
            
    # 2. V7.0 Baseline: Recursive to_taylor
    print(f"\n[Phase 2] Benchmark: Recursive Evaluation (V7.0 Style)")
    try:
        # Warmup
        _ = node.to_taylor(1.0, 5, ws.model.hilbert)
        
        start = time.time()
        for _ in range(iters):
            val_v7 = node.to_taylor(1.0, 5, ws.model.hilbert)
        t_v7 = (time.time() - start) / iters
        print(f"Average Time: {t_v7:.6f}s")
    except Exception as e:
        print(f"V7.0 Failed (As expected for depth {depth}): {e}")
        t_v7 = float('inf')
        val_v7 = None

    # 3. V8.0 Upgrade: Compiled 'Turbo' Evaluation
    print(f"\n[Phase 3] Benchmark: Compiled 'Turbo' Evaluation (V8.0 Style)")
    program = ws.run("return soft_compile(node)", {"node": node})
    
    # Warmup
    _ = program.to_taylor(1.0, 5, ws.model.hilbert)
    
    start = time.time()
    for _ in range(iters):
        val_v8 = program.to_taylor(1.0, 5, ws.model.hilbert)
    t_v8 = (time.time() - start) / iters
    print(f"Average Time: {t_v8:.6f}s")
    
    # 4. Success Metrics
    print("\n" + "="*40)
    if t_v7 == float('inf'):
        print(f"V7.0 Status: FAILED (Recursion Limit)")
        print(f"V8.0 Status: PASSED (Turbo Success)")
        speedup_str = "INF (V7.0 couldn't even run)"
    else:
        speedup = t_v7 / t_v8
        print(f"SPEEDUP: {speedup:.2f}x faster")
        speedup_str = f"{speedup:.2f}x"
        diff = torch.norm(val_v7 - val_v8)
        print(f"PRECISION: {diff.item():.2e} error")
        
    print("="*40)
    
    if t_v8 < 0.1: # Reasonable threshold for 800 ops
        print("✓ SUCCESS: Turbo Scaling targets achieved.")
    else:
        print("? NOTE: Performance could be improved with torch.compile.")

if __name__ == "__main__":
    demo_turbo()
