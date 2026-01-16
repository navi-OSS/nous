import time
from nous.interpreter import NeuralInterpreter
from nous.engine import NousModel
from nous.symbolic import ExprVar

def benchmark():
    model = NousModel()
    interpreter = NeuralInterpreter(model)
    
    code = """
y = x * 2 + 1
z = y ** 2
return z
"""
    inputs = {'x': ExprVar('x')}
    
    print("=== Caching Performance Benchmark ===")
    
    # 1. Warm up and first run (cache miss)
    start = time.perf_counter()
    interpreter.execute(code, inputs)
    first_run_time = time.perf_counter() - start
    print(f"First run (compile + trace): {first_run_time*1000:.4f} ms")
    
    # 2. Subsequent runs (cache hit)
    n_iters = 1000
    start = time.perf_counter()
    for _ in range(n_iters):
        interpreter.execute(code, inputs)
    avg_cached_time = (time.perf_counter() - start) / n_iters
    print(f"Average cached run: {avg_cached_time*1000:.4f} ms")
    
    # 3. Compare with a new string (forced re-compile)
    new_code = """
y = x * 3 + 1
return y
"""
    start = time.perf_counter()
    interpreter.execute(new_code, inputs)
    recompile_time = time.perf_counter() - start
    print(f"Recompile run (new code): {recompile_time*1000:.4f} ms")
    
    improvement = (recompile_time / avg_cached_time) if avg_cached_time > 0 else 0
    print(f"\nSpeedup from cache: {improvement:.2f}x")

if __name__ == "__main__":
    benchmark()
