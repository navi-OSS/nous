#!/usr/bin/env python3
"""
Nous Full Evaluation Suite

Comprehensive benchmarks for:
- Symbolic operations
- Taylor expansion
- Root solving
- Interpreter execution
- Soft logic
- Neural memory
- End-to-end inference
"""

import time
import torch
import math
import statistics
from nous.engine import NousModel, NousHilbertCore, TaylorODESolver
from nous.interpreter import NeuralInterpreter
from nous.symbolic import ExprVar, ExprFunc, ExprConst
from nous.memory import NeuralMemory

# ============================================================
# UTILITIES
# ============================================================

def benchmark(fn, n_runs=100, warmup=10, name="Operation"):
    """Run a function multiple times and return timing stats."""
    # Warmup
    for _ in range(warmup):
        fn()
    
    # Timed runs
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        fn()
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)  # Convert to ms
    
    return {
        'name': name,
        'mean_ms': statistics.mean(times),
        'std_ms': statistics.stdev(times) if len(times) > 1 else 0,
        'min_ms': min(times),
        'max_ms': max(times),
        'n_runs': n_runs
    }

def print_result(result):
    print(f"  {result['name']:<40} {result['mean_ms']:>8.4f} ms ± {result['std_ms']:.4f} (n={result['n_runs']})")

# ============================================================
# BENCHMARKS
# ============================================================

def benchmark_hilbert_core():
    """Benchmark core polynomial operations."""
    print("\n" + "="*60)
    print("HILBERT CORE BENCHMARKS")
    print("="*60)
    
    hilbert = NousHilbertCore(max_terms=32)
    
    # Create test polynomials
    a = torch.randn(32)
    b = torch.randn(32)
    b[0] = 1.0  # Ensure non-zero for division
    
    results = []
    
    # Polynomial multiplication
    results.append(benchmark(
        lambda: hilbert.multiply(a, b),
        name="Polynomial Multiply (32 terms)"
    ))
    
    # Polynomial division
    results.append(benchmark(
        lambda: hilbert.divide(a, b),
        name="Polynomial Divide (32 terms)"
    ))
    
    # Derivative
    results.append(benchmark(
        lambda: hilbert.derivative(a),
        name="Polynomial Derivative"
    ))
    
    # Integration
    results.append(benchmark(
        lambda: hilbert.integrate(a),
        name="Polynomial Integration"
    ))
    
    # Composition
    results.append(benchmark(
        lambda: hilbert.compose(a, b),
        name="Polynomial Composition"
    ))
    
    # Evaluation (Horner)
    x = torch.tensor(0.5)
    results.append(benchmark(
        lambda: hilbert.eval_at(a, x),
        name="Horner Evaluation"
    ))
    
    # Taylor series generation
    results.append(benchmark(
        lambda: hilbert.get_taylor_exp(),
        name="Generate exp Taylor series"
    ))
    
    results.append(benchmark(
        lambda: hilbert.get_taylor_sin(),
        name="Generate sin Taylor series"
    ))
    
    for r in results:
        print_result(r)
    
    return results

def benchmark_symbolic():
    """Benchmark symbolic graph construction and expansion."""
    print("\n" + "="*60)
    print("SYMBOLIC LAYER BENCHMARKS")
    print("="*60)
    
    model = NousModel()
    results = []
    
    # Graph construction
    def build_simple_graph():
        x = ExprVar('x')
        return x * x + 2 * x + 1
    
    results.append(benchmark(
        build_simple_graph,
        name="Build simple graph (x²+2x+1)"
    ))
    
    def build_complex_graph():
        x = ExprVar('x')
        return ExprFunc('sin', x * x) + ExprFunc('exp', -x) * x
    
    results.append(benchmark(
        build_complex_graph,
        name="Build complex graph (sin(x²)+e^(-x)*x)"
    ))
    
    # Taylor expansion
    simple_expr = build_simple_graph()
    results.append(benchmark(
        lambda: model.expand(simple_expr, center=0.0),
        name="Expand simple graph to Taylor"
    ))
    
    complex_expr = build_complex_graph()
    results.append(benchmark(
        lambda: model.expand(complex_expr, center=0.0),
        name="Expand complex graph to Taylor"
    ))
    
    # Symbolic differentiation
    results.append(benchmark(
        lambda: simple_expr.diff('x'),
        name="Symbolic diff (simple)"
    ))
    
    results.append(benchmark(
        lambda: complex_expr.diff('x'),
        name="Symbolic diff (complex)"
    ))
    
    for r in results:
        print_result(r)
    
    return results

def benchmark_root_solver():
    """Benchmark polynomial root finding."""
    print("\n" + "="*60)
    print("ROOT SOLVER BENCHMARKS")
    print("="*60)
    
    model = NousModel()
    results = []
    
    # Quadratic (2 roots)
    quad = torch.tensor([[1.0, -5.0, 6.0]], dtype=torch.float64)
    results.append(benchmark(
        lambda: model.forward(quad, op='solve'),
        n_runs=50,
        name="Solve quadratic (2 roots)"
    ))
    
    # Cubic (3 roots)
    cubic = torch.tensor([[1.0, -6.0, 11.0, -6.0]], dtype=torch.float64)
    results.append(benchmark(
        lambda: model.forward(cubic, op='solve'),
        n_runs=50,
        name="Solve cubic (3 roots)"
    ))
    
    # Degree 5
    deg5 = torch.tensor([[1.0, -15.0, 85.0, -225.0, 274.0, -120.0]], dtype=torch.float64)
    results.append(benchmark(
        lambda: model.forward(deg5, op='solve'),
        n_runs=50,
        name="Solve degree-5 polynomial"
    ))
    
    # Batch solving
    batch = torch.randn(10, 4, dtype=torch.float64)
    batch[:, 0] = 1.0  # Monic
    results.append(benchmark(
        lambda: model.forward(batch, op='solve'),
        n_runs=50,
        name="Batch solve (10 cubics)"
    ))
    
    for r in results:
        print_result(r)
    
    return results

def benchmark_interpreter():
    """Benchmark Python interpreter execution."""
    print("\n" + "="*60)
    print("INTERPRETER BENCHMARKS")
    print("="*60)
    
    model = NousModel()
    interpreter = NeuralInterpreter(model)
    results = []
    
    # Simple arithmetic
    code_simple = "y = x * 2 + 1; return y"
    inputs_simple = {'x': ExprVar('x')}
    
    # First run (cache miss)
    start = time.perf_counter()
    interpreter.execute(code_simple, inputs_simple)
    first_run = (time.perf_counter() - start) * 1000
    print(f"  First run (cache miss): {first_run:.4f} ms")
    
    # Cached runs
    results.append(benchmark(
        lambda: interpreter.execute(code_simple, inputs_simple),
        name="Simple arithmetic (cached)"
    ))
    
    # Loop
    code_loop = """
total = 0
for i in range(10):
    total = total + i
return total
"""
    results.append(benchmark(
        lambda: interpreter.execute(code_loop),
        name="Loop (sum 0..9)"
    ))
    
    # Function definition
    code_func = """
def square(n):
    return n * n
return square(x)
"""
    results.append(benchmark(
        lambda: interpreter.execute(code_func, {'x': ExprVar('x')}),
        name="Function call"
    ))
    
    # Soft logic
    code_soft = """
a = x * 2
b = x * 0.5
result = soft_if(x, a, b)
return result
"""
    results.append(benchmark(
        lambda: interpreter.execute(code_soft, {'x': ExprVar('x')}),
        name="soft_if execution"
    ))
    
    for r in results:
        print_result(r)
    
    return results

def benchmark_soft_while():
    """Benchmark soft_while loop performance."""
    print("\n" + "="*60)
    print("SOFT_WHILE BENCHMARKS")
    print("="*60)
    
    model = NousModel()
    interpreter = NeuralInterpreter(model)
    results = []
    
    # Simple counter with tensor inputs
    code = """
state = initial_state

def condition(s):
    return 5.0 - s  # Continue while s < 5

def body(s):
    return s + 1.0

final = soft_while(condition, body, state, max_iters=10)
return final
"""
    
    inputs = {'initial_state': torch.tensor([0.0], requires_grad=True)}
    
    results.append(benchmark(
        lambda: interpreter.execute(code, inputs),
        n_runs=50,
        name="soft_while (10 max iters)"
    ))
    
    # With backprop
    def forward_backward():
        inp = {'initial_state': torch.tensor([0.0], requires_grad=True)}
        result = interpreter.execute(code, inp)
        if torch.is_tensor(result):
            loss = result.sum()
            loss.backward()
    
    results.append(benchmark(
        forward_backward,
        n_runs=50,
        name="soft_while + backward pass"
    ))
    
    for r in results:
        print_result(r)
    
    return results

def benchmark_neural_memory():
    """Benchmark neural memory operations."""
    print("\n" + "="*60)
    print("NEURAL MEMORY BENCHMARKS")
    print("="*60)
    
    results = []
    
    mem = NeuralMemory(num_slots=16, slot_size=32)
    mem.init_random()
    
    address = torch.randn(16)
    value = torch.randn(32)
    query = torch.randn(32)
    
    results.append(benchmark(
        lambda: mem.read(address),
        name="Memory read (16 slots)"
    ))
    
    results.append(benchmark(
        lambda: mem.write(address, value),
        name="Memory write (16 slots)"
    ))
    
    results.append(benchmark(
        lambda: mem.content_addressing(query, beta=5.0),
        name="Content addressing"
    ))
    
    # Larger memory
    mem_large = NeuralMemory(num_slots=128, slot_size=64)
    mem_large.init_random()
    address_large = torch.randn(128)
    value_large = torch.randn(64)
    
    results.append(benchmark(
        lambda: mem_large.read(address_large),
        name="Memory read (128 slots)"
    ))
    
    for r in results:
        print_result(r)
    
    return results

def benchmark_end_to_end():
    """Benchmark complete inference pipeline."""
    print("\n" + "="*60)
    print("END-TO-END INFERENCE BENCHMARKS")
    print("="*60)
    
    model = NousModel()
    interpreter = NeuralInterpreter(model)
    results = []
    
    # Full pipeline: code -> symbolic graph -> Taylor -> evaluate
    code = """
y = sin(x * x) + exp(-x) * cos(x)
return y
"""
    
    def full_inference():
        x_sym = ExprVar('x')
        graph = interpreter.execute(code, {'x': x_sym})
        coeffs = model.expand(graph, center=0.5)
        return model.hilbert.eval_at(coeffs, torch.tensor(0.5))
    
    results.append(benchmark(
        full_inference,
        n_runs=50,
        name="Full inference (complex expr)"
    ))
    
    # With differentiation
    def inference_with_diff():
        x_sym = ExprVar('x')
        graph = interpreter.execute(code, {'x': x_sym})
        deriv = graph.diff('x')
        coeffs = model.expand(deriv, center=0.5)
        return model.hilbert.eval_at(coeffs, torch.tensor(0.5))
    
    results.append(benchmark(
        inference_with_diff,
        n_runs=50,
        name="Inference + symbolic diff"
    ))
    
    # Root finding pipeline
    def root_pipeline():
        coeffs = torch.tensor([[1.0, -5.0, 6.0]], dtype=torch.float64)
        roots = model.forward(coeffs, op='solve')
        return roots
    
    results.append(benchmark(
        root_pipeline,
        n_runs=50,
        name="Root finding pipeline"
    ))
    
    for r in results:
        print_result(r)
    
    return results

def benchmark_accuracy():
    """Verify accuracy of computations."""
    print("\n" + "="*60)
    print("ACCURACY VALIDATION")
    print("="*60)
    
    model = NousModel()
    errors = []
    
    # Test transcendental functions at various points
    test_points = [0.0, 0.1, 0.5, 1.0, 1.5]
    funcs = [
        ('exp', torch.exp),
        ('sin', torch.sin),
        ('cos', torch.cos),
        ('sinh', torch.sinh),
        ('cosh', torch.cosh),
    ]
    
    print("\nTranscendental function accuracy:")
    for name, torch_fn in funcs:
        max_error = 0.0
        for pt in test_points:
            x = torch.tensor(pt)
            coeffs = model.forward(None, op='get_identity', name=name)
            nous_val = model.hilbert.eval_at(coeffs, x).item()
            torch_val = torch_fn(x).item()
            error = abs(nous_val - torch_val)
            max_error = max(max_error, error)
        
        status = "✓" if max_error < 1e-6 else "✗"
        print(f"  {status} {name:<8} max error: {max_error:.2e}")
        errors.append((name, max_error))
    
    # Root solving accuracy
    print("\nRoot solving accuracy:")
    test_cases = [
        ("x²-5x+6=0", torch.tensor([[1.0, -5.0, 6.0]], dtype=torch.float64), {2.0, 3.0}),
        ("x²-4x+4=0", torch.tensor([[1.0, -4.0, 4.0]], dtype=torch.float64), {2.0}),
    ]
    
    for name, coeffs, expected in test_cases:
        roots = model.forward(coeffs, op='solve')
        actual = {round(roots[0, i, 0].item(), 4) for i in range(roots.shape[1])}
        match = all(any(abs(a - e) < 0.01 for a in actual) for e in expected)
        status = "✓" if match else "✗"
        print(f"  {status} {name}: roots = {actual}")
    
    return errors

def generate_report(all_results):
    """Generate summary report."""
    print("\n" + "="*60)
    print("SUMMARY REPORT")
    print("="*60)
    
    # Compute overall stats
    total_ops = sum(len(r) for r in all_results)
    
    # Find fastest/slowest
    all_benchmarks = [b for r in all_results for b in r]
    if all_benchmarks:
        fastest = min(all_benchmarks, key=lambda x: x['mean_ms'])
        slowest = max(all_benchmarks, key=lambda x: x['mean_ms'])
        
        print(f"\nTotal benchmarks run: {total_ops}")
        print(f"Fastest operation: {fastest['name']} ({fastest['mean_ms']:.4f} ms)")
        print(f"Slowest operation: {slowest['name']} ({slowest['mean_ms']:.4f} ms)")
    
    # Throughput estimates
    print("\nEstimated throughput:")
    for b in all_benchmarks:
        ops_per_sec = 1000 / b['mean_ms']
        print(f"  {b['name']:<40} {ops_per_sec:>10.1f} ops/sec")

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("="*60)
    print("NOUS FULL EVALUATION SUITE")
    print("="*60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    all_results = []
    
    # Run all benchmarks
    all_results.append(benchmark_hilbert_core())
    all_results.append(benchmark_symbolic())
    all_results.append(benchmark_root_solver())
    all_results.append(benchmark_interpreter())
    all_results.append(benchmark_soft_while())
    all_results.append(benchmark_neural_memory())
    all_results.append(benchmark_end_to_end())
    
    # Accuracy validation
    benchmark_accuracy()
    
    # Generate report
    generate_report(all_results)
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
