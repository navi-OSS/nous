"""
Nous GPU Benchmark for Google Colab

Instructions:
1. Open Google Colab (https://colab.research.google.com)
2. Go to Runtime -> Change runtime type -> Select GPU (T4)
3. Paste this entire script into a cell and run

This will clone Nous, install it, and benchmark on GPU.
"""

# ============================================================
# SETUP (Run once)
# ============================================================
!git clone https://github.com/your-username/nous.git 2>/dev/null || echo "Already cloned"
%cd nous
!pip install -e . -q

# ============================================================
# INLINE NOUS CODE (If not using git, paste core modules here)
# ============================================================
# Alternatively, you can paste the content of engine.py and llm.py
# directly here to avoid needing the git repo.

# ============================================================
# BENCHMARK SCRIPT
# ============================================================
import torch
import time

# Check GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Import Nous
from nous.engine import NousHilbertCore, NousModel
from nous.llm import NousLayer, NousTransformerBlock

def benchmark(fn, n_runs=100, warmup=20, name="Op"):
    """Benchmark with CUDA sync."""
    for _ in range(warmup):
        fn()
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    times = []
    for _ in range(n_runs):
        if device.type == 'cuda':
            torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        if device.type == 'cuda':
            torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
    
    mean = sum(times) / len(times)
    std = (sum((t - mean)**2 for t in times) / len(times)) ** 0.5
    print(f"  {name:<45} {mean:>8.3f} ms ± {std:.3f}")
    return mean

print("\n" + "="*60)
print("NOUS GPU BENCHMARKS")
print("="*60)

# ============================================================
# 1. HILBERT CORE BENCHMARKS
# ============================================================
print("\n--- Hilbert Core (32 terms) ---")
hilbert = NousHilbertCore(max_terms=32)

a = torch.randn(32, device=device)
b = torch.randn(32, device=device)
b[0] = 1.0

benchmark(lambda: hilbert._poly_mul(a, b), name="FFT Polynomial Multiply")
benchmark(lambda: hilbert.derivative(a), name="Polynomial Derivative")
benchmark(lambda: hilbert.compose(a, b), name="Polynomial Composition")

# Batched
a_batch = torch.randn(64, 32, device=device)
b_batch = torch.randn(64, 32, device=device)
benchmark(lambda: hilbert._poly_mul(a_batch, b_batch), name="Batched Multiply (64 polys)")

# ============================================================
# 2. LLM LAYER BENCHMARKS
# ============================================================
print("\n--- NousLayer (Batch=32, Seq=128, Hidden=512) ---")
layer = NousLayer(hidden_dim=512, max_terms=16).to(device)
x = torch.randn(32, 128, 512, device=device)

benchmark(lambda: layer(x, op='derivative'), name="NousLayer - Derivative")
benchmark(lambda: layer(x, op='integrate'), name="NousLayer - Integration")
benchmark(lambda: layer(x), name="NousLayer - Soft Selection")

# Forward + Backward
def fwd_bwd():
    y = layer(x)
    y.sum().backward()
    layer.zero_grad()

benchmark(fwd_bwd, n_runs=50, name="NousLayer - Forward + Backward")

# ============================================================
# 3. SCALING BENCHMARKS
# ============================================================
print("\n--- Scaling with Batch Size ---")
for batch in [8, 32, 128, 512]:
    x = torch.randn(batch, 64, 512, device=device)
    ms = benchmark(lambda: layer(x), name=f"Batch={batch}, Seq=64")
    tokens_per_sec = batch * 64 / ms * 1000
    print(f"      Throughput: {tokens_per_sec:,.0f} tokens/sec")

print("\n--- Scaling with Sequence Length ---")
for seq in [32, 128, 512, 2048]:
    x = torch.randn(8, seq, 512, device=device)
    ms = benchmark(lambda: layer(x), name=f"Batch=8, Seq={seq}")
    tokens_per_sec = 8 * seq / ms * 1000
    print(f"      Throughput: {tokens_per_sec:,.0f} tokens/sec")

# ============================================================
# 4. TRANSFORMER BLOCK BENCHMARK
# ============================================================
print("\n--- NousTransformerBlock ---")
block = NousTransformerBlock(hidden_dim=512, num_heads=8, max_terms=16).to(device)
x = torch.randn(16, 256, 512, device=device)

benchmark(lambda: block(x), name="Full Block Forward (Batch=16, Seq=256)")

def block_fwd_bwd():
    y = block(x)
    y.sum().backward()
    block.zero_grad()

benchmark(block_fwd_bwd, n_runs=30, name="Full Block Forward + Backward")

# ============================================================
# 5. MEMORY USAGE
# ============================================================
if device.type == 'cuda':
    print("\n--- GPU Memory Usage ---")
    torch.cuda.reset_peak_memory_stats()
    
    x = torch.randn(32, 512, 768, device=device)
    layer_large = NousLayer(hidden_dim=768, max_terms=32).to(device)
    y = layer_large(x)
    y.sum().backward()
    
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"  Peak memory (Batch=32, Seq=512, Hidden=768): {peak:.2f} GB")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "="*60)
print("BENCHMARK COMPLETE")
print("="*60)
