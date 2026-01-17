# Nous Architecture

This document explains the internal architecture of the Nous differentiable symbolic engine.

---

## Overview

Nous is built on three core layers:

```
┌─────────────────────────────────────────────────────────────┐
│                    User Python Code                         │
├─────────────────────────────────────────────────────────────┤
│                  NeuralInterpreter                          │
│  (exec() tracing, soft logic, bytecode caching)             │
├─────────────────────────────────────────────────────────────┤
│                   Symbolic Layer                            │
│  (ExprVar, ExprFunc, ExprAdd, ... → DAG construction)       │
├─────────────────────────────────────────────────────────────┤
│                   Hilbert Engine                            │
│  (Taylor series, polynomial ops, root solving, ODE)         │
├─────────────────────────────────────────────────────────────┤
│                      PyTorch                                │
│  (Tensors, autograd, GPU acceleration)                      │
└─────────────────────────────────────────────────────────────┘
```

---

## 1. Neural Interpreter (`interpreter.py`)

The `NeuralInterpreter` executes Python code using Python's `exec()` function, but injects a **tracing context** that replaces standard operations with symbolic node construction.

### How It Works

1. **Code Compilation**: The Python code string is compiled to bytecode and cached (MD5 hash key).

2. **Context Injection**: A custom `globals` dict is passed to `exec()` containing:
   - Symbolic constructors (`ExprVar`, `ExprConst`, etc.)
   - Math function wrappers (`exp`, `sin`, `cos`, etc.)
   - Soft logic primitives (`soft_if`, `soft_while`, etc.)

3. **Operator Overloading**: When the code executes `x + y`, Python calls `x.__add__(y)`. If `x` is an `ExprVar`, this returns an `ExprAdd(x, y)` node instead of computing a value.

4. **Graph Building**: The result is a **Directed Acyclic Graph (DAG)** of symbolic nodes representing the computation.

### Bytecode Caching

```python
code_hash = hashlib.md5(code_str.encode()).hexdigest()
if code_hash in self._code_cache:
    compiled = self._code_cache[code_hash]
else:
    compiled = compile(wrapper_str, '<nous_trace>', 'exec')
    self._code_cache[code_hash] = compiled
```

This avoids recompiling the same code string on repeated calls.

---

## 2. Symbolic Layer (`symbolic.py`)

The symbolic layer defines node types that form the computation graph.

### Node Hierarchy

```
SymbolicNode (base)
├── ExprConst      # Constant value (e.g., 3.14)
├── ExprVar        # Variable (e.g., 'x')
├── ExprAdd        # Addition (left + right)
├── ExprSub        # Subtraction
├── ExprMul        # Multiplication
├── ExprDiv        # Division
├── ExprPow        # Power (base^exp)
└── ExprFunc       # Function call (sin, exp, log, ...)
```

### Key Methods

Each node implements:

- **`to_taylor(center, max_terms, hilbert)`**: Expand to Taylor series coefficients.
- **`diff(var)`**: Symbolic differentiation (returns a new node).
- **`__add__`, `__mul__`, etc.**: Operator overloads with algebraic simplification.

### Algebraic Simplification

Simplification rules are applied during graph construction:

```python
def __add__(self, other):
    if isinstance(other, ExprConst) and other.value == 0:
        return self  # x + 0 = x
    ...
```

This reduces expression swell before Taylor expansion.

---

## 3. Hilbert Engine (`engine.py`)

The `NousHilbertCore` is the numerical backbone. It represents functions as **Taylor series coefficients** and performs polynomial arithmetic.

### Taylor Series Representation

A function `f(x)` is represented as:
```
f(x) = c₀ + c₁x + c₂x² + ... + cₙxⁿ
```

The coefficients `[c₀, c₁, ..., cₙ]` are stored as a PyTorch tensor.

### Core Operations

| Method | Description |
|--------|-------------|
| `derivative(coeffs)` | Shift & scale coefficients: `[c₁, 2c₂, 3c₃, ...]` |
| `integrate(coeffs)` | Shift right & divide: `[0, c₀, c₁/2, c₂/3, ...]` |
| `multiply(a, b)` | Discrete convolution (polynomial multiplication) |
| `divide(a, b)` | Deconvolution (requires non-zero constant term) |
| `compose(outer, inner)` | Function composition via Horner-like iteration |
| `eval_at(coeffs, x)` | Horner's method evaluation |

### Hilbert Engine: FFT Acceleration
To ensure scalability on GPUs, the Hilbert Engine uses **FFT-based polynomial convolution** for multiplication and composition. This replaces sequential O(N²) loops with parallelizable O(N log N) operations, enabling massive speedups for batched workloads.

### LLM Integration Layer
The `nous.llm` module provides the glue for embedding Nous inside neural architectures:
- **EmbeddingBridge**: Learns a bidirectional mapping between LLM hidden states and Taylor coefficients.
- **NousLayer**: A differentiable `nn.Module` that applies symbolic operations (derivatives, roots, etc.) to hidden space.
- **Soft Selection**: Uses Gumbel-Softmax or predicted logits to choose which symbolic operation to apply, allowing the model to learn the optimal "reasoning strategy" via backpropagation.

### Root Solving

The `NousAlgebra` class finds polynomial roots using **Durand-Kerner iteration** with:
- Newton-Raphson polishing
- Multiplicity-aware refinement (Halley's method)
- Vieta's formulas for clustered roots

Gradients flow back via the **Implicit Function Theorem**.

### Performance and Speed

Nous is optimized for high-throughput GPU execution. By vectorizing symbolic operations and using FFT-based convolution, it scales linearly with batch size and sequence length.

| Setup | Metric | Result |
|-------|--------|--------|
| **Tesla T4 (Batch 128)** | Throughput | **3,500,000 tokens/sec** |
| **Tesla T4** | Forward + Backward | **3.2 ms** |
| **Apple M1 (Cached)** | Interpreter Speed | **400,000+ ops/sec** |

#### Why it's fast:
1. **FFT Convolution**: Polynomial multiplication scales O(N log N) instead of O(N²).
2. **Bytecode Caching**: Python tracing overhead is eliminated after the first call.
3. **Einsum vectorization**: Taylor composition is computed across batch/seq dimensions simultaneously.

---

## 4. Soft Logic

Soft logic primitives make discrete control flow differentiable. See [Soft Logic Tutorial](soft_logic.md) for details.

| Primitive | Purpose |
|-----------|---------|
| `sigmoid(x)` | Smooth step function (0 to 1) |
| `soft_if(cond, t, f)` | Blend true/false branches |
| `soft_while(cond, body, state)` | Differentiable loop |
| `soft_switch(weights, values)` | Weighted sum over options |
| `soft_index(weights, array)` | Attention-based indexing |

---

## 5. Memory Architecture (System 3 & 3.5)
 
 Nous separates memory into two distinct systems, analogous to the brain's working memory vs. long-term memory.
 
 ### 5.1 Neural Memory (System 2 / Working Memory)
 *Analogous to RAM or Hippocampal state.*
 
 `NeuralMemory` (`memory.py`) stores **differentiable latent vectors**. It is limited in size but fully differentiable, allowing the model to learn *how* to read/write algorithmically.
 
 - **Structure**: Fixed-size circular buffer or addressed slots `[Slots, Dimension]`.
 - **Ops**: `read(weights)`, `write(weights, val)`.
 - **Addressing**: Content-based (Cosine) or Location-based (Shift).
 
 ### 5.2 Text Memory (System 3 / Long-Term Memory)
 *Analogous to Hard Drive or Cortex.*
 
 `TextMemory` (`memory.py`) stores **raw text and embeddings**. It is "infinite" (dynamically resizing) and non-differentiable wrt storage, but differentiable wrt queries.
 
 - **Structure**: Dynamic list of `(text_str, vector_emb, timestamp)`.
 - **Retrieval (System 3.5)**: `retrieve(query_vector, k)` returns top-k text chunks.
 - **Deep RAG**: The `NeuralInterpreter` exposes `memory_retrieve` to the DSL, allowing agents to autonomously query this memory during execution.
 
 ---
 
 ## 6. Logical Solver (System 4)
 
 The `solve_logic` primitive extends the engine to handle discrete boolean constraints (SAT problems) by converting them into continuous loss landscapes.
 
 ### Boolean-to-Continuous Mapping
 
 | Boolean Logic | Continuous Penalty ($Loss \to 0$) |
 |:-------------:|:--------------------------------:|
 | `True`        | $0.0$                            |
 | `A == B`      | $(A - B)^2$                      |
 | `A and B`     | $Loss(A) + Loss(B)$              |
 | `not A`       | $1.0 - A$ (if $A \in \{0, 1\}$)  |
 
 ### Binary Forcing
 Variables are forced to binary values $\{0, 1\}$ by adding the constraint:
 $x \cdot (1 - x) = 0$
 
 This creates potential wells at 0 and 1, which the optimizer settles into.
 
 ---
 
 ## 7. ARC / Computer Vision Toolkit (System 5 & 6)
 
 The `noun.arc` module provides **differentiable 2D grid manipulation**, enabling the engine to solve visual reasoning tasks (like ARC).
 
 ### SoftGrid Primitive
 `SoftGrid` wraps a PyTorch tensor `[H, W, C]` and exposes high-level, differentiable methods.
 
 ### Differentiable Spatial Transformers (STN)
 To learn *where* to look or crop, we cannot use integer slicing. System 6 uses **Spatial Transformer Networks**:
 
 1. **Grid Generation**: `F.affine_grid` creates a sampling mesh based on continuous $x, y, w, h$ parameters.
 2. **Sampling**: `F.grid_sample` interpolates values from the source grid.
 
 This allows gradients to flow from the loss back to the coordinate parameters: $\frac{\partial Loss}{\partial x}, \frac{\partial Loss}{\partial y}$.
 
 ### Available Primitives
 - **`crop(x, y, w, h)`**: STN-based crop.
 - **`rotate(deg)`**: Affine rotation.
 - **`flip(axis)`**: Tensor flip.
 - **`find(template)`**: Convolutional pattern matching (returns heatmap).
 - **`replace_color(old, new)`**: Soft masking (`sigmoid(dist)`).
 - **`canvas_resize(h, w)`**: Non-distorting padding/cropping.
 
 ---
 
 ## 8. Data Flow Example

```python
code = "y = sin(x * x)"
```

1. **Tracing**: `x` is `ExprVar('x')`. The expression `x * x` creates `ExprMul(x, x)`. Then `sin(...)` creates `ExprFunc('sin', ExprMul(x, x))`.

2. **Taylor Expansion**: Starting from leaves:
   - `ExprVar('x').to_taylor(0, 32)` → `[0, 1, 0, 0, ...]`
   - `ExprMul` convolves both sides → `[0, 0, 1, 0, ...]` (x²)
   - `ExprFunc('sin')` composes with sin's Taylor series.

3. **Differentiation**: `expr.diff('x')` applies chain rule:
   - `sin(u)' = cos(u) * u'`
   - `u = x²`, `u' = 2x`
   - Result: `ExprMul(ExprFunc('cos', x*x), ExprMul(2, x))`

4. **Evaluation**: `model.expand(derivative, center=1.0)[0]` gives the numerical value.

---

## Performance Considerations

1. **Caching**: Bytecode compilation is cached. Repeated execution of the same code is fast.

2. **JIT Compilation**: `model.compile()` uses `torch.compile()` for the Hilbert core.

3. **Graph Simplification**: Algebraic rules reduce node count before expansion.

4. **Taylor Terms**: More terms (`max_terms`) = higher accuracy but slower. Default is 32.

---

## Extension Points

- **New Symbolic Nodes**: Subclass `SymbolicNode`, implement `to_taylor` and `diff`.
- **New Functions**: Add Taylor series to `NousHilbertCore.get_taylor_*`.
- **Custom Solvers**: Extend `NousAlgebra` or add new solution methods.
