# Nous: Differentiable Symbolic Programming

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Nous** is a neuro-symbolic engine that enables **autonomous problem solving** and **end-to-end differentiation** through standard Python code.

It bridges the gap between neural networks and symbolic logic by allowing gradients to flow *through* Python functions, loops, and conditionals, effectively turning code into a differentiable learning substrate.

---
 
 ## 🚀 Key Features
 
 | Feature | Description |
 |---------|-------------|
 | **Autonomous Solver (System 2)** | `solve_system` autonomously derives gradients and finds exact roots for multivariate systems. |
 | **Infinite Memory (System 3)** | `TextMemory` bridge enables the interpreter to retrieve unlimited text context (RAG). |
 | **Logical Solver (System 4)** | `solve_logic` maps discrete boolean constraints (SAT) to continuous optimization landscapes. |
 | **ARC Vision Toolkit (System 6)** | Differentiable primitives (`soft_crop`, `rotate`, `find`) for visual reasoning and grid manipulation. |
 | **Native Tracing** | Execute standard Python (`for`, `if`, functions) and capture a dynamic computation graph. |
 | **Soft Logic** | `soft_if`, `soft_while` enable gradient flow through discrete control structures. |
 
 ---
 
 ## ⚡ Quickstart

### 1. The Autonomous Solver
Solve complex calculus problems in a single line. The engine handles differentiation, optimization, and exactness snapping.

```python
from nous.workspace import NousWorkspace

ws = NousWorkspace()

code = """
# Define function: f(x,y) = x^3 + y^3 - 3xy
f = x**3 + y**3 - 3*x*y

# 1. Symbolically Compute Gradient
# .simplify() prunes expressions like '3*x^2 + 0'
df_dx = f.diff('x').simplify()
df_dy = f.diff('y').simplify()

# 2. Autonomously Solve System
# Finds roots where Gradient = 0
roots = solve_system(equations=[df_dx, df_dy], vars=['x', 'y'])

return roots
"""

# Returns: [[0.0, 0.0], [1.0, 1.0]]
print(ws.run(code)) 
```

### 2. Learning with Soft Logic
Train a model to control a Python loop using `soft_while`.

```python
from nous.interpreter import NeuralInterpreter
from nous.engine import NousModel
import torch

model = NousModel()
interpreter = NeuralInterpreter(model)

# Python code with differentiable control flow
code = """
state = initial_state
count = 0.0

def condition(s):
    # Differentiable stopping condition
    return model_controller(s)

def body(s):
    return s + 1.0

# soft_while executes loop while maintaining gradients
final_state = soft_while(condition, body, state, max_iters=10)
return final_state
"""

# ... (standard PyTorch training loop) ...
```

### 3. Logical Solver (System 4)
Solve discrete logic puzzles ($x \oplus y$ and $\neg x$) using gradients:
```python
code = """
# Find x, y in {0, 1}
x, y = ExprVar('x'), ExprVar('y')

# Constraints: (x OR y) AND (NOT x)
c1 = x + y - x*y # OR
c2 = 1 - x       # NOT

# solve_logic converts this to a continuous landscape
solutions = solve_logic([c1, c2], vars=['x', 'y'])
return solutions
"""
print(ws.run(code)) # [[0.0, 1.0]]
```

### 4. ARC Vision (System 6)
Perform differentiable Grid manipulations. Gradients flow backwards to parameters!
```python
code = """
g = grid(torch.rand(10, 10))

# Differentiable Spatial Transformer Crop
# The gradients for x, y, h, w are computed automatically
crop = g.crop(x=2.5, y=3.0, h=4.0, w=4.0)

# Pattern Matching
heatmap = g.find(template)

return crop.data
"""
```

---

## 🔧 API Reference

### `NeuralInterpreter`
The runtime that executes Python code. It exposes a specialized **Differentiable Standard Library (DSL)**.

**Accessible Functions:**
- **Math**: `sin`, `cos`, `tan`, `exp`, `log`, `sqrt`, `sinh`, `cosh`, `tanh`, `sigmoid`
- **Soft Control Flow**:
    - `soft_if(cond, true_fn, false_fn)`: Differentiable branching.
    - `soft_while(cond, body, state)`: Differentiable looping.
    - `soft_switch(logits, branches)`: Differentiable multi-way branching.
    - `soft_index(tensor, index)`: Differentiable array access.
- **Analysis**: `soft_max`, `soft_min`, `soft_argmax`, `soft_mean`, `soft_var`.
- **System 2**: `solve_system(equations, vars)`: Autonomous root finder.

### `NeuralMemory`
A differentiable memory bank for algorithmic learning.

```python
from nous.memory import NeuralMemory
mem = NeuralMemory(num_slots=16, slot_size=32)

# Methods
mem.write(weights, value, erase_strength=0.5)  # Diff. Write
readout = mem.read(weights)                    # Diff. Read
weights = mem.content_addressing(query)        # Cosine Search
weights = mem.top_k_attention(query, k=5)      # Sparse Attention
mem.reset()                                    # Clear Memory
```

### `NousModel`
The core engine backend.

- `solve_system(equations, vars)`: Finds roots for symbolic systems.
- `expand(expr, center)`: Computes Taylor series coefficients.
- `forward(op=...)`: Dispatches operations (`diff`, `integrate`, `simplify`).

---

## 📂 Project Structure

```
nous/
├── nous/
│   ├── engine.py           # Core Hilbert Engine & Solver
│   ├── interpreter.py      # Python Tracing & DSL
│   ├── symbolic.py         # Symbolic Nodes & Simplification
│   ├── memory.py           # Neural Memory Implementation
│   ├── workspace.py        # High-level User API
│   └── security_policy.py  # Sandboxing Rules
├── demos/
│   ├── demo_autonomous_calculus.py  # V8.2 Solver Demo
│   ├── demo_step_by_step_reasoning.py # Chain-of-Thought
│   └── ...
└── tests/
```

---

## 📄 License
MIT License.