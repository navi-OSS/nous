# Nous: Differentiable Symbolic Programming

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Nous** is a neuro-symbolic engine that enables **end-to-end differentiation through Python code**.  
It allows neural networks to learn by backpropagating gradients *through* the execution of standard Python functions, loops, and conditionals.

---

## 🚀 Features

| Feature | Description |
|---------|-------------|
| **Native Python Tracing** | Execute standard Python (`for`, `if`, lists, functions) and build a computation graph via operator overloading. |
| **Soft Logic** | `soft_if`, `soft_while`, `soft_switch` enable gradient flow through discrete control structures. |
| **Taylor-Mode Differentiation** | The Hilbert Engine computes high-order derivatives symbolically using Taylor series. |
| **Neural Memory** | NTM-style differentiable read/write operations for algorithmic learning. |
| **JIT Compilation** | `torch.compile()` integration for optimized execution. |
| **GPU Ready** | Built on PyTorch; runs on CPU, CUDA, and MPS. |

---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/your-org/nous.git
cd nous

# Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Install in editable mode
pip install -e .
```

### Requirements
- Python 3.8+
- PyTorch 2.0+

---

## ⚡ Quickstart

### 1. Symbolic Differentiation
```python
from nous.engine import NousModel
from nous.symbolic import ExprVar, ExprFunc

# Create the model
model = NousModel()

# Define a symbolic expression: f(x) = sin(x^2)
x = ExprVar('x')
expr = ExprFunc('sin', x * x)

# Compute symbolic derivative: f'(x) = 2x * cos(x^2)
derivative = expr.diff('x')
print(derivative)  # (cos((x * x)) * (2.0 * x))

# Evaluate at x = 1.0
coeffs = model.expand(derivative, center=1.0)
print(f"f'(1.0) = {coeffs[0].item():.4f}")  # ≈ 1.0806
```

### 2. Differentiable Python Execution
```python
from nous.engine import NousModel
from nous.interpreter import NeuralInterpreter
from nous.symbolic import ExprVar
import torch

model = NousModel()
interpreter = NeuralInterpreter(model)

# Python code to trace
code = """
y = x * x + 2 * x + 1
return y
"""

# Build symbolic graph
x_sym = ExprVar('x')
result = interpreter.execute(code, {'x': x_sym})

# Differentiate symbolically
dy_dx = result.diff('x')  # 2x + 2

# Evaluate at x = 3.0
val = model.expand(dy_dx, center=3.0)[0].item()
print(f"dy/dx at x=3: {val}")  # 8.0
```

### 3. Learning Control Flow with `soft_while`
```python
from nous.engine import NousModel
from nous.interpreter import NeuralInterpreter
import torch
import torch.nn as nn

model = NousModel()
interpreter = NeuralInterpreter(model)

# A neural controller that decides when to stop looping
class Controller(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Linear(1, 1)
    
    def forward(self, state):
        logit = self.net(state.unsqueeze(0)).squeeze()
        return logit, state + 1  # continue_logit, next_state

controller = Controller()
optimizer = torch.optim.Adam(controller.parameters(), lr=0.1)

code = """
state = initial_state

def condition(s):
    logit, _ = controller(s)
    return logit

def body(s):
    _, next_s = controller(s)
    return next_s

final = soft_while(condition, body, state, max_iters=10)
return final
"""

# Train the controller to stop at state=5
for step in range(100):
    optimizer.zero_grad()
    inputs = {
        'initial_state': torch.tensor([0.0], requires_grad=True),
        'controller': controller
    }
    result = interpreter.execute(code, inputs)
    loss = (result - 5.0) ** 2
    loss.backward()
    optimizer.step()

print(f"Learned stopping point: {result.item():.2f}")  # ≈ 5.0
```

---

## 📂 Project Structure

```
nous/
├── nous/                   # Core package
│   ├── engine.py           # NousModel, Hilbert Engine, algebra, ODE solver
│   ├── interpreter.py      # NeuralInterpreter (Python tracing via exec)
│   ├── symbolic.py         # Symbolic nodes (ExprVar, ExprFunc, etc.)
│   ├── memory.py           # Neural Memory (NTM-style read/write)
│   ├── geometry.py         # Symbolic geometry primitives
│   └── debugger.py         # Graph visualization (DOT export)
├── tests/                  # Test suite
│   └── verify.py           # Regression tests (25 cases)
├── demos/                  # Example scripts
│   ├── demo_algorithmic_learning.py  # Learn x^5 via soft_while
│   ├── demo_neural_landing.py        # Rocket landing control
│   ├── demo_program_synthesis.py     # Synthesize 2x+1
│   └── demo_register_machine.py      # Neural RAM
├── scripts/                # Utilities
│   ├── count_params.py
│   └── export_model.py
├── docs/                   # Documentation
│   ├── architecture.md
│   └── soft_logic.md
├── pyproject.toml          # Package configuration
└── README.md
```

---

## 🔧 API Reference

### `NousModel`
The main differentiable symbolic engine.

```python
from nous.engine import NousModel

model = NousModel(max_terms=32, solver_iterations=60)

# Expand symbolic expression to Taylor coefficients
coeffs = model.expand(expr, center=0.0)

# Operations
model.forward(coeffs, op='derivative')        # Differentiate
model.forward(coeffs, op='integrate')         # Integrate
model.forward(coeffs, op='evaluate', at=x)    # Evaluate at point
model.forward(coeffs, op='solve')             # Find polynomial roots
model.forward(coeffs, op='compose', inner=g)  # Function composition
```

### `NeuralInterpreter`
Executes Python code with symbolic tracing.

```python
from nous.interpreter import NeuralInterpreter

interpreter = NeuralInterpreter(model)

# Execute code with inputs
result = interpreter.execute(code_string, {'x': ExprVar('x')})

# Available in code context:
# - exp, sin, cos, log, sqrt, sinh, cosh, tan, tanh
# - sigmoid, soft_if, soft_while, soft_switch, soft_index, softmax
```

### `NeuralMemory`
Differentiable memory bank with attention-based addressing.

```python
from nous.memory import NeuralMemory

mem = NeuralMemory(num_slots=16, slot_size=32)

# Write with soft attention
mem.write(address_weights, value)

# Read with soft attention
readout = mem.read(address_weights)

# Content-based addressing
weights = mem.content_addressing(query, beta=1.0)
```

---

## 🧪 Demos

| Demo | Description | Run |
|------|-------------|-----|
| **Algorithmic Learning** | Learn to compute x^5 by discovering loop count | `python demos/demo_algorithmic_learning.py` |
| **Neural Rocket Landing** | Train a policy to soft-land a rocket | `python demos/demo_neural_landing.py` |
| **Program Synthesis** | Synthesize `2x + 1` from op primitives | `python demos/demo_program_synthesis.py` |
| **Register Machine** | Neural RAM learns `x^2 + 2x` | `python demos/demo_register_machine.py` |

---

## 🧠 How It Works

1. **Tracing**: The `NeuralInterpreter` executes Python code via `exec()`, injecting `SymbolicNode` objects that overload operators (`+`, `*`, etc.).

2. **Graph Building**: Each operation creates a node in a computation DAG (Directed Acyclic Graph).

3. **Taylor Expansion**: The Hilbert Engine expands the symbolic graph into Taylor series coefficients around a center point.

4. **Differentiation**: Symbolic `diff()` applies chain rule recursively. Taylor coefficients enable high-order derivatives.

5. **Soft Logic**: Control flow primitives blend branches using sigmoid probabilities, preserving gradient flow.

```
Python Code → Symbolic DAG → Taylor Coefficients → Gradient via PyTorch
```

---

## 📚 Further Reading

- [Architecture Deep Dive](docs/architecture.md)
- [Soft Logic Tutorial](docs/soft_logic.md)

---

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Run `python tests/verify.py` to ensure all tests pass
4. Submit a pull request

---

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.

---

**Built with ❤️ for differentiable programming research.**