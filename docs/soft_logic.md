# Soft Logic Tutorial

This guide explains how to use Nous's **soft logic** primitives to make discrete control flow differentiable.

---

## The Problem

Standard control flow is **non-differentiable**:

```python
if x > 0:
    y = x * 2
else:
    y = x * 0.5
```

The gradient `dy/dx` is **undefined at x=0** because the function has a discontinuity. Backpropagation cannot flow through the `if` statement.

---

## The Solution: Soft Logic

Soft logic replaces hard decisions with **probability-weighted blends**:

```python
# Hard: if cond then A else B
# Soft: p * A + (1-p) * B, where p = sigmoid(cond)
```

This creates a smooth approximation that PyTorch can differentiate.

---

## Primitives

### 1. `sigmoid(x)`

Converts a real number to a probability (0 to 1).

```
sigmoid(x) = 1 / (1 + exp(-x))
```

| x | sigmoid(x) |
|---|------------|
| -10 | ≈ 0 |
| 0 | 0.5 |
| +10 | ≈ 1 |

**Usage**: Use as a "soft boolean" for conditions.

---

### 2. `soft_if(condition, true_value, false_value)`

Blends two values based on a condition (interpreted as a logit).

```python
# Equivalent to:
p = sigmoid(condition)
result = p * true_value + (1 - p) * false_value
```

**Example**:
```python
from nous.interpreter import NeuralInterpreter
from nous.engine import NousModel

model = NousModel()
interpreter = NeuralInterpreter(model)

code = """
# condition is a logit: positive = more likely true
branch_a = x * 2
branch_b = x * 0.5
result = soft_if(x, branch_a, branch_b)
return result
"""

# At x=10 (very positive), result ≈ x*2 = 20
# At x=-10 (very negative), result ≈ x*0.5 = -5
# At x=0, result = 0.5*(0) + 0.5*(0) = 0 (blended)
```

**Gradient Flow**: Both branches contribute to the gradient, weighted by their probability.

---

### 3. `soft_while(condition_fn, body_fn, state, max_iters=20)`

A differentiable loop that **blends** the current state with the next state based on a continuation probability.

```python
for _ in range(max_iters):
    p_continue = sigmoid(condition_fn(state))
    next_state = body_fn(state)
    state = p_continue * next_state + (1 - p_continue) * state
```

**Key Insight**: Instead of "stop" or "continue", the loop **softly decays** toward stopping.

**Example: Counting to N**
```python
import torch
import torch.nn as nn

class Counter(nn.Module):
    def __init__(self, target):
        super().__init__()
        self.target = target
        self.threshold = nn.Parameter(torch.tensor([0.0]))
    
    def forward(self, count):
        # Continue while count < target (logit: target - count)
        logit = (self.target - count) * 5  # Scale for sharper decision
        return logit

code = """
count = 0.0

def condition(c):
    return counter(c)

def body(c):
    return c + 1.0

final = soft_while(condition, body, count, max_iters=20)
return final
"""
```

After training, the counter learns to stop at exactly `target` iterations.

---

### 4. `soft_switch(weights, values)`

Selects from multiple options using soft attention.

```python
# weights: [w0, w1, w2, ...] (should sum to 1, e.g., via softmax)
# values: [v0, v1, v2, ...]
# result = w0*v0 + w1*v1 + w2*v2 + ...
```

**Example: Program Synthesis**
```python
code = """
# Choose operation based on learned weights
op_add = x + 1
op_mul = x * 2
op_sqr = x ** 2

# weights from a neural network (softmax output)
result = soft_switch(weights, [op_add, op_mul, op_sqr])
return result
"""
```

---

### 5. `soft_index(weights, array)`

Attention-based array indexing (alias for `soft_switch`).

```python
# Hard: array[i]
# Soft: sum(weights[j] * array[j] for j in range(len(array)))
```

**Example: Neural Memory Read**
```python
code = """
memory = [slot_0, slot_1, slot_2, slot_3]

# Attention weights from a neural network
readout = soft_index(attention_weights, memory)
return readout
"""
```

---

### 6. `softmax(logits)`

Converts a list of logits to probabilities (sums to 1).

```python
# logits: [l0, l1, l2]
# probs: [exp(l0)/Z, exp(l1)/Z, exp(l2)/Z] where Z = sum(exp(li))
```

**Usage**: Generate weights for `soft_switch`.

---

## Design Patterns

### Pattern 1: Learnable Branching

```python
class BranchController(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Linear(input_size, 1)
    
    def forward(self, x):
        return self.net(x)  # Logit for soft_if

# In interpreter code:
# branch_logit = controller(state)
# result = soft_if(branch_logit, path_a, path_b)
```

### Pattern 2: Learned Loop Count

```python
class LoopController(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 16),
            nn.Tanh(),
            nn.Linear(16, 2)  # [continue_logit, state_update]
        )
    
    def forward(self, hidden):
        out = self.net(hidden)
        return out[0], out[1]

# In interpreter code:
# def condition(s):
#     logit, _ = controller(s)
#     return logit
# def body(s):
#     _, update = controller(s)
#     return s + update
# soft_while(condition, body, initial_state)
```

### Pattern 3: Discrete → Soft → Discrete (Annealing)

During training:
1. Start with high temperature (soft decisions).
2. Gradually decrease temperature (sharper decisions).
3. At inference, use hard `argmax`.

```python
for epoch in range(epochs):
    temp = max(0.1, 2.0 - epoch / 100)
    weights = torch.softmax(logits / temp, dim=-1)
    # ... training ...

# Inference:
chosen = torch.argmax(logits)
```

---

## Numerical Stability Tips

1. **Gradient Clipping**: Use `torch.nn.utils.clip_grad_norm_` to prevent exploding gradients.

2. **Input Scaling**: Keep inputs in a reasonable range (e.g., [-1, 1] or [0, 1]).

3. **Logit Initialization**: Bias initial logits toward the expected behavior:
   ```python
   self.net[-1].bias.data[0] = 2.0  # Encourage looping initially
   ```

4. **Max Iterations**: Set `max_iters` high enough for the task, but not so high that gradients vanish.

---

## Common Pitfalls

| Issue | Cause | Solution |
|-------|-------|----------|
| Loss doesn't decrease | Gradients too small | Increase logit scale (e.g., `logit * 5`) |
| NaN gradients | Exploding values | Add gradient clipping, scale inputs |
| Wrong loop count | Insufficient `max_iters` | Increase `max_iters` |
| Slow convergence | Hard decisions too early | Use temperature annealing |

---

## Summary

| Primitive | Use Case |
|-----------|----------|
| `soft_if` | Differentiable if-else |
| `soft_while` | Differentiable loops with learned termination |
| `soft_switch` | Select from N options |
| `soft_index` | Attention over arrays |
| `softmax` | Convert logits to probabilities |

These primitives unlock **algorithmic learning**: training neural networks to discover programs, control flows, and discrete structures through gradient descent.
