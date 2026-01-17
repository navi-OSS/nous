"""
Nous Security Policy - Whitelisting for Trusted Execution.
"""
import torch
import math

# Whitelisted torch operations (Pure mathematical/tensor ops)
SAFE_TORCH_OPS = [
    'add', 'sub', 'mul', 'div', 'pow', 'exp', 'log', 'log1p', 'sin', 'cos', 'tan',
    'sinh', 'cosh', 'tanh', 'sqrt', 'abs', 'mean', 'sum', 'std', 'var',
    'dot', 'matmul', 'norm', 'sigmoid', 'softmax', 'round', 'floor', 'ceil',
    'clamp', 'minimum', 'maximum', 'where', 'stack', 'cat', 'unique',
    'tensor', 'arange', 'linspace', 'ones', 'zeros', 'eye',
    'unsqueeze', 'squeeze', 'reshape', 'view', 'transpose', 'permute',
    'argmin', 'argmax', 'topk', 'sort', 'roll', 'flip', 'rot90'
]
# Whitelisted math operations
SAFE_MATH_OPS = [
    'exp', 'log', 'log10', 'log2', 'log1p', 'sin', 'cos', 'tan', 'sinh', 'cosh', 'tanh', 'sqrt',
    'pow', 'pi', 'e', 'inf', 'nan', 'abs', 'ceil', 'floor', 'factorial', 'gcd', 'lcm',
    'degrees', 'radians'
]
# Whitelisted built-ins for Python
SAFE_BUILTINS = [
    'range', 'len', 'list', 'dict', 'tuple', 'set', 'int', 'float', 'str',
    'zip', 'enumerate', 'isinstance', 'type', 'print', 'bool', 'iter', 'next',
    'min', 'max', 'sum', 'round', 'abs', 'any', 'all', 'sorted', 'reversed', 'slice'
]
