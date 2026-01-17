"""
Nous Workspace - LLM Scratchpad & General Environment.

Provides a stateful, differentiable workspace for LLMs to perform 
computations, store intermediate details, and run analysis.
"""
import torch
from .engine import NousModel
from .interpreter import NeuralInterpreter
from .symbolic import SymbolicNode, ExprConst, ExprList

class NousWorkspace:
    """
    A persistent environment for LLM interactions.
    
    Attributes:
        model: The underlying NousModel.
        interpreter: NeuralInterpreter for code execution.
        scratchpad: A dictionary of named symbolic variables.
    """
    def __init__(self, model=None, hard_logic=False):
        self.model = model or NousModel(hard_logic=hard_logic)
        self.interpreter = NeuralInterpreter(self.model)
        self.scratchpad = {}
        
    def save(self, name, value):
        """Store a value in the scratchpad, ensuring it's symbolic."""
        self.scratchpad[name] = self.model.to_symbolic(value)
        
    def load(self, name):
        """Retrieve a value from the scratchpad."""
        if name not in self.scratchpad:
            raise KeyError(f"Variable '{name}' not found in scratchpad.")
        return self.scratchpad[name]
        
    def run(self, code, inputs=None):
        """
        Run code in the workspace context.
        Injects scratchpad variables into the execution context.
        """
        exec_inputs = {**self.scratchpad}
        if inputs:
            # Wrap manual inputs
            wrapped_inputs = {k: self.model.to_symbolic(v) for k, v in inputs.items()}
            exec_inputs.update(wrapped_inputs)
            
        result = self.interpreter.execute(code, exec_inputs)
        
        # If result is a dict, we can multi-update the scratchpad
        # (Convention: if code returns a dict, those are new/updated variables)
        if isinstance(result, dict):
            for k, v in result.items():
                self.save(k, v)
            return result
            
        return result

    def clear(self):
        """Reset the scratchpad."""
        self.scratchpad = {}

    def summary(self):
        """Return a human-readable summary of the scratchpad contents."""
        lines = ["=== Nous Workspace Scratchpad ==="]
        for k, v in self.scratchpad.items():
            lines.append(f"{k}: {v}")
        return "\n".join(lines)

    def to_taylor(self, val, center=0.0):
        """Helper to get numerical value (as Taylor coefficients) from a symbolic node."""
        node = self.model.to_symbolic(val)
        return self.model.expand(node, center)
