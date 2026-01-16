import torch
import math
import textwrap
import hashlib
from .symbolic import SymbolicNode, ExprConst, ExprVar, ExprAdd, ExprSub, ExprMul, ExprDiv, ExprPow, ExprFunc

class NeuralInterpreter:
    """
    Executes Python code natively using 'exec', injecting symbolic types 
    to build a computational graph (Native Tracing).
    """
    def __init__(self, model):
        self.model = model
        self._code_cache = {}
        # Pre-initialize math wrappers and base context
        self._base_ctx = self._get_context()

    def execute(self, code_str, inputs=None):
        """
        Execute python code string with given inputs.
        Inputs should be a dict of {name: SymbolicNode | tensor | float}.
        """
        # 1. Prepare execution context (copy from base)
        ctx = self._base_ctx.copy()
        
        if inputs:
            # Wrap inputs if they are raw numbers/tensors
            wrapped_inputs = {k: self._wrap(v) for k, v in inputs.items()}
            ctx.update(wrapped_inputs)
            
        # 2. Compile or retrieve from cache
        code_hash = hashlib.md5(code_str.encode()).hexdigest()
        if code_hash in self._code_cache:
            compiled_wrapper = self._code_cache[code_hash]
        else:
            # Wrap code in a function to capture return value
            indented_code = textwrap.indent(code_str, '    ')
            wrapper_str = f"""
def _main_():
{indented_code}

result = _main_()
"""
            compiled_wrapper = compile(wrapper_str, '<nous_trace>', 'exec')
            self._code_cache[code_hash] = compiled_wrapper
            
        # 3. Execute
        exec(compiled_wrapper, ctx)
        
        # 4. Return result
        res = ctx.get('result')
        return self._wrap(res)

    def _get_context(self):
        """Returns the dictionary of globals/locals for exec."""
        ctx = {
            'SymbolicNode': SymbolicNode,
            'ExprConst': ExprConst,
            'ExprVar': ExprVar,
            'ExprFunc': ExprFunc,
            # We can also expose torch/math if needed, but we want to intercept math calls
            'abs': abs, # Builtin abs works if __abs__ is defined, or we might need wrapper
            'print': print,
            'range': range,
            'len': len,
            'max': max,
            'min': min,
        }
        
        # Inject eager-capable math functions
        # These function handle both SymbolicNode (returning ExprFunc) 
        # and concrete values (returning concrete result via ExprConst or float)
        math_funcs = ['exp', 'sin', 'cos', 'log', 'sinh', 'cosh', 'tan', 'tanh', 'sqrt']
        for name in math_funcs:
            ctx[name] = self._create_math_wrapper(name)
            
        # Expose Soft Logic
        ctx['sigmoid'] = self._create_sigmoid_wrapper()
        ctx['soft_if'] = self._create_soft_if_wrapper()
        ctx['softmax'] = self._create_softmax_wrapper()
        ctx['soft_switch'] = self._create_soft_switch_wrapper()
        ctx['soft_index'] = self._create_soft_index_wrapper()
        ctx['soft_while'] = self._create_soft_while_wrapper()
            
        return ctx

    def _create_sigmoid_wrapper(self):
        def sigmoid(x):
            # 1/(1+exp(-x))
            # Graph tracing implementation for both symbolic and concrete
            # If x is concrete, it evaluates eagerly.
            # If x is symbolic, it builds the graph: 1 / (1 + exp(-x))
            one = 1.0
            if isinstance(x, SymbolicNode):
                one = ExprConst(1.0)
            
            if isinstance(x, (int, float)):
                 return 1.0 / (1.0 + math.exp(-x))
            
            if isinstance(x, ExprConst):
                 val = x.value
                 return ExprConst(1.0 / (1.0 + math.exp(-val)))
            
            if torch.is_tensor(x):
                return torch.sigmoid(x)
            
            return one / (one + ExprFunc('exp', -x))
        return sigmoid

    def _create_softmax_wrapper(self):
        def softmax(logits):
            # logits is a list of symbolic/float values.
            # returns list of probs.
            # exp_vals = [exp(x) for x in logits]
            # sum_val = sum(exp_vals)
            # return [v/sum_val for v in exp_vals]
            
            # Need to handle list input.
            # Assuming logits is a list (Python list).
            
            # Use injected 'exp'. But we are IN the interpreter class.
            # We can use symbolic construction directly.
            
            exps = []
            for x in logits:
                if isinstance(x, (int, float, ExprConst)):
                    val = x.value if isinstance(x, ExprConst) else x
                    exps.append(ExprConst(math.exp(val)))
                else:
                    exps.append(ExprFunc('exp', x))
            
            total = exps[0]
            for i in range(1, len(exps)):
                total = total + exps[i]
            
            probs = [e / total for e in exps]
            return probs
        return softmax

    def _create_soft_switch_wrapper(self):
        def soft_switch(weights, values):
            # weights: list of probs (sum to 1)
            # values: list of results
            # return sum(w*v)
            if len(weights) != len(values):
                raise ValueError("Weights and Values must have same length")
            
            res = weights[0] * values[0]
            for i in range(1, len(weights)):
                res = res + weights[i] * values[i]
            return res
        return soft_switch

    def _create_soft_index_wrapper(self):
        # Alias for soft_switch, but semantically "indexing"
        return self._create_soft_switch_wrapper()

    def _create_soft_if_wrapper(self):
        def soft_if(cond, true_val, false_val):
            # cond is expected to be a logit.
            # prob = sigmoid(cond)
            # res = prob * true_val + (1-prob) * false_val
            # We can use the injected sigmoid wrapper
            # But wait, self._create_sigmoid_wrapper returns a FUNCTION.
            sig = self._create_sigmoid_wrapper()
            prob = sig(cond)
            
            # Weighted sum
            # We rely on operator overloading of SymbolicNode
            return prob * true_val + (1.0 - prob) * false_val
        return soft_if

    def _create_soft_while_wrapper(self):
        """
        Creates a differentiable while loop.
        
        soft_while(condition_fn, body_fn, state, max_iters=20)
        - condition_fn(state) -> logit (positive = continue, negative = stop)
        - body_fn(state) -> new_state
        - state: initial state (can be tuple/list/tensor/symbolic)
        - max_iters: maximum iterations to unroll
        
        Returns final state after soft-blended iterations.
        """
        sig = self._create_sigmoid_wrapper()
        
        def soft_while(condition_fn, body_fn, state, max_iters=20):
            for _ in range(max_iters):
                # Compute probability of continuing
                cond_logit = condition_fn(state)
                p_continue = sig(cond_logit)
                
                # Compute next state from body
                next_state = body_fn(state)
                
                # Blend: state = p_continue * next_state + (1-p_continue) * state
                # This allows gradients to flow through "how many" iterations
                if isinstance(state, (list, tuple)):
                    # Handle tuple/list state
                    blended = []
                    for s, ns in zip(state, next_state):
                        blended.append(p_continue * ns + (1.0 - p_continue) * s)
                    state = type(state)(blended)
                else:
                    state = p_continue * next_state + (1.0 - p_continue) * state
            
            return state
        
        return soft_while

    def _create_math_wrapper(self, name):
        """Creates a wrapper for a math function that supports symbolic & concrete execution."""
        def wrapper(x):
            # If x is SymbolicNode, we check if it's a constant
            if isinstance(x, SymbolicNode):
                if isinstance(x, ExprConst):
                    # Eager evaluation
                    val = x.value
                    func = getattr(math, name)
                    return ExprConst(func(val))
                # Otherwise purely symbolic
                return ExprFunc(name, x)
            
            # If x is concrete (int, float, tensor)
            if isinstance(x, (int, float)):
                func = getattr(math, name)
                return func(x)
            
            if torch.is_tensor(x):
                 # If tensor, we might want to return tensor or ExprConst
                 func = getattr(torch, name)
                 return func(x)

            raise TypeError(f"Unsupported type for {name}: {type(x)}")
        return wrapper

    def _wrap(self, val):
        if isinstance(val, (int, float)): return ExprConst(float(val))
        if isinstance(val, SymbolicNode): return val
        if torch.is_tensor(val): 
            if val.numel() == 1 and not val.requires_grad: return ExprConst(float(val.item()))
            return val # Allow tensors to pass through if needed
        if isinstance(val, list):
             return [self._wrap(v) for v in val]
        return val
