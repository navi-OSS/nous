import torch
import torch.nn.functional as F
import math
import textwrap
import hashlib
import ast
import csv
from .symbolic import SymbolicNode, ExprConst, ExprVar, ExprAdd, ExprSub, ExprMul, ExprDiv, ExprPow, ExprFunc, ExprIf, ExprList
from .memory import NeuralMemory
from .security_policy import SAFE_TORCH_OPS, SAFE_MATH_OPS, SAFE_BUILTINS

class CodeSafetyError(Exception):
    """Raised when unsafe code is detected in the interpreter."""
    pass

class SafeNodeVisitor(ast.NodeVisitor):
    """
    AST visitor that ensures code only uses a safe subset of Python.
    Blocks:
    - Imports (ast.Import, ast.ImportFrom)
    - With statements (ast.With)
    - Try/Except (ast.Try, ast.ExceptHandler)
    - Class definitions (ast.ClassDef)
    - Private attribute access (starting with _)
    - Calling dangerous built-ins by name
    """
    FORBIDDEN_NODES = (
        ast.Import, ast.ImportFrom, 
        ast.With, ast.Try, ast.ExceptHandler,
        ast.ClassDef, ast.Delete, ast.Global, ast.Nonlocal
    )
    
    FORBIDDEN_BUILTINS = {
        'eval', 'exec', 'open', 'compile', 'getattr', 'setattr', 
        'delattr', 'hasattr', 'globals', 'locals', 'vars', 'dir',
        'input', 'help', 'copyright', 'exit', 'quit'
    }

    def generic_visit(self, node):
        if isinstance(node, self.FORBIDDEN_NODES):
            raise CodeSafetyError(f"Forbidden syntax detected: {type(node).__name__}")
        super().generic_visit(node)

    def visit_Attribute(self, node):
        if node.attr.startswith('_'):
            raise CodeSafetyError(f"Access to private attribute '{node.attr}' is forbidden")
        self.generic_visit(node)

    def visit_Name(self, node):
        if node.id in self.FORBIDDEN_BUILTINS:
            raise CodeSafetyError(f"Call to forbidden built-in '{node.id}' detected")
        self.generic_visit(node)

class NeuralInterpreter:
    """
    Executes Python code natively using 'exec', injecting symbolic types 
    to build a computational graph (Native Tracing).
    
    Hardened with AST-based sandboxing and restricted execution context.
    """
    def __init__(self, model, memory=None):
        self.model = model
        self.memory = memory
        self.hard_logic = getattr(model, 'hard_logic', False)
        self._cache = {} # Bytecode cache
        # Pre-initialize math wrappers and base context
        self._base_ctx = self._get_context()

    def execute(self, code_str, inputs=None):
        """
        Execute python code string with given inputs.
        Inputs should be a dict of {name: SymbolicNode | tensor | float}.
        """
        # 1. Validate code safety via AST
        try:
            tree = ast.parse(code_str)
            visitor = SafeNodeVisitor()
            visitor.visit(tree)
        except Exception as e:
            if isinstance(e, CodeSafetyError):
                raise e
            raise CodeSafetyError(f"Failed to parse or validate code: {e}")

        # 2. Prepare execution context (copy from base)
        ctx = self._base_ctx.copy()
        
        if inputs:
            # Wrap inputs if they are raw numbers/tensors
            wrapped_inputs = {k: self._wrap(v) for k, v in inputs.items()}
            ctx.update(wrapped_inputs)
            
        # 3. Compile or retrieve from cache
        code_hash = hashlib.md5(code_str.encode()).hexdigest()
        if code_hash in self._cache:
            compiled_wrapper = self._cache[code_hash]
        else:
            # Wrap code in a function to capture return value
            indented_code = textwrap.indent(code_str, '    ')
            wrapper_str = f"""
def _main_():
{indented_code}

result = _main_()
"""
            compiled_wrapper = compile(wrapper_str, '<nous_trace>', 'exec')
            self._cache[code_hash] = compiled_wrapper
            
        # 4. Execute in restricted environment
        exec(compiled_wrapper, ctx)
        
        # 5. Return result
        res = ctx.get('result')
        return self._wrap(res)

    def _get_context(self):
        """Returns the dictionary of globals/locals for exec with restricted builtins."""
        
        # 1. Build restricted builtins
        safe_builtins = {}
        for b in SAFE_BUILTINS:
            if b in __builtins__ if isinstance(__builtins__, dict) else dir(__builtins__):
                if isinstance(__builtins__, dict):
                    safe_builtins[b] = __builtins__[b]
                else:
                    safe_builtins[b] = getattr(__builtins__, b)
        
        # Explicitly block __import__ to avoid KeyError while maintaining security
        # but ALLOW torch and math if requested (they are already whitelisted)
        def restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name in ['torch', 'math']:
                return torch if name == 'torch' else math
            raise CodeSafetyError(f"Forbidden syntax detected: Import ({name})")
        safe_builtins['__import__'] = restricted_import

        # 2. Build Safe Torch (Proxy object)
        class SafeTorch: pass
        safe_torch = SafeTorch()
        for op in SAFE_TORCH_OPS:
            if hasattr(torch, op):
                setattr(safe_torch, op, getattr(torch, op))

        # 3. Build Safe Math (Proxy object)
        class SafeMath: pass
        safe_math = SafeMath()
        for op in SAFE_MATH_OPS:
            if hasattr(math, op):
                setattr(safe_math, op, getattr(math, op))

        ctx = {
            '__builtins__': safe_builtins,
            'torch': safe_torch,
            'math': safe_math,
            'SymbolicNode': SymbolicNode,
            'ExprConst': ExprConst,
            'ExprVar': ExprVar,
            'ExprFunc': ExprFunc,
        }
        
        # Inject eager-capable math functions as top-level for convenience
        for name in SAFE_MATH_OPS:
            if hasattr(math, name) and not isinstance(getattr(math, name), (float, int)):
                ctx[name] = self._create_math_wrapper(name)
            
        # Expose Differentiable Standard Library (DSL)
        ctx['sigmoid'] = self._create_sigmoid_wrapper()
        ctx['soft_if'] = self._create_soft_if_wrapper()
        ctx['softmax'] = self._create_softmax_wrapper()
        ctx['soft_switch'] = self._create_soft_switch_wrapper()
        ctx['soft_index'] = self._create_soft_index_wrapper()
        ctx['soft_while'] = self._create_soft_while_wrapper()
        ctx['soft_map'] = self._create_soft_map_wrapper()
        ctx['soft_filter'] = self._create_soft_filter_wrapper()
        ctx['soft_enumerate'] = self._create_soft_enumerate_wrapper()
        ctx['soft_sort'] = self._create_soft_sort_wrapper()
        ctx['soft_embed'] = self._create_soft_embed_wrapper()
        
        # Data Analysis Primitives
        ctx['soft_sum'] = self._create_soft_sum_wrapper()
        ctx['soft_mean'] = self._create_soft_mean_wrapper()
        ctx['soft_var'] = self._create_soft_var_wrapper()
        ctx['soft_std'] = self._create_soft_std_wrapper()
        ctx['soft_cov'] = self._create_soft_cov_wrapper()
        ctx['soft_pearson'] = self._create_soft_pearson_wrapper()
        
        # Linear Algebra
        ctx['soft_dot'] = self._create_soft_dot_wrapper()
        ctx['soft_matmul'] = self._create_soft_matmul_wrapper()
        ctx['soft_norm'] = self._create_soft_norm_wrapper()
        
        # Selection / Optimization
        ctx['soft_max'] = self._create_soft_max_wrapper()
        ctx['soft_min'] = self._create_soft_min_wrapper()
        ctx['soft_argmax'] = self._create_soft_argmax_wrapper()
        ctx['soft_argmin'] = self._create_soft_argmin_wrapper()
        
        # Advanced Analysis
        ctx['soft_standardize'] = self._create_soft_standardize_wrapper()
        ctx['soft_linreg'] = self._create_soft_linreg_wrapper()
        ctx['soft_compile'] = self._create_soft_compile_wrapper()
        
        # Big Data / I/O (SUPERVISED)
        ctx['soft_read_csv'] = self._create_soft_read_csv_wrapper()
        ctx['soft_groupby_mean'] = self._create_soft_groupby_mean_wrapper()
        
        # Knowledge / Books
        ctx['soft_load_book'] = self._create_soft_load_book_wrapper()
        ctx['soft_search'] = self._create_soft_search_wrapper()
        
        # Inject Memory Operations if memory is attached
        if self.memory:
            ctx['mem_read'] = self.memory.read
            ctx['mem_write'] = self.memory.write
            ctx['mem_content_read'] = self.memory.content_addressing
            
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
             if torch.is_tensor(val):
                 return ExprConst(torch.sigmoid(val))
             return ExprConst(1.0 / (1.0 + math.exp(-float(val))))
           
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
            if isinstance(cond, SymbolicNode):
                return ExprIf(cond, true_val, false_val)
            
            # Eager concrete evaluation
            sig = self._create_sigmoid_wrapper()
            prob = sig(cond)
            
            # STE snapping for soft_if (concrete)
            if self.hard_logic:
                # Convert prob to tensor if symbolic for snapping decision
                p0 = prob
                if isinstance(prob, SymbolicNode):
                    p0 = prob.to_taylor(0.0, 1, self.model.hilbert)[0]
                
                p_hard = torch.round(p0)
                # STE: Use p_hard for forward, but keep grad from p0 (which is derived from prob)
                p_ste = p_hard + (p0 - p0.detach())
                prob = p_ste
            tv = true_val() if callable(true_val) else true_val
            fv = false_val() if callable(false_val) else false_val
            
            res = prob * tv + (1 - prob) * fv
            if torch.is_tensor(res) and not isinstance(res, SymbolicNode):
                return ExprConst(res)
            return res
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
                
                # STE snapping for soft_while
                if self.hard_logic:
                    p0 = p_continue
                    if isinstance(p_continue, SymbolicNode):
                        p0 = p_continue.to_taylor(0.0, 1, self.model.hilbert)[0]
                    
                    p_hard = torch.round(p0)
                    p_ste = p_hard + (p0 - p0.detach())
                    p_continue = p_ste
               
                # Compute next state from body
                next_state = body_fn(state)
                
                # Blend: state = p_continue * next_state + (1-p_continue) * state
                if isinstance(state, (list, tuple)):
                    blended = []
                    for s, ns in zip(state, next_state):
                        val = p_continue * ns + (1.0 - p_continue) * s
                        if torch.is_tensor(val) and not isinstance(val, SymbolicNode):
                            val = ExprConst(val)
                        blended.append(val)
                    state = type(state)(blended)
                else:
                    state = p_continue * next_state + (1.0 - p_continue) * state
                    if torch.is_tensor(state) and not isinstance(state, SymbolicNode):
                        state = ExprConst(state)
            return state
        
        return soft_while

    def _create_soft_map_wrapper(self):
        def soft_map(fn, collection):
            """
            Differentiable mapping.
            soft_map(fn, collection) -> [fn(x1), fn(x2), ...]
            """
            if isinstance(collection, ExprList):
                items = collection.elements
            elif isinstance(collection, (list, tuple)):
                items = collection
            else:
                raise TypeError(f"soft_map requires a collection, got {type(collection)}")
            
            return [fn(x) for x in items]
        return soft_map

    def _create_soft_filter_wrapper(self):
        sig = self._create_sigmoid_wrapper()
        def soft_filter(predicate, collection):
            """
            Differentiable filtering.
            Returns elements scaled by their predicate probability.
            Vectorized if collection is a tensor.
            """
            if torch.is_tensor(collection):
                # Vectorized apply
                probs = sig(predicate(collection))
                return probs * collection
                
            if isinstance(collection, ExprList):
                items = collection.elements
            elif isinstance(collection, (list, tuple)):
                items = collection
            else:
                raise TypeError(f"soft_filter requires a collection, got {type(collection)}")
            
            probs = [sig(predicate(x)) for x in items]
            return [p * x for p, x in zip(probs, items)]
        return soft_filter

    def _create_soft_enumerate_wrapper(self):
        def soft_enumerate(collection):
            """
            Differentiable enumeration.
            Returns (index, value) pairs.
            """
            if isinstance(collection, ExprList):
                items = collection.elements
            elif isinstance(collection, (list, tuple)):
                items = collection
            else:
                raise TypeError(f"soft_enumerate requires a collection, got {type(collection)}")
            
            return list(enumerate(items))
        return soft_enumerate

    def _create_soft_sort_wrapper(self):
        def soft_sort(collection, key=None, reverse=False):
            """
            Differentiable sorting.
            Uses attention-based permutation relaxation.
            """
            if isinstance(collection, ExprList):
                items = collection.elements
            elif isinstance(collection, (list, tuple)):
                items = collection
            else:
                raise TypeError(f"soft_sort requires a collection, got {type(collection)}")
            
            n = len(items)
            if n <= 1: return items
            
            # 1. Get scores (logits)
            if key is None:
                scores = items
            else:
                scores = [key(x) for x in items]
            
            # 2. Compute pairwise similarity between scores and implicit ranks
            # For simplicity, we'll use a relaxed sorting approach:
            # Each output j is a softmax over inputs i based on (score_i - ideal_score_j)
            
            # But what are 'ideal_score_j'? 
            # If we don't have them, we can use the relative ranking:
            # P_ij = softmax(score_i * temperature) ... this is just selection.
            
            # Better: Bitonic sort or other differentiable sort.
            # For this DSL, we'll implement a "Relational Sort":
            # Matrix M_ij = score_i - score_j
            # Rank_i = sum_j sigmoid(M_ij)
            # This gives an approximate rank [0, n-1] for each i.
            
            sig = self._create_sigmoid_wrapper()
            ranks = []
            for i in range(n):
                r_i = 0.0
                for j in range(n):
                    if i == j: continue
                    # p(score_i > score_j)
                    diff = scores[i] - scores[j]
                    if reverse:
                        diff = -diff
                    r_i = r_i + sig(diff)
                ranks.append(r_i)
            
            # 3. Use ranks to create attention matrix A_jk (prob that item j has rank k)
            # A_jk = softmax_over_j(-(rank_j - k)^2 / temperature)
            outputs = []
            for k in range(n):
                # Probability weights for each item i being at rank k
                # k is the target index [0, n-1]
                logits = [-(r_i - k)**2 for r_i in ranks]
                
                # Softmax over items
                weights = self._create_softmax_wrapper()(logits)
                
                # Weighted sum of items
                val_k = weights[0] * items[0]
                for i in range(1, n):
                    val_k = val_k + weights[i] * items[i]
                outputs.append(val_k)
            
            return outputs
        return soft_sort

    def _create_soft_embed_wrapper(self):
        def soft_embed(token, dim=32):
            """
            Differentiable embedding.
            """
            # If token is a list, stack them
            if isinstance(token, list):
                return torch.stack([soft_embed(t, dim) for t in token])
            
            # Simple deterministic embedding based on hashing + float conversion
            # In real LLM integration, this would call the shared embedding layer
            h = hashlib.md5(str(token).encode()).digest()
            # Convert hash to a 32-dim vector of floats [-1, 1]
            vec = []
            for i in range(min(dim, len(h))):
                val = (h[i] / 127.5) - 1.0
                vec.append(val)
            # Pad if needed
            while len(vec) < dim: vec.append(0.0)
                
            res = torch.tensor(vec, dtype=torch.float32, requires_grad=True)
            return res
        return soft_embed

    def _create_soft_compile_wrapper(self):
        def soft_compile(expr):
            """Compiles a SymbolicNode into a fast SymbolicProgram."""
            if hasattr(expr, 'compile'):
                return expr.compile()
            return expr
        return soft_compile

    def _create_soft_load_book_wrapper(self):
        def soft_load_book(text, chunk_size=100, memory_name='book'):
            # Simple chunking
            words = text.split()
            chunks = [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
            
            dim = 32
            mem = NeuralMemory(num_slots=len(chunks), slot_size=dim)
            embedder = self._create_soft_embed_wrapper()
            
            for i, chunk in enumerate(chunks):
                # We store the chunk index and the vector
                vec = embedder(chunk, dim=dim)
                # One-hot address weight for exact storage
                addr = torch.zeros(len(chunks))
                addr[i] = 10.0 # High logit
                mem.write(addr, vec)
            
            # Record it in the model or return as state
            return mem, chunks
        return soft_load_book


    def _create_soft_search_wrapper(self):
        def soft_search(query_text, memory_tuple, k=3):
            mem, chunks = memory_tuple
            embedder = self._create_soft_embed_wrapper()
            q_vec = embedder(query_text)
            
            # Semantic weight (differentiable!)
            weights = mem.top_k_attention(q_vec, k=k)
            
            # In a real system, we'd return a weighted sum of text embeddings 
            # Or strings. For this demo, let's return a summary of the indices found.
            top_indices = torch.topk(weights, k=k).indices
            relevant_chunks = [chunks[i.item()] for i in top_indices]
            
            return {
                'relevance': weights,
                'passages': relevant_chunks
            }
        return soft_search

    def _wrap(self, val):
        if isinstance(val, (int, float)): return ExprConst(float(val))
        if isinstance(val, SymbolicNode): return val
        if torch.is_tensor(val): 
            if val.numel() == 1:
                return ExprConst(val)
            return val
        if isinstance(val, (list, tuple)):
             return type(val)([self._wrap(v) for v in val])
        return val

    def _create_math_wrapper(self, name):
        """Creates a wrapper for a math function that supports symbolic & concrete execution."""
        def wrapper(x):
            # If x is SymbolicNode, we check if it's a constant
            if isinstance(x, SymbolicNode):
                if isinstance(x, ExprConst):
                    # Eager evaluation
                    val = x.value
                    if torch.is_tensor(val):
                        func = getattr(torch, name)
                        return ExprConst(func(val))
                    func = getattr(math, name)
                    return ExprConst(func(float(val)))
                # Otherwise purely symbolic
                return ExprFunc(name, x)
            
            # If x is concrete (int, float, tensor)
            if isinstance(x, (int, float)):
                func = getattr(math, name)
                return func(x)
            
            if torch.is_tensor(x):
                 # If tensor, we might want to return tensor or ExprConst
                 func = getattr(torch, name)
                 res = func(x)
                 if x.numel() == 1:
                     return ExprConst(res)
                 return res

            raise TypeError(f"Unsupported type for {name}: {type(x)}")
        return wrapper

    def _create_soft_sum_wrapper(self):
        def soft_sum(collection):
            if torch.is_tensor(collection):
                return collection.sum()
                
            if isinstance(collection, ExprList):
                elems = collection.elements
            elif torch.is_tensor(collection) and collection.dim() > 0:
                elems = [collection[i] for i in range(len(collection))]
            else:
                elems = collection
            
            if not elems: return 0.0
            
            res = elems[0]
            for i in range(1, len(elems)):
                res = res + elems[i]
            return self._wrap(res)
        return soft_sum

    def _create_soft_mean_wrapper(self):
        def soft_mean(collection):
            if torch.is_tensor(collection):
                return collection.mean()
            n = len(collection) if not isinstance(collection, (int, float)) else 1
            if n == 0: return 0.0
            s = self._create_soft_sum_wrapper()(collection)
            return s / n
        return soft_mean

    def _create_soft_var_wrapper(self):
        def soft_var(collection):
            if torch.is_tensor(collection):
                return collection.var(unbiased=True)
            n = len(collection) if not isinstance(collection, (int, float)) else 1
            if n < 2: return 0.0
            m = self._create_soft_mean_wrapper()(collection)
            
            def get_elems(c):
                if isinstance(c, ExprList): return c.elements
                if torch.is_tensor(c) and c.dim() > 0: return [c[i] for i in range(len(c))]
                return c
            
            elems = get_elems(collection)
                
            sq_diffs = [(x - m)**2 for x in elems]
            return self._create_soft_sum_wrapper()(sq_diffs) / (n - 1)
        return soft_var

    def _create_soft_std_wrapper(self):
        def soft_std(collection):
            v = self._create_soft_var_wrapper()(collection)
            # Use ExprFunc('sqrt', v) if symbolic
            if isinstance(v, SymbolicNode):
                return ExprFunc('sqrt', v)
            if torch.is_tensor(v):
                return torch.sqrt(v)
            return math.sqrt(v)
        return soft_std

    def _create_soft_cov_wrapper(self):
        def soft_cov(x_coll, y_coll):
            n = len(x_coll)
            if n < 2: return 0.0
            mx = self._create_soft_mean_wrapper()(x_coll)
            my = self._create_soft_mean_wrapper()(y_coll)
            
            def get_elems(c):
                if isinstance(c, ExprList): return c.elements
                if torch.is_tensor(c) and c.dim() > 0: return [c[i] for i in range(len(c))]
                return c
            
            ex = get_elems(x_coll)
            ey = get_elems(y_coll)
            
            products = [(ex[i] - mx) * (ey[i] - my) for i in range(n)]
            return self._create_soft_sum_wrapper()(products) / (n - 1)
        return soft_cov

    def _create_soft_pearson_wrapper(self):
        def soft_pearson(x_coll, y_coll):
            cov = self._create_soft_cov_wrapper()(x_coll, y_coll)
            sx = self._create_soft_std_wrapper()(x_coll)
            sy = self._create_soft_std_wrapper()(y_coll)
            return cov / (sx * sy + 1e-12)
        return soft_pearson

    def _create_soft_dot_wrapper(self):
        def soft_dot(a, b):
            def get_elems(c):
                if isinstance(c, ExprList): return c.elements
                if torch.is_tensor(c) and c.dim() > 0: return [c[i] for i in range(len(c))]
                return c
            ea, eb = get_elems(a), get_elems(b)
            if len(ea) != len(eb): raise ValueError("Dot product requires equal length collections.")
            products = [ea[i] * eb[i] for i in range(len(ea))]
            return self._create_soft_sum_wrapper()(products)
        return soft_dot

    def _create_soft_matmul_wrapper(self):
        def soft_matmul(A, B):
            # For simplicity, if inputs are already tensors, use torch.matmul
            # Otherwise we'd have to implement symbolic matmul
            if torch.is_tensor(A) and torch.is_tensor(B):
                return torch.matmul(A, B)
            # If symbolic, we currently only support matrix-vector via dot product
            if not isinstance(A, list): raise TypeError("soft_matmul requires lists of lists for symbolic matrices.")
            res = [self._create_soft_dot_wrapper()(row, B) for row in A]
            return res
        return soft_matmul

    def _create_soft_norm_wrapper(self):
        def soft_norm(collection, p=2):
            s_sum = self._create_soft_sum_wrapper()
            powered = [x**p if not isinstance(x, (int, float)) or x >= 0 else (-x)**p for x in collection]
            res = s_sum(powered) ** (1.0/p)
            return res
        return soft_norm

    def _create_soft_max_wrapper(self):
        def soft_max(collection, temp=0.1):
            if not collection: return 0.0
            # softmax weighting
            logits = torch.stack([self.model.expand(self._wrap(x), 0.0)[0] for x in collection])
            weights = torch.softmax(logits / temp, dim=0)
            res = 0.0
            for i, w in enumerate(weights):
                res = res + w * collection[i]
            return res
        return soft_max

    def _create_soft_min_wrapper(self):
        def soft_min(collection, temp=0.1):
            # Same as soft_max with negative logits
            neg_coll = [-x for x in collection]
            res = self._create_soft_max_wrapper()(neg_coll, temp)
            return -res
        return soft_min

    def _create_soft_argmax_wrapper(self):
        def soft_argmax(collection, temp=0.1):
            logits = torch.stack([self.model.expand(self._wrap(x), 0.0)[0] for x in collection])
            return torch.softmax(logits / temp, dim=0)
        return soft_argmax

    def _create_soft_argmin_wrapper(self):
        def soft_argmin(collection, temp=0.1):
            logits = torch.stack([self.model.expand(self._wrap(x), 0.0)[0] for x in collection])
            return torch.softmax(-logits / temp, dim=0)
        return soft_argmin

    def _create_soft_standardize_wrapper(self):
        def soft_standardize(collection):
            m = self._create_soft_mean_wrapper()(collection)
            s = self._create_soft_std_wrapper()(collection)
            return [(x - m) / (s + 1e-12) for x in collection]
        return soft_standardize

    def _create_soft_linreg_wrapper(self):
        def soft_linreg(x_coll, y_coll):
            # y = mx + c
            # Simple linear regression formulas:
            # m = cov(x,y) / var(x)
            # c = mean(y) - m * mean(x)
            m_val = self._create_soft_cov_wrapper()(x_coll, y_coll) / (self._create_soft_var_wrapper()(x_coll) + 1e-12)
            c_val = self._create_soft_mean_wrapper()(y_coll) - m_val * self._create_soft_mean_wrapper()(x_coll)
            return {'slope': m_val, 'intercept': c_val}
        return soft_linreg

    def _create_soft_read_csv_wrapper(self):
        def soft_read_csv(filepath):
            columns = {}
            with open(filepath, 'r') as f:
                reader = csv.DictReader(f)
                headers = reader.fieldnames
                for h in headers: columns[h] = []
                for row in reader:
                    for h in headers:
                        try:
                            val = float(row[h])
                        except ValueError:
                            val = row[h] # Keep as string if not numeric
                        columns[h].append(val)
            
            # Convert numeric columns to tensors
            for h in headers:
                if all(isinstance(x, (int, float)) for x in columns[h]):
                    columns[h] = torch.tensor(columns[h], dtype=torch.float32)
            return columns
        return soft_read_csv

    def _create_soft_groupby_mean_wrapper(self):
        def soft_groupby_mean(keys, values):
            # keys: tensor of IDs (integers)
            # values: tensor of data
            # Returns: dict of {key: mean_value}
            # Differentiable via indexing
            unique_keys = torch.unique(keys)
            results = {}
            for k in unique_keys:
                mask = (keys == k).float()
                # Differentiable mean over mask
                count = mask.sum()
                if count > 0:
                    results[k.item()] = (values * mask).sum() / count
                else:
                    results[k.item()] = 0.0
            return results
        return soft_groupby_mean

    def _wrap(self, val):
        if isinstance(val, (int, float)): return ExprConst(float(val))
        if isinstance(val, SymbolicNode): return val
        if torch.is_tensor(val): 
            if val.numel() == 1:
                return ExprConst(val)
            return val
        if isinstance(val, list):
             return [self._wrap(v) for v in val]
        return val
