import torch
import math

class SymbolicNode:
    """Base class for all symbolic nodes in the expression DAG."""
    def __add__(self, other): 
        other = self._wrap(other)
        if isinstance(self, ExprConst) and isinstance(other, ExprConst): return ExprConst(self.value + other.value)
        # Simplification: x + 0 = x
        if isinstance(other, ExprConst) and other.value == 0: return self
        if isinstance(self, ExprConst) and self.value == 0: return other
        return ExprAdd(self, other)
    
    def __radd__(self, other): 
        other = self._wrap(other)
        if isinstance(self, ExprConst) and isinstance(other, ExprConst): return ExprConst(other.value + self.value)
        if isinstance(other, ExprConst) and other.value == 0: return self
        if isinstance(self, ExprConst) and self.value == 0: return other
        return ExprAdd(other, self)
    
    def __sub__(self, other): 
        other = self._wrap(other)
        if isinstance(self, ExprConst) and isinstance(other, ExprConst): return ExprConst(self.value - other.value)
        # Simplification: x - 0 = x, x - x = 0
        if isinstance(other, ExprConst) and other.value == 0: return self
        if self is other: return ExprConst(0.0)
        return ExprSub(self, other)
    
    def __rsub__(self, other): 
        other = self._wrap(other)
        if isinstance(self, ExprConst) and isinstance(other, ExprConst): return ExprConst(other.value - self.value)
        if isinstance(self, ExprConst) and self.value == 0: return other
        return ExprSub(other, self)
    
    def __mul__(self, other): 
        other = self._wrap(other)
        if isinstance(self, ExprConst) and isinstance(other, ExprConst): return ExprConst(self.value * other.value)
        # Simplification: x * 0 = 0, x * 1 = x
        if isinstance(other, ExprConst):
            if other.value == 0: return ExprConst(0.0)
            if other.value == 1: return self
        if isinstance(self, ExprConst):
            if self.value == 0: return ExprConst(0.0)
            if self.value == 1: return other
        
        # Support for tensor scaling
        if torch.is_tensor(other) and other.numel() == 1:
            return self * ExprConst(float(other.item()))
            
        return ExprMul(self, other)
    
    def __rmul__(self, other): 
        other = self._wrap(other)
        if isinstance(self, ExprConst) and isinstance(other, ExprConst): return ExprConst(other.value * self.value)
        if isinstance(other, ExprConst):
            if other.value == 0: return ExprConst(0.0)
            if other.value == 1: return self
        if isinstance(self, ExprConst):
            if self.value == 0: return ExprConst(0.0)
            if self.value == 1: return other
        return ExprMul(other, self)
    
    def __truediv__(self, other): 
        other = self._wrap(other)
        if isinstance(self, ExprConst) and isinstance(other, ExprConst): return ExprConst(self.value / other.value)
        # Simplification: 0 / x = 0, x / 1 = x, x / x = 1
        if isinstance(self, ExprConst) and self.value == 0: return ExprConst(0.0)
        if isinstance(other, ExprConst) and other.value == 1: return self
        if self is other: return ExprConst(1.0)
        return ExprDiv(self, other)
    
    def __rtruediv__(self, other): 
        other = self._wrap(other)
        if isinstance(self, ExprConst) and isinstance(other, ExprConst): return ExprConst(other.value / self.value)
        if isinstance(other, ExprConst) and other.value == 0: return ExprConst(0.0)
        if isinstance(self, ExprConst) and self.value == 1: return other
        return ExprDiv(other, self)
    
    def __pow__(self, other): 
        other = self._wrap(other)
        if isinstance(self, ExprConst) and isinstance(other, ExprConst): return ExprConst(self.value ** other.value)
        return ExprPow(self, other)
    
    def __neg__(self): 
        if isinstance(self, ExprConst): return ExprConst(-self.value)
        return ExprMul(ExprConst(-1.0), self)

    def __lt__(self, other):
        other = self._wrap(other)
        # Returns (other - self) as a logit for p(self < other)
        return other - self

    def __gt__(self, other):
        other = self._wrap(other)
        # Returns (self - other) as a logit for p(self > other)
        return self - other

    def __le__(self, other):
        # Approximation: roughly same as < for soft logic, 
        # but we could add a small margin if needed.
        return self.__lt__(other)

    def __ge__(self, other):
        return self.__gt__(other)

    def __eq__(self, other):
        other = self._wrap(other)
        # returns -abs(self - other) as a logit for p(self == other)
        # High when diff is 0, low otherwise.
        return -ExprFunc('abs', self - other)

    def __ne__(self, other):
        return -self.__eq__(other)

    def flatten(self, memo=None, ops=None, constants=None, vars_dict=None):
        """
        Iterative topological sort to convert tree to linear ops.
        Avoids RecursionError for deep graphs.
        """
        if memo is None: memo = {}
        if ops is None: ops = []
        if constants is None: constants = []
        if vars_dict is None: vars_dict = {}
        
        # 1. Collect all nodes using iterative post-order traversal
        stack = [(self, False)] # (node, children_visited)
        topo_order = []
        
        while stack:
            curr, visited = stack.pop()
            if id(curr) in memo:
                continue
            
            if visited:
                topo_order.append(curr)
                # Mark as visited in memo (we'll fill the index later)
                # We use a temporary value to avoid re-traversing
                memo[id(curr)] = None 
            else:
                stack.append((curr, True))
                # Add children to stack
                if isinstance(curr, (ExprAdd, ExprSub, ExprMul, ExprDiv, ExprPow)):
                    stack.append((curr.right, False))
                    stack.append((curr.left, False))
                elif isinstance(curr, ExprFunc):
                    stack.append((curr.arg, False))
                elif isinstance(curr, ExprList):
                    for el in reversed(curr.elements):
                        stack.append((el, False))
                elif isinstance(curr, ExprIf):
                    stack.append((curr.false_branch, False))
                    stack.append((curr.true_branch, False))
                    stack.append((curr.cond, False))

        # 2. Process nodes in topological order
        # We need to clear the temporary None values from memo
        for k in list(memo.keys()):
            if memo[k] is None: del memo[k]

        for node in topo_order:
            if id(node) not in memo:
                res_idx = node._flatten_impl(memo, ops, constants, vars_dict)
                memo[id(node)] = res_idx
        
        return memo[id(self)]

    def _flatten_impl(self, memo, ops, constants, vars_dict):
        raise NotImplementedError()

    def _wrap(self, other):
        if isinstance(other, (int, float)): return ExprConst(float(other))
        if torch.is_tensor(other):
            # Preserve gradient if it's a tensor, wrap even if vector
            return ExprConst(other)
        return other

    def compile(self):
        """Compiles the symbolic tree into a SymbolicProgram for Turbo execution."""
        ops = []
        constants = []
        vars_dict = {}
        memo = {}
        final_idx = self.flatten(memo, ops, constants, vars_dict)
        return SymbolicProgram(constants, list(vars_dict.keys()), ops, final_idx)

    def to_taylor(self, center, max_terms, hilbert):
        # 'center' can be a float/tensor (scalar context) or a dict {name: val}
        raise NotImplementedError()

    def diff(self, var='x'):
        raise NotImplementedError()

class ExprConst(SymbolicNode):
    def __init__(self, value):
        self.value = value
    def to_taylor(self, center, max_terms, hilbert):
        # To preserve gradients if self.value is a tensor:
        v = self.value
        if not torch.is_tensor(v):
            v = torch.tensor(float(v), dtype=torch.float32)
        
        # Result shape: v.shape + (max_terms,)
        # We want to stack v at index 0, and zeros elsewhere along the last dim.
        
        # Add a dimension for terms
        v_expanded = v.unsqueeze(-1) 
        
        if max_terms > 1:
            zeros_shape = v.shape + (max_terms - 1,)
            zeros = torch.zeros(zeros_shape, dtype=v.dtype, device=v.device)
            return torch.cat([v_expanded, zeros], dim=-1)
        else:
            return v_expanded

    def diff(self, var='x'):
        return ExprConst(0.0)
    def _flatten_impl(self, memo, ops, constants, vars):
        idx = len(constants)
        constants.append(self.value)
        return ('const', idx)

    def __repr__(self): return str(self.value)

class ExprVar(SymbolicNode):
    def __init__(self, name='x'):
        self.name = name
    def to_taylor(self, center, max_terms, hilbert):
        # Resolve center for this variable
        c_val = 0.0
        if isinstance(center, dict):
            c_val = center.get(self.name, 0.0) # Default 0.0
            # If default is used, maybe we should treat it as 0?
        else:
            # Assume center is for 'x' or the primary variable
            # If self.name != 'x', maybe it should be considered parameter/constant?
            # Existing behavior assumed 1 variable 'x'.
            if self.name == 'x':
                c_val = center
            else:
                # If expanding w.r.t 'x', then 'y' is constant?
                # For safety, let's assume 'x' is default param.
                # If expanding w.r.t 'x' at c, then ExprVar('x') is (x-c)+c.
                # ExprVar('y') is just y (constant).
                # But here we want to EVALUATE 'y'.
                # Let's assume if center is scalar, it applies to 'x'.
                if hasattr(center, 'device'): # Tensor
                    c_val = center if self.name == 'x' else 0.0 # TODO: handle other vars?
                else: 
                     c_val = center if self.name == 'x' else 0.0

        res = torch.zeros(max_terms, dtype=torch.float32)
        if torch.is_tensor(c_val):
             res = res.to(c_val.device) # Match device
             
        res[0] = c_val
        if max_terms > 1:
            # The derivative term.
            # If we are expanding w.r.t THIS variable, coeff 1 is 1.0.
            # If we are evaluating multiple variables, we are effectively setting them to (v + epsilon).
            # But wait, this is Taylor expansion.
            # If result is f(a,b), we want value.
            # If we want GRADIENT, PyTorch handles it through res[0].
            # Do we need higher order terms?
            # For "Value" computation, max_terms=1 is sufficient?
            # But the symbolic engine computes Taylor series continuously.
            # If we pass max_terms=1, we get value.
            
            # Let's keep 1.0 at index 1 for "x" if center is scalar.
            # If center is dict, do we treat all as variables?
            # Yes, if we want to differentiate w.r.t any of them.
            # But Taylor series is 1D (powers of one variable t).
            # This is ambiguous.
            
            # For System 2 Experiment, we just want to propagate the Value (0-th term) and Gradients.
            # So res[0] matters. res[1+] matters if we have differentiation in the script.
            # If script has code.diff(), it uses Symbolic differentiation logic.
            # If we just execute, we don't use .diff().
            # WE ONLY USE to_taylor for evaluation.
            # So res[0] is critical. res[1] is irrelevant for pure value.
            res[1] = 1.0 
        return res
    def _flatten_impl(self, memo, ops, constants, vars):
        if self.name not in vars:
            vars[self.name] = len(vars)
        return ('var', self.name)

    def diff(self, var='x'):
        return ExprConst(1.0) if self.name == var else ExprConst(0.0)
    def __repr__(self): return self.name

class ExprAdd(SymbolicNode):
    def __init__(self, left, right):
        self.left, self.right = left, right
    def to_taylor(self, center, max_terms, hilbert):
        return self.left.to_taylor(center, max_terms, hilbert) + self.right.to_taylor(center, max_terms, hilbert)
    def _flatten_impl(self, memo, ops, constants, vars):
        l_idx = self.left.flatten(memo, ops, constants, vars)
        r_idx = self.right.flatten(memo, ops, constants, vars)
        res_idx = len(ops)
        ops.append(('add', l_idx, r_idx, res_idx))
        return res_idx

    def diff(self, var='x'):
        return ExprAdd(self.left.diff(var), self.right.diff(var))
    def __repr__(self): return f"({self.left} + {self.right})"

class ExprSub(SymbolicNode):
    def __init__(self, left, right):
        self.left, self.right = left, right
    def to_taylor(self, center, max_terms, hilbert):
        return self.left.to_taylor(center, max_terms, hilbert) - self.right.to_taylor(center, max_terms, hilbert)
    def _flatten_impl(self, memo, ops, constants, vars):
        l_idx = self.left.flatten(memo, ops, constants, vars)
        r_idx = self.right.flatten(memo, ops, constants, vars)
        res_idx = len(ops)
        ops.append(('sub', l_idx, r_idx, res_idx))
        return res_idx

    def diff(self, var='x'):
        return ExprSub(self.left.diff(var), self.right.diff(var))
    def __repr__(self): return f"({self.left} - {self.right})"

class ExprMul(SymbolicNode):
    def __init__(self, left, right):
        self.left, self.right = left, right
    def to_taylor(self, center, max_terms, hilbert):
        l_t = self.left.to_taylor(center, max_terms, hilbert)
        r_t = self.right.to_taylor(center, max_terms, hilbert)
        return hilbert.multiply(l_t, r_t)
    def _flatten_impl(self, memo, ops, constants, vars):
        l_idx = self.left.flatten(memo, ops, constants, vars)
        r_idx = self.right.flatten(memo, ops, constants, vars)
        res_idx = len(ops)
        ops.append(('mul', l_idx, r_idx, res_idx))
        return res_idx

    def diff(self, var='x'):
        # (uv)' = u'v + uv'
        return ExprAdd(ExprMul(self.left.diff(var), self.right), ExprMul(self.left, self.right.diff(var)))
    def __repr__(self): return f"({self.left} * {self.right})"

class ExprPow(SymbolicNode):
    def __init__(self, base, exp):
        self.base, self.exp = base, exp
    def to_taylor(self, center, max_terms, hilbert):
        if not isinstance(self.exp, ExprConst):
            raise ValueError("Only constant powers supported for Taylor expansion currently.")
        p = int(self.exp.value)
        b_t = self.base.to_taylor(center, max_terms, hilbert)
        res = b_t
        for _ in range(p - 1):
            res = hilbert.multiply(res, b_t)
        return res
    def diff(self, var='x'):
        # (u^n)' = n * u^(n-1) * u'
        if isinstance(self.exp, ExprConst):
            n = self.exp.value
            return ExprMul(ExprMul(ExprConst(n), ExprPow(self.base, ExprConst(n-1))), self.base.diff(var))
        raise NotImplementedError("General power differentiation not implemented")
    def __repr__(self): return f"({self.base}**{self.exp})"

class ExprDiv(SymbolicNode):
    def __init__(self, left, right):
        self.left, self.right = left, right
    def to_taylor(self, center, max_terms, hilbert):
        if isinstance(self.right, ExprConst):
            return self.left.to_taylor(center, max_terms, hilbert) / self.right.value
        # Use full polynomial division
        l_t = self.left.to_taylor(center, max_terms, hilbert)
        r_t = self.right.to_taylor(center, max_terms, hilbert)
        return hilbert.divide(l_t, r_t)
    def _flatten_impl(self, memo, ops, constants, vars):
        l_idx = self.left.flatten(memo, ops, constants, vars)
        r_idx = self.right.flatten(memo, ops, constants, vars)
        res_idx = len(ops)
        ops.append(('div', l_idx, r_idx, res_idx))
        return res_idx

    def diff(self, var='x'):
        # (u/v)' = (u'v - uv') / v^2
        return ExprDiv(ExprSub(ExprMul(self.left.diff(var), self.right), ExprMul(self.left, self.right.diff(var))), ExprPow(self.right, ExprConst(2)))
    def __repr__(self): return f"({self.left} / {self.right})"

class ExprFunc(SymbolicNode):
    def __init__(self, name, arg):
        self.name, self.arg = name, arg
    def to_taylor(self, center, max_terms, hilbert):
        g_coeffs = self.arg.to_taylor(center, max_terms, hilbert)
        return self._to_taylor_from_coeffs(self.name, g_coeffs, center, max_terms, hilbert)

    @staticmethod
    def _to_taylor_from_coeffs(name, g_coeffs, center, max_terms, hilbert):
        L = g_coeffs[0] # Keep it as a tensor
        
        # For simplicity in this implementation, we handle common identities at L=0
        # If L != 0, we can use the identity f(x+L) if available
        if name == 'exp':
            base = hilbert.get_taylor_exp()[:max_terms]
            # exp(L + delta) = exp(L) * exp(delta)
            multiplier = torch.exp(L)
            combined = base * multiplier
            g_shifted = g_coeffs.clone()
            g_shifted[0] = 0.0 # Clear L so we expand f at 0 with delta
            res = hilbert.compose(combined, g_shifted)
            
            val0 = torch.exp(L)
            zeros = torch.zeros(max_terms - 1, dtype=val0.dtype, device=val0.device)
            res0_full = torch.cat([val0.unsqueeze(0) if val0.dim()==0 else val0, zeros])
            return res0_full + (res - res[0:1])
        
        if name == 'sin':
            sin_base = hilbert.get_taylor_sin()[:max_terms]
            cos_base = hilbert.get_taylor_cos()[:max_terms]
            # sin(L + delta) = sin(L)cos(delta) + cos(L)sin(delta)
            combined = sin_base * torch.cos(L) + cos_base * torch.sin(L)
            g_shifted = g_coeffs.clone()
            g_shifted[0] = 0.0
            res = hilbert.compose(combined, g_shifted)
            
            val0 = torch.sin(L)
            zeros = torch.zeros(max_terms - 1, dtype=val0.dtype, device=val0.device)
            res0_full = torch.cat([val0.unsqueeze(0) if val0.dim()==0 else val0, zeros])
            return res0_full + (res - res[0:1])

        if name == 'cos':
            sin_base = hilbert.get_taylor_sin()[:max_terms]
            cos_base = hilbert.get_taylor_cos()[:max_terms]
            # cos(L + delta) = cos(L)cos(delta) - sin(L)sin(delta)
            combined = cos_base * torch.cos(L) - sin_base * torch.sin(L)
            g_shifted = g_coeffs.clone()
            g_shifted[0] = 0.0
            res = hilbert.compose(combined, g_shifted)
            
            val0 = torch.cos(L)
            zeros = torch.zeros(max_terms - 1, dtype=val0.dtype, device=val0.device)
            res0_full = torch.cat([val0.unsqueeze(0) if val0.dim()==0 else val0, zeros])
            return res0_full + (res - res[0:1])

        if name == 'sinh':
            sinh_base = hilbert.get_taylor_sinh()[:max_terms]
            cosh_base = hilbert.get_taylor_cosh()[:max_terms]
            # sinh(L + delta) = sinh(L)cosh(delta) + cosh(L)sinh(delta)
            combined = sinh_base * torch.cosh(L) + cosh_base * torch.sinh(L)
            g_shifted = g_coeffs.clone()
            g_shifted[0] = 0.0
            res = hilbert.compose(combined, g_shifted)
            
            val0 = torch.sinh(L)
            zeros = torch.zeros(max_terms - 1, dtype=val0.dtype, device=val0.device)
            res0_full = torch.cat([val0.unsqueeze(0) if val0.dim()==0 else val0, zeros])
            return res0_full + (res - res[0:1])

        if name == 'cosh':
            sinh_base = hilbert.get_taylor_sinh()[:max_terms]
            cosh_base = hilbert.get_taylor_cosh()[:max_terms]
            # cosh(L + delta) = cosh(L)cosh(delta) + sinh(L)sinh(delta)
            combined = cosh_base * torch.cosh(L) + sinh_base * torch.sinh(L)
            g_shifted = g_coeffs.clone()
            g_shifted[0] = 0.0
            res = hilbert.compose(combined, g_shifted)
            
            val0 = torch.cosh(L)
            zeros = torch.zeros(max_terms - 1, dtype=val0.dtype, device=val0.device)
            res0_full = torch.cat([val0.unsqueeze(0) if val0.dim()==0 else val0, zeros])
            return res0_full + (res - res[0:1])

        if name == 'sqrt':
            # sqrt(x) expansion around L: sqrt(L) * sqrt(1 + (x-L)/L)
            if L.item() < 0:
                raise ValueError(f"sqrt(x) requires x >= 0, got x(0)={L.item()}")
            
            # Use a small eps for L if near zero to avoid div by zero in expansion
            # but keep it differentiable
            L_safe = L if L.abs() > 1e-12 else torch.tensor(1e-12, device=L.device, dtype=L.dtype)
            
            base = hilbert.get_taylor_sqrt1p()[:max_terms]
            g_scaled = g_coeffs.clone()
            g_scaled[0] = 0.0 # delta = x - L
            g_scaled = g_scaled / L_safe
            res = hilbert.compose(base, g_scaled)
            res = res * torch.sqrt(L_safe)
            
            val0 = torch.sqrt(L)
            zeros = torch.zeros(max_terms - 1, dtype=val0.dtype, device=val0.device)
            res0_full = torch.cat([val0.unsqueeze(0) if val0.dim()==0 else val0, zeros])
            return res0_full + (res - res[0:1])

        if name == 'log':
            # log(x) expansion around L: log(L) + log(1 + (x-L)/L)
            if L.item() <= 0:
                raise ValueError(f"log(x) requires x > 0, got x(0)={L.item()}")
            
            base = hilbert.get_taylor_log1p()[:max_terms]
            g_scaled = g_coeffs.clone()
            g_scaled[0] = 0.0
            g_scaled = g_scaled / L
            res = hilbert.compose(base, g_scaled)
            # Add log(L) to the constant term
            val0 = res[0] + torch.log(L)
            zeros = torch.zeros(max_terms - 1, dtype=val0.dtype, device=val0.device)
            res0_full = torch.cat([val0.unsqueeze(0) if val0.dim()==0 else val0, zeros])
            return res0_full + (res - res[0:1]) # res[0:1] is constant term node

        if name == 'abs':
            # Use L.item() only for BRANCH selection. 
            # The result itself will be a differentiable function of g_coeffs.
            val_L = L.item()
            if val_L > 1e-9:
                return g_coeffs
            elif val_L < -1e-9:
                return -g_coeffs
            else:
                base = hilbert.get_identity('abs')[:max_terms]
                g_shifted = g_coeffs.clone()
                g_shifted[0] = 0.0
                return hilbert.compose(base, g_shifted)

        if name == 'sigmoid':
            # s(x) = 1 / (1 + exp(-x))
            val_L = L.item()
            if abs(val_L) < 1e-9:
                base = hilbert.get_taylor_sigmoid()[:max_terms]
                g_shifted = g_coeffs.clone()
                g_shifted[0] = 0.0
                return hilbert.compose(base, g_shifted)
            else:
                # sigmoid(L + delta) = 1 / (1 + exp(-L) * exp(-delta))
                exp_neg_L = torch.exp(-L)
                exp_base = hilbert.get_taylor_exp()[:max_terms]
                # Delta is g_coeffs[1:]. So delta shifted by L=0
                g_delta_neg = g_coeffs.clone()
                g_delta_neg[0] = 0.0
                g_delta_neg = -g_delta_neg
                
                exp_delta_neg = hilbert.compose(exp_base, g_delta_neg)
                denom = 1.0 + exp_neg_L * exp_delta_neg
                res = 1.0 / denom
                
            # Final safety: Ensure constant term is exactly f(L) and differentiable
            # We overwrite res[0] with the differentiable torch.sigmoid(L)
            # but we must do it carefully to avoid in-place issues if needed.
            # However, res is already ours here.
            val0 = torch.sigmoid(L)
            zeros = torch.zeros(max_terms - 1, dtype=val0.dtype, device=val0.device)
            res0_full = torch.cat([val0.unsqueeze(0) if val0.dim()==0 else val0, zeros])
            return res0_full + (res - res[0:1])

        # For tan and tanh, we can use the identity at 0 if we shift the input.
        # But we need the identity for tan(L + delta).
        # Better: use the quotient sin(L+d)/cos(L+d) or sinh(L+d)/cosh(L+d)
        if name == 'tan':
            # tan(u) = sin(u)/cos(u)
            sin_u = ExprFunc._to_taylor_from_coeffs('sin', g_coeffs, center, max_terms, hilbert)
            cos_u = ExprFunc._to_taylor_from_coeffs('cos', g_coeffs, center, max_terms, hilbert)
            return hilbert.divide(sin_u, cos_u)

        if name == 'tanh':
            # tanh(u) = sinh(u)/cosh(u)
            sinh_u = ExprFunc._to_taylor_from_coeffs('sinh', g_coeffs, center, max_terms, hilbert)
            cosh_u = ExprFunc._to_taylor_from_coeffs('cosh', g_coeffs, center, max_terms, hilbert)
            return hilbert.divide(sinh_u, cosh_u)

        # Fallback to standard identity at L=0 for others
        base = hilbert.get_identity(name=name)
        return hilbert.compose(base, g_coeffs)

    def _flatten_impl(self, memo, ops, constants, vars):
        a_idx = self.arg.flatten(memo, ops, constants, vars)
        res_idx = len(ops)
        ops.append(('func', self.name, a_idx, res_idx))
        return res_idx

    def diff(self, var='x'):
        # Chain rule: f(g(x))' = f'(g(x)) * g'(x)
        if self.name == 'exp': return ExprMul(ExprFunc('exp', self.arg), self.arg.diff(var))
        if self.name == 'sin': return ExprMul(ExprFunc('cos', self.arg), self.arg.diff(var))
        if self.name == 'cos': return ExprMul(ExprMul(ExprConst(-1.0), ExprFunc('sin', self.arg)), self.arg.diff(var))
        if self.name == 'sinh': return ExprMul(ExprFunc('cosh', self.arg), self.arg.diff(var))
        if self.name == 'cosh': return ExprMul(ExprFunc('sinh', self.arg), self.arg.diff(var))
        if self.name == 'tanh':
            # tanh'(x) = 1 - tanh^2(x)
            t = ExprFunc('tanh', self.arg)
            dt = ExprSub(ExprConst(1.0), ExprMul(t, t))
            return ExprMul(dt, self.arg.diff(var))
        if self.name == 'tan':
            # tan'(x) = 1 + tan^2(x)
            t = ExprFunc('tan', self.arg)
            dt = ExprAdd(ExprConst(1.0), ExprMul(t, t))
            return ExprMul(dt, self.arg.diff(var))
        if self.name == 'log':
            # log'(x) = 1/x
            return ExprMul(ExprDiv(ExprConst(1.0), self.arg), self.arg.diff(var))
        if self.name == 'sqrt':
            # sqrt'(x) = 1 / (2*sqrt(x))
            denom = ExprMul(ExprConst(2.0), ExprFunc('sqrt', self.arg))
            return ExprMul(ExprDiv(ExprConst(1.0), denom), self.arg.diff(var))
        if self.name == 'sigmoid':
            # s'(x) = s(x)*(1-s(x)) * x'
            s = ExprFunc('sigmoid', self.arg)
            ds = ExprMul(s, ExprSub(ExprConst(1.0), s))
            return ExprMul(ds, self.arg.diff(var))
            
        raise NotImplementedError(f"Differentiation for function '{self.name}' not implemented")
    def __repr__(self): return f"{self.name}({self.arg})"
class ExprList(SymbolicNode):
    """
    A symbolic representation of a list of symbolic nodes.
    Supports soft indexing and differentiable collection operations.
    """
    def __init__(self, elements):
        self.elements = elements # List of SymbolicNodes or values

    def __getitem__(self, index):
        # If index is an integer, return the element
        if isinstance(index, int):
            return self.elements[index]
        # If index is symbolic, use soft_index (which must be provided by interpreter)
        # But we don't have back-reference to interpreter here.
        # We'll rely on the interpreter to handle the __getitem__ call 
        # or we'll return a special ExprIndex node.
        return ExprIndex(self, index)

    def __len__(self):
        return len(self.elements)

    def to_taylor(self, center, max_terms, hilbert):
        # A list doesn't have a single Taylor expansion.
        # This shouldn't be called directly.
        raise TypeError("ExprList cannot be converted to Taylor series directly.")

    def _flatten_impl(self, memo, ops, constants, vars_dict):
        # Elements are flattened individually
        indices = [el.flatten(memo, ops, constants, vars_dict) if isinstance(el, SymbolicNode) else self._wrap(el).flatten(memo, ops, constants, vars_dict) for el in self.elements]
        res_idx = len(ops)
        ops.append(('list', indices, res_idx))
        return res_idx

    def __repr__(self):
        return f"[{', '.join(map(str, self.elements))}]"

class ExprIndex(SymbolicNode):
    """Represents a symbolic index into a collection."""
    def __init__(self, collection, index):
        self.collection = collection
        self.index = index

    def to_taylor(self, center, max_terms, hilbert):
        # This will be resolved by the interpreter's soft_index logic
        # or it will error if trying to expand a soft index without context.
        raise NotImplementedError("ExprIndex expansion requires soft_index resolution logic.")

    def _flatten_impl(self, memo, ops, constants, vars_dict):
        coll_idx = self.collection.flatten(memo, ops, constants, vars_dict)
        idx_idx = self.index.flatten(memo, ops, constants, vars_dict) if isinstance(self.index, SymbolicNode) else self._wrap(self.index).flatten(memo, ops, constants, vars_dict)
        res_idx = len(ops)
        ops.append(('index', coll_idx, idx_idx, res_idx))
        return res_idx

    def __repr__(self):
        return f"{self.collection}[{self.index}]"

class ExprIf(SymbolicNode):
    """
    Differentiable branching node.
    Blends true_branch and false_branch based on sigmoid(cond).
    Supports lazy evaluation and depth-limited recursion.
    """
    def __init__(self, cond, true_branch, false_branch, max_depth=16):
        self.cond = cond
        self.true_branch = true_branch # Can be SymbolicNode or callable
        self.false_branch = false_branch
        self.max_depth = max_depth

    def to_taylor(self, center, max_terms, hilbert):
        # Prevent infinite recursion
        if getattr(hilbert, '_recursion_depth', 0) > self.max_depth:
            # Return a constant at the limit
            return torch.zeros(max_terms, device=hilbert.factorials.device)
        
        # Increment depth
        hilbert._recursion_depth = getattr(hilbert, '_recursion_depth', 0) + 1
        
        try:
            # Evaluate condition logit using robust sigmoid
            # This will use ExprFunc's analytic re-centering logic
            sig_node = ExprFunc('sigmoid', self.cond)
            prob = sig_node.to_taylor(center, max_terms, hilbert)
            
            # Straight-Through Estimator (STE) for Exact Logic
            if getattr(hilbert, 'hard_logic', False):
                # Snap 0-th coefficient (probability) to 0 or 1
                p0 = prob[0]
                p0_hard = torch.round(p0)
                
                # STE trick: p_ste = p_hard + (p_soft - p_soft.detach())
                # This makes the forward pass p_hard, and the backward pass p_soft
                p0_ste = p0_hard + (p0 - p0.detach())
                
                # Apply snapping to the entire Taylor series
                # Higher order terms correspond to derivatives; we scale them by p0_ste
                # so that if branch is not chosen, derivatives are 0.
                prob = prob * (p0_ste / (p0 + 1e-18))
                prob[0] = p0_ste

            # Evaluate branches
            tv = self.true_branch() if callable(self.true_branch) else self.true_branch
            fv = self.false_branch() if callable(self.false_branch) else self.false_branch
            
            # Ensure branch results are symbolic
            if not isinstance(tv, SymbolicNode): tv = self._wrap(tv)
            if not isinstance(fv, SymbolicNode): fv = self._wrap(fv)
            
            t_t = tv.to_taylor(center, max_terms, hilbert)
            f_t = fv.to_taylor(center, max_terms, hilbert)
            
            # Blend
            res = prob * t_t + (1.0 - prob) * f_t
            return res
        finally:
            hilbert._recursion_depth -= 1

    def _flatten_impl(self, memo, ops, constants, vars_dict):
        cond_idx = self.cond.flatten(memo, ops, constants, vars_dict)
        # Evaluate branches to get nodes (they are usually callables)
        tv = self.true_branch() if callable(self.true_branch) else self.true_branch
        fv = self.false_branch() if callable(self.false_branch) else self.false_branch
        if not isinstance(tv, SymbolicNode): tv = self._wrap(tv)
        if not isinstance(fv, SymbolicNode): fv = self._wrap(fv)
        
        t_idx = tv.flatten(memo, ops, constants, vars_dict)
        f_idx = fv.flatten(memo, ops, constants, vars_dict)
        
        res_idx = len(ops)
        ops.append(('if', cond_idx, t_idx, f_idx, res_idx))
        return res_idx

    def __repr__(self):
        return f"soft_if({self.cond}, ...)"

class SymbolicProgram:
    """
    A compiled, flattened representation of a symbolic expression.
    Executes in a linear sequence to bypass Python's recursion limit and overhead.
    """
    def __init__(self, constants, variable_names, ops, final_idx):
        self.constants = constants
        self.variable_names = variable_names
        self.ops = ops
        self.final_idx = final_idx

    def to_taylor(self, center, max_terms, hilbert):
        """Vectorized execution of the flattened graph."""
        # 1. Pre-resolve constants and variables
        const_taylors = []
        for c in self.constants:
            v = c
            if not torch.is_tensor(v):
                v = torch.tensor(float(v), dtype=torch.float32)
            zeros = torch.zeros(max_terms - 1, dtype=v.dtype, device=v.device)
            const_taylors.append(torch.cat([v.unsqueeze(0) if v.dim()==0 else v, zeros]))

        var_taylors = {}
        for name in self.variable_names:
            c_val = 0.0
            if isinstance(center, dict):
                c_val = center.get(name, 0.0)
            else:
                c_val = center if name == 'x' else 0.0
            
            res = torch.zeros(max_terms, dtype=torch.float32)
            if torch.is_tensor(c_val): res = res.to(c_val.device)
            res[0] = c_val
            if max_terms > 1: res[1] = 1.0
            var_taylors[name] = res

        # 2. Register file for intermediate results
        regs = [None] * len(self.ops)

        def get_val(ref):
            if isinstance(ref, tuple):
                if ref[0] == 'const': return const_taylors[ref[1]]
                if ref[0] == 'var': return var_taylors[ref[1]]
            return regs[ref]

        # 3. Linear Execution Loop
        for op in self.ops:
            op_type = op[0]
            if op_type == 'add':
                regs[op[3]] = get_val(op[1]) + get_val(op[2])
            elif op_type == 'sub':
                regs[op[3]] = get_val(op[1]) - get_val(op[2])
            elif op_type == 'mul':
                regs[op[3]] = hilbert.multiply(get_val(op[1]), get_val(op[2]))
            elif op_type == 'div':
                regs[op[3]] = hilbert.divide(get_val(op[1]), get_val(op[2]))
            elif op_type == 'func':
                name, arg_ref, res_idx = op[1], op[2], op[3]
                arg_taylor = get_val(arg_ref)
                regs[res_idx] = ExprFunc._to_taylor_from_coeffs(name, arg_taylor, center, max_terms, hilbert)
            elif op_type == 'if':
                cond_idx, t_idx, f_idx, res_idx = op[1], op[2], op[3], op[4]
                prob = ExprFunc._to_taylor_from_coeffs('sigmoid', get_val(cond_idx), center, max_terms, hilbert)
                
                # Straight-Through Estimator (STE) for Exact Logic
                if getattr(hilbert, 'hard_logic', False):
                    p0 = prob[0]
                    p0_hard = torch.round(p0)
                    p0_ste = p0_hard + (p0 - p0.detach())
                    # Avoid div by zero
                    prob = prob * (p0_ste / (p0 + 1e-18))
                    prob[0] = p0_ste

                t_t = get_val(t_idx)
                f_t = get_val(f_idx)
                regs[res_idx] = prob * t_t + (1.0 - prob) * f_t
            elif op_type == 'list':
                # Just stores internal indices for index op
                regs[op[2]] = [get_val(idx) for idx in op[1]]
            elif op_type == 'index':
                # Simplified index resolution (requires Hilbert/Interpreter context usually)
                # But for Turbo, we can implement basic soft indexing if the collection is a 'list' register result.
                coll = get_val(op[1])
                target_idx = get_val(op[2])
                res_idx = op[3]
                
                # Soft indexing implementation
                # result = sum(p_i * element_i)
                # target_idx is a Taylor series; its 0-th coef is the index value.
                # We'll use a softmax-like approach if not using hard logic.
                
                idx_val = target_idx[0]
                n = len(coll)
                
                # Vectorized weights
                positions = torch.arange(n, device=idx_val.device, dtype=idx_val.dtype)
                logits = -((positions - idx_val)**2) * 10.0 # High precision sharpness
                weights = torch.softmax(logits, dim=0)
                
                if getattr(hilbert, 'hard_logic', False):
                    # Hard indexing STE
                    best_idx = torch.argmax(weights)
                    w_hard = torch.zeros_like(weights)
                    w_hard[best_idx] = 1.0
                    weights = w_hard + (weights - weights.detach())

                # Weighted sum of Taylor series elements
                acc = weights[0] * coll[0]
                for i in range(1, n):
                    acc = acc + weights[i] * coll[i]
                regs[res_idx] = acc

        return get_val(self.final_idx)
