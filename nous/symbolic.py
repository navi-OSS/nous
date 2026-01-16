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

    def _wrap(self, other):
        if isinstance(other, (int, float)): return ExprConst(float(other))
        if torch.is_tensor(other) and other.numel() == 1: return ExprConst(float(other.item()))
        return other

    def to_taylor(self, center, max_terms, hilbert):
        # 'center' can be a float/tensor (scalar context) or a dict {name: val}
        raise NotImplementedError()

    def diff(self, var='x'):
        raise NotImplementedError()

class ExprConst(SymbolicNode):
    def __init__(self, value):
        self.value = value
    def to_taylor(self, center, max_terms, hilbert):
        res = torch.zeros(max_terms, dtype=torch.float32)
        res[0] = self.value
        return res
    def diff(self, var='x'):
        return ExprConst(0.0)
    def __int__(self): return int(self.value)
    def __index__(self): return int(self.value)
    def __bool__(self): return bool(self.value)
    def __float__(self): return float(self.value)
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
    def diff(self, var='x'):
        return ExprConst(1.0) if self.name == var else ExprConst(0.0)
    def __repr__(self): return self.name

class ExprAdd(SymbolicNode):
    def __init__(self, left, right):
        self.left, self.right = left, right
    def to_taylor(self, center, max_terms, hilbert):
        return self.left.to_taylor(center, max_terms, hilbert) + self.right.to_taylor(center, max_terms, hilbert)
    def diff(self, var='x'):
        return ExprAdd(self.left.diff(var), self.right.diff(var))
    def __repr__(self): return f"({self.left} + {self.right})"

class ExprSub(SymbolicNode):
    def __init__(self, left, right):
        self.left, self.right = left, right
    def to_taylor(self, center, max_terms, hilbert):
        return self.left.to_taylor(center, max_terms, hilbert) - self.right.to_taylor(center, max_terms, hilbert)
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
    def diff(self, var='x'):
        # (u/v)' = (u'v - uv') / v^2
        return ExprDiv(ExprSub(ExprMul(self.left.diff(var), self.right), ExprMul(self.left, self.right.diff(var))), ExprPow(self.right, ExprConst(2)))
    def __repr__(self): return f"({self.left} / {self.right})"

class ExprFunc(SymbolicNode):
    def __init__(self, name, arg):
        self.name, self.arg = name, arg
    def to_taylor(self, center, max_terms, hilbert):
        # f(g(x)) expansion centered at c:
        # 1. Expand g(x) around c -> g_coeffs (monomials in (x-c))
        # 2. Let g(c) = L. Expand f(y) around L.
        # 3. If L=0, use standard get_identity(f).
        # 4. If L!=0, use Taylor shift if identity is known, or compose translation.
        
        g_coeffs = self.arg.to_taylor(center, max_terms, hilbert)
        L = g_coeffs[0].item()
        
        # For simplicity in this implementation, we handle common identities at L=0
        # If L != 0, we can use the identity f(x+L) if available (e.g. exp(x+L) = exp(L)exp(x))
        if self.name == 'exp':
            base = hilbert.get_taylor_exp()
            if abs(L) > 1e-9:
                base = base * math.exp(L)
            g_shifted = g_coeffs.clone()
            g_shifted[0] -= L
            return hilbert.compose(base, g_shifted)
        
        if self.name == 'sin':
            sin_base = hilbert.get_taylor_sin()
            cos_base = hilbert.get_taylor_cos()
            combined = sin_base * math.cos(L) + cos_base * math.sin(L)
            g_shifted = g_coeffs.clone()
            g_shifted[0] -= L
            return hilbert.compose(combined, g_shifted)

        if self.name == 'cos':
            sin_base = hilbert.get_taylor_sin()
            cos_base = hilbert.get_taylor_cos()
            combined = cos_base * math.cos(L) - sin_base * math.sin(L)
            g_shifted = g_coeffs.clone()
            g_shifted[0] -= L
            return hilbert.compose(combined, g_shifted)

        if self.name == 'sinh':
            sinh_base = hilbert.get_taylor_sinh()
            cosh_base = hilbert.get_taylor_cosh()
            combined = sinh_base * math.cosh(L) + cosh_base * math.sinh(L)
            g_shifted = g_coeffs.clone()
            g_shifted[0] -= L
            return hilbert.compose(combined, g_shifted)

        if self.name == 'cosh':
            sinh_base = hilbert.get_taylor_sinh()
            cosh_base = hilbert.get_taylor_cosh()
            combined = cosh_base * math.cosh(L) + sinh_base * math.sinh(L)
            g_shifted = g_coeffs.clone()
            g_shifted[0] -= L
            return hilbert.compose(combined, g_shifted)

        if self.name == 'log':
            # log(x) expansion around L: log(L) + log(1 + (x-L)/L)
            if L <= 0:
                raise ValueError(f"log(x) requires x > 0, got x(0)={L}")
            base = hilbert.get_taylor_log1p()
            # log(1+u) where u = (g(x)-L)/L
            g_scaled = g_coeffs.clone()
            g_scaled[0] -= L
            g_scaled = g_scaled / L
            res = hilbert.compose(base, g_scaled)
            res[0] += math.log(L)
            return res

        if self.name == 'sqrt':
            # sqrt(x) expansion around L: sqrt(L) * sqrt(1 + (x-L)/L)
            if L < 0:
                raise ValueError(f"sqrt(x) requires x >= 0, got x(0)={L}")
            if abs(L) < 1e-12:
                # If L=0, sqrt is not analytic. We might need a small eps or just fail.
                # However, for neural nets, a small eps is often used.
                L = 1e-12
            
            base = hilbert.get_taylor_sqrt1p()
            g_scaled = g_coeffs.clone()
            g_scaled[0] -= L
            g_scaled = g_scaled / L
            res = hilbert.compose(base, g_scaled)
            return res * math.sqrt(L)

        # For tan and tanh, we can use the identity at 0 if we shift the input.
        # But we need the identity for tan(L + delta).
        # Better: use the quotient sin(L+d)/cos(L+d) or sinh(L+d)/cosh(L+d)
        if self.name == 'tan':
            # tan(u) = sin(u)/cos(u)
            sin_u = ExprFunc('sin', self.arg).to_taylor(center, max_terms, hilbert)
            cos_u = ExprFunc('cos', self.arg).to_taylor(center, max_terms, hilbert)
            return hilbert.divide(sin_u, cos_u)

        if self.name == 'tanh':
            # tanh(u) = sinh(u)/cosh(u)
            sinh_u = ExprFunc('sinh', self.arg).to_taylor(center, max_terms, hilbert)
            cosh_u = ExprFunc('cosh', self.arg).to_taylor(center, max_terms, hilbert)
            return hilbert.divide(sinh_u, cosh_u)

        # Fallback to standard identity at L=0 for others
        base = hilbert.get_identity(name=self.name)
        return hilbert.compose(base, g_coeffs)

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
