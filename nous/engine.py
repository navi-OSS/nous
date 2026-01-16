import torch
import torch.nn as nn
import math

class NousHilbertCore(nn.Module):
    """
    Core symbolic computation engine using Taylor series representations.
    """
    def __init__(self, max_terms=32):
        super().__init__()
        self.max_terms = max_terms
        self.indices = nn.Parameter(torch.arange(max_terms, dtype=torch.float32), requires_grad=False)
        self.factorials = nn.Parameter(torch.tensor([float(math.factorial(i)) for i in range(max_terms)], dtype=torch.float32), requires_grad=False)

    def get_taylor_exp(self):
        """Taylor series for exp(x) = sum(x^n / n!)."""
        return 1.0 / self.factorials

    def get_taylor_sin(self):
        """Taylor series for sin(x) = x - x^3/3! + x^5/5! - ..."""
        coeffs = torch.zeros(self.max_terms, dtype=self.factorials.dtype, device=self.factorials.device)
        for n in range(self.max_terms // 2):
            coeffs[2*n + 1] = ((-1)**n) / self.factorials[2*n + 1]
        return coeffs

    def get_taylor_cos(self):
        """Taylor series for cos(x) = 1 - x^2/2! + x^4/4! - ..."""
        coeffs = torch.zeros(self.max_terms, dtype=self.factorials.dtype, device=self.factorials.device)
        for n in range(self.max_terms // 2):
            coeffs[2*n] = ((-1)**n) / self.factorials[2*n]
        return coeffs

    def get_taylor_log1p(self):
        """Taylor series for log(1+x) = x - x^2/2 + x^3/3 - ..."""
        coeffs = torch.zeros(self.max_terms, dtype=self.factorials.dtype, device=self.factorials.device)
        for n in range(1, self.max_terms):
            coeffs[n] = ((-1)**(n+1)) / n
        return coeffs

    def get_taylor_sinh(self):
        """Taylor series for sinh(x) = x + x^3/3! + x^5/5! + ..."""
        coeffs = torch.zeros(self.max_terms, dtype=self.factorials.dtype, device=self.factorials.device)
        for n in range(self.max_terms // 2):
            coeffs[2*n + 1] = 1.0 / self.factorials[2*n + 1]
        return coeffs

    def get_taylor_cosh(self):
        """Taylor series for cosh(x) = 1 + x^2/2! + x^4/4! + ..."""
        coeffs = torch.zeros(self.max_terms, dtype=self.factorials.dtype, device=self.factorials.device)
        for n in range(self.max_terms // 2):
            coeffs[2*n] = 1.0 / self.factorials[2*n]
        return coeffs

    def get_taylor_tanh(self):
        """Taylor series for tanh(x) = sinh(x) / cosh(x)."""
        sinh = self.get_taylor_sinh()
        cosh = self.get_taylor_cosh()
        return self.divide(sinh, cosh)

    def get_taylor_tan(self):
        """Taylor series for tan(x) = sin(x) / cos(x)."""
        sin = self.get_taylor_sin()
        cos = self.get_taylor_cos()
        return self.divide(sin, cos)

    def get_taylor_sqrt1p(self):
        """Taylor series for sqrt(1+x) = sum( (0.5 choose n) * x^n )."""
        coeffs = torch.zeros(self.max_terms, dtype=self.factorials.dtype, device=self.factorials.device)
        coeffs[0] = 1.0
        if self.max_terms > 1:
            curr = 1.0
            for n in range(1, self.max_terms):
                # binomial(0.5, n) = binomial(0.5, n-1) * (0.5 - n + 1) / n
                curr = curr * (0.5 - n + 1) / n
                coeffs[n] = curr
        return coeffs

    def derivative(self, coeffs):
        """Compute polynomial derivative by coefficient shifting and scaling."""
        n = coeffs.shape[-1]
        p_idx = torch.arange(n, dtype=coeffs.dtype, device=coeffs.device)
        res = coeffs * p_idx
        return torch.cat([res[..., 1:], torch.zeros_like(res[..., :1])], dim=-1)

    def nth_derivative(self, coeffs, n_derivs):
        """Compute n-th order derivative efficiently."""
        if n_derivs == 0: return coeffs
        dim = coeffs.shape[-1]
        if n_derivs >= dim: return torch.zeros_like(coeffs)
        
        # Compute scaling factor: k * (k-1) * ... * (k-n+1)
        k = torch.arange(dim, dtype=coeffs.dtype, device=coeffs.device)
        factor = torch.ones_like(k)
        for i in range(n_derivs):
            factor = factor * (k - i)
        
        # Apply scaling and shift coefficients left by n_derivs positions
        weighted = coeffs * factor
        res = torch.zeros_like(coeffs)
        res[..., :-n_derivs] = weighted[..., n_derivs:]
        return res

    def integrate(self, coeffs):
        """Compute indefinite integral with constant of integration C=0."""
        dim = coeffs.shape[-1]
        k = torch.arange(dim, dtype=coeffs.dtype, device=coeffs.device)
        divisor = k + 1
        weighted = coeffs / divisor
        
        # Shift coefficients right by 1 position
        res = torch.zeros_like(coeffs)
        res[..., 1:] = weighted[..., :-1]
        return res

    def definite_integrate(self, coeffs, a, b):
        """
        Compute definite integral from a to b.
        
        Args:
            coeffs: Polynomial coefficients (ascending powers)
            a: Lower bound
            b: Upper bound
        Returns:
            Scalar or tensor of integral values
        """
        antideriv = self.integrate(coeffs)
        return self.eval_at(antideriv, b) - self.eval_at(antideriv, a)

    def simplify(self, coeffs, tolerance=1e-12):
        """
        Zero out near-zero coefficients and return effective degree.
        
        Args:
            coeffs: Polynomial coefficients
            tolerance: Threshold below which coefficients are zeroed
        Returns:
            Tuple of (simplified_coeffs, effective_degree)
        """
        mask = torch.abs(coeffs) > tolerance
        simplified = coeffs * mask.float()
        
        # Find effective degree (highest non-zero term)
        nonzero_indices = torch.where(mask)[0] if mask.dim() == 1 else torch.where(mask.any(dim=0))[0]
        degree = nonzero_indices[-1].item() if len(nonzero_indices) > 0 else 0
        
        return simplified, degree

    def compose(self, outer, inner):
        """
        Compute function composition f(g(x)) in Taylor representation.
        Optimized version using cached powers and batched multiplication.
        """
        max_terms = inner.shape[-1]
        
        # Pre-compute all powers of inner: [1, g, g², g³, ...]
        # This enables batched operations instead of sequential
        g_powers = torch.zeros(max_terms, max_terms, dtype=inner.dtype, device=inner.device)
        g_powers[0, 0] = 1.0  # g^0 = 1
        
        if max_terms > 1:
            g_powers[1] = inner  # g^1 = g
            for k in range(2, max_terms):
                g_powers[k] = self._poly_mul(g_powers[k-1], inner)
        
        # Vectorized composition: sum(outer[k] * g^k)
        # outer: [max_terms], g_powers: [max_terms, max_terms]
        # Result: sum over k of outer[k] * g_powers[k]
        res = torch.einsum('k,kn->n', outer, g_powers)
        return res
        
    def multiply(self, a, b):
        """Public alias for polynomial multiplication."""
        return self._poly_mul(a, b)

    def _poly_mul(self, a, b):
        """
        Truncated polynomial multiplication via FFT convolution.
        O(n log n) and fully parallelizable on GPU.
        """
        n = a.shape[-1]
        
        # FFT-based convolution: 
        # 1. Pad to 2n to avoid circular wrap-around
        # 2. FFT both, multiply, IFFT
        # 3. Truncate to n terms
        
        # Handle batched inputs
        original_shape = a.shape[:-1]
        a_flat = a.reshape(-1, n)
        b_flat = b.reshape(-1, n)
        
        # FFT convolution
        a_fft = torch.fft.fft(a_flat, n=2*n, dim=-1)
        b_fft = torch.fft.fft(b_flat, n=2*n, dim=-1)
        c_fft = a_fft * b_fft
        c_full = torch.fft.ifft(c_fft, dim=-1).real
        
        # Truncate to original size
        c = c_full[..., :n]
        
        # Reshape back
        return c.reshape(*original_shape, n)

    def _poly_mul_naive(self, a, b):
        """
        Naive O(n²) polynomial multiplication for reference/fallback.
        Kept for numerical precision comparison.
        """
        n = a.shape[-1]
        c = torch.zeros_like(a)
        for k in range(n):
            i_indices = torch.arange(k + 1, device=a.device)
            c[..., k] = (a[..., i_indices] * b[..., k - i_indices]).sum(dim=-1)
        return c


    def eval_at(self, coeffs, x):
        """Horner's method evaluation."""
        res = torch.zeros_like(x)
        for i in range(coeffs.shape[-1] - 1, -1, -1):
            res = res * x + coeffs[..., i]
        return res

    def divide(self, a, b):
        """
        Calculates polynomial division A / B using recursive deconvolution.
        Assumes b[..., 0] is non-zero (division by series with constant term).
        """
        n = a.shape[-1]
        q = torch.zeros_like(a)
        
        # We need b0 to be non-zero (invertible)
        # Check happens implicitly or we can warn?
        # b[..., 0] is the constant term
        b0 = b[..., 0]
        
        for k in range(n):
            # q[k] = (a[k] - sum_{j=0}^{k-1} q[j]*b[k-j]) / b0
            # sum term:
            sum_term = torch.zeros_like(b0)
            if k > 0:
                # convolve q[0..k-1] with b[1..k]
                # q[j] corresponds to q[..., j]
                # b[k-j] corresponds to b[..., k-j]
                j_indices = torch.arange(k, device=a.device)
                sum_term = (q[..., j_indices] * b[..., k - j_indices]).sum(dim=-1)
            
            q[..., k] = (a[..., k] - sum_term) / b0
            
        return q

class DifferentiableRootSolver(torch.autograd.Function):
    """
    Differentiable polynomial root finding using the Durand-Kerner method.
    
    Forward: Iterative root finding with adaptive stopping and native complex tensor support.
    Backward: Implicit differentiation via the Implicit Function Theorem.
    """
    @staticmethod
    def forward(ctx, coeffs, iterations=100, tolerance=1e-9):
        is_real_input = not coeffs.is_complex()
        ctx.is_real_input = is_real_input
        
        if is_real_input:
            coeffs_complex = coeffs.to(dtype=torch.complex128) if coeffs.dtype == torch.float64 else coeffs.to(dtype=torch.complex64)
        else:
            coeffs_complex = coeffs
            
        batch_size, n_plus_1 = coeffs_complex.shape
        n = n_plus_1 - 1
        device = coeffs_complex.device
        
        # Normalize to monic polynomial
        leading = coeffs_complex[:, 0:1]
        norm_coeffs = coeffs_complex / leading
        
        # Initialize roots on complex circle with perturbation to break symmetry
        real_dtype = coeffs_complex.real.dtype
        angles = torch.arange(n, device=device, dtype=real_dtype) * (2 * math.pi / n)
        angles = angles + torch.randn_like(angles) * 0.01
        
        radius = 0.5 + 0.6 * torch.abs(norm_coeffs[:, 1:2])**(1/n) if n > 0 else torch.ones(1, device=device)
        
        z = radius * torch.exp(1j * angles)
        z = z.expand(batch_size, -1).clone()

        # Durand-Kerner iteration
        for k in range(iterations):
            # Evaluate polynomial at current root estimates using Horner's method
            p_val = torch.zeros_like(z)
            for i in range(n_plus_1):
                p_val = p_val * z + norm_coeffs[:, i:i+1]
            
            # Compute denominator: product of differences from other roots
            z_expanded = z.unsqueeze(2)
            z_trans = z.unsqueeze(1)
            diffs = z_expanded - z_trans
            
            eye = torch.eye(n, device=device, dtype=torch.bool)
            diffs = diffs.masked_fill(eye.unsqueeze(0), 1.0)
            
            denom = torch.prod(diffs, dim=2)
            
            # Update root estimates
            correction = p_val / (denom + 1e-30)
            z = z - correction
            
            # Early stopping if converged
            max_correction = torch.max(torch.abs(correction))
            if max_correction < tolerance:
                break
        # Multiplicity-aware refinement using Halley's method
        # For multiple roots, standard Newton converges slowly because P'(z)→0
        # Halley's method: z = z - 2*P*P' / (2*P'^2 - P*P'') converges cubically
        for _ in range(30):
            # Compute P(z), P'(z), P''(z) simultaneously using Horner's
            p_val = torch.zeros_like(z)
            dp_val = torch.zeros_like(z)
            d2p_val = torch.zeros_like(z)
            
            for i in range(n_plus_1):
                d2p_val = d2p_val * z + 2 * dp_val
                dp_val = dp_val * z + p_val
                p_val = p_val * z + norm_coeffs[:, i:i+1]
            
            # Estimate multiplicity: m ≈ P*P'' / (P'^2 - P*P'')
            # For simple root: m ≈ 1. For double: m ≈ 2
            numerator = p_val * d2p_val
            denominator = dp_val * dp_val - p_val * d2p_val
            
            # Avoid division by zero
            safe_denom = torch.where(torch.abs(denominator) < 1e-50, 
                                     torch.ones_like(denominator), denominator)
            m_estimate = torch.where(torch.abs(denominator) < 1e-50,
                                     torch.ones_like(numerator),
                                     numerator / safe_denom + 1)
            
            # Clamp multiplicity estimate to reasonable range [1, n]
            m_estimate = torch.clamp(m_estimate.real, 1.0, float(n))
            
            # Modified Newton with multiplicity: z = z - m * P(z) / P'(z)
            safe_dp = torch.where(torch.abs(dp_val) < 1e-100,
                                  torch.ones_like(dp_val) * 1e-100, dp_val)
            correction = m_estimate * p_val / safe_dp
            z = z - correction
            
            max_correction = torch.max(torch.abs(correction))
            if max_correction < tolerance * 1e-6:
                break
        
        # Cluster nearly-identical roots and use Vieta's formulas
        # For m equal roots: r = -c1/(m*c0) where c1 is the second coefficient
        cluster_threshold = 1e-6
        for b in range(batch_size):
            # Build adjacency for clustering
            clustered = [False] * n
            for i in range(n):
                if clustered[i]:
                    continue
                cluster_indices = [i]
                for j in range(i + 1, n):
                    if torch.abs(z[b, i] - z[b, j]) < cluster_threshold:
                        cluster_indices.append(j)
                        clustered[j] = True
                
                if len(cluster_indices) > 1:
                    # Use Vieta: sum of all roots = -c1/c0, so m equal roots = -c1/(m*c0)
                    # But we only have m clustered, not all n roots
                    # Better: average the clustered roots, then refine
                    m = len(cluster_indices)
                    # Vieta for full multiplicity: r = -c1/(n*c0) if ALL roots are equal
                    # For partial: use average of cluster
                    avg = sum(z[b, idx] for idx in cluster_indices) / m
                    
                    # If cluster size equals n, use exact Vieta
                    if m == n:
                        exact_root = -norm_coeffs[b, 1] / n
                        for idx in cluster_indices:
                            z[b, idx] = exact_root
                    else:
                        for idx in cluster_indices:
                            z[b, idx] = avg
        
        # Final Newton polish on clustered roots
        for _ in range(20):
            p_val = torch.zeros_like(z)
            dp_val = torch.zeros_like(z)
            for i in range(n_plus_1):
                dp_val = dp_val * z + p_val
                p_val = p_val * z + norm_coeffs[:, i:i+1]
            
            # For clustered roots, use higher multiplicity
            z_expanded = z.unsqueeze(2)
            z_trans = z.unsqueeze(1)
            diffs = torch.abs(z_expanded - z_trans)
            eye = torch.eye(n, device=device, dtype=torch.bool)
            diffs = diffs.masked_fill(eye.unsqueeze(0), float('inf'))
            min_diff = diffs.min(dim=2)[0]
            multiplicity = torch.where(min_diff < cluster_threshold,
                                       torch.tensor(2.0, device=device),
                                       torch.tensor(1.0, device=device))
            
            safe_dp = torch.where(torch.abs(dp_val) < 1e-100,
                                  torch.ones_like(dp_val) * 1e-100, dp_val)
            correction = multiplicity * p_val / safe_dp
            z = z - correction
            
            if torch.max(torch.abs(correction)) < tolerance * 1e-9:
                break
        
        ctx.save_for_backward(coeffs_complex, z)
        return z
    
    @staticmethod
    def backward(ctx, grad_roots):
        """
        Implicit differentiation: dz/dc = -(dP/dz)^-1 * dP/dc
        """
        coeffs, z = ctx.saved_tensors
        n = z.shape[-1]
        dp_dz = torch.zeros_like(z)
        
        # Compute dP/dz using Horner's method
        for i in range(n):
            power = n - i
            term = coeffs[:, i:i+1] * power
            if i == 0:
                dp_dz = term
            else:
                dp_dz = dp_dz * z + term
        
        # Compute weighted gradient
        weighted_grad = -grad_roots / (dp_dz + 1e-30)
        
        # Accumulate gradients for each coefficient
        grad_coeffs = torch.zeros_like(coeffs)
        z_pow = torch.ones_like(z)
        
        for k in range(n, -1, -1):
            grad_k = (weighted_grad * z_pow).sum(dim=1)
            grad_coeffs[:, k] = grad_k
            
            if k > 0:
                z_pow = z_pow * z
        
        if not ctx.needs_input_grad[0]:
            return None, None, None

        if ctx.is_real_input:
            return grad_coeffs.real, None, None
            
        return grad_coeffs, None, None


class MultivariatePolynomial(nn.Module):
    """
    Multivariate polynomial operations using tensor representation.
    Coefficients are stored as [batch, d1, d2, ..., dn] where di is max degree for variable i.
    """
    def __init__(self, num_vars, max_degree=8):
        super().__init__()
        self.num_vars = num_vars
        self.max_degree = max_degree
    
    def evaluate(self, coeffs, points):
        """
        Evaluate multivariate polynomial at given points.
        
        Args:
            coeffs: Coefficient tensor of shape [d1, d2, ..., dn] or [batch, d1, d2, ...]
            points: Points tensor of shape [num_points, num_vars] or [batch, num_points, num_vars]
        Returns:
            Values at each point
        """
        has_batch = coeffs.dim() > self.num_vars
        if not has_batch:
            coeffs = coeffs.unsqueeze(0)
            points = points.unsqueeze(0)
        
        batch_size = coeffs.shape[0]
        num_points = points.shape[1]
        result = torch.zeros(batch_size, num_points, dtype=coeffs.dtype, device=coeffs.device)
        
        # Generate all multi-indices for the polynomial
        degrees = [range(coeffs.shape[i+1]) for i in range(self.num_vars)]
        
        for idx in torch.cartesian_prod(*[torch.arange(d) for d in [coeffs.shape[i+1] for i in range(self.num_vars)]]):
            coeff_idx = tuple([slice(None)] + idx.tolist())
            coeff = coeffs[coeff_idx]
            
            term = torch.ones(batch_size, num_points, dtype=coeffs.dtype, device=coeffs.device)
            for var_idx, power in enumerate(idx):
                term = term * (points[:, :, var_idx] ** power.item())
            
            result = result + coeff.unsqueeze(1) * term
        
        return result.squeeze(0) if not has_batch else result
    
    def partial_derivative(self, coeffs, var_index):
        """
        Compute partial derivative with respect to specified variable.
        
        Args:
            coeffs: Coefficient tensor
            var_index: Index of variable to differentiate
        Returns:
            Coefficient tensor of the partial derivative
        """
        result = torch.zeros_like(coeffs)
        
        # Create slicing objects for source and destination
        src_slices = [slice(None)] * coeffs.dim()
        dst_slices = [slice(None)] * coeffs.dim()
        
        var_dim = var_index + (1 if coeffs.dim() > self.num_vars else 0)
        max_deg = coeffs.shape[var_dim]
        
        for k in range(1, max_deg):
            src_slices[var_dim] = k
            dst_slices[var_dim] = k - 1
            result[tuple(dst_slices)] = coeffs[tuple(src_slices)] * k
        
        return result
    
    def gradient(self, coeffs):
        """
        Compute gradient (all partial derivatives).
        
        Returns:
            List of coefficient tensors for each partial derivative
        """
        return [self.partial_derivative(coeffs, i) for i in range(self.num_vars)]


class TaylorODESolver(nn.Module):
    """
    ODE solver using Taylor series method for dy/dt = f(y).
    Differentiable and suitable for neural network integration.
    """
    def __init__(self, max_terms=16):
        super().__init__()
        self.max_terms = max_terms
        self.hilbert = NousHilbertCore(max_terms=max_terms)
    
    def solve(self, f_coeffs, y0, t_span, num_steps=100):
        """
        Solve first-order ODE dy/dt = f(y) using Taylor series method.
        
        Args:
            f_coeffs: Taylor coefficients of f(y) representing the RHS
            y0: Initial condition y(t0)
            t_span: Tuple (t0, t1) for integration range
            num_steps: Number of time steps
        Returns:
            Tuple (t_values, y_values) with solution trajectory
        """
        t0, t1 = t_span
        dt = (t1 - t0) / num_steps
        
        t_values = torch.linspace(t0, t1, num_steps + 1, dtype=f_coeffs.dtype, device=f_coeffs.device)
        y_values = torch.zeros(num_steps + 1, dtype=f_coeffs.dtype, device=f_coeffs.device)
        y_values[0] = y0
        
        y = y0
        for i in range(num_steps):
            # Compute Taylor expansion of solution around current point
            # y(t+dt) ≈ y(t) + y'(t)*dt + y''(t)*dt²/2 + ...
            # where y' = f(y), y'' = f'(y)*f(y), etc.
            
            # Evaluate f and its derivatives at current y
            f_val = self.hilbert.eval_at(f_coeffs, y)
            
            # Simple Euler step for stability, enhanced with midpoint correction
            k1 = f_val
            y_mid = y + 0.5 * dt * k1
            k2 = self.hilbert.eval_at(f_coeffs, y_mid)
            
            # Midpoint method (second-order)
            y = y + dt * k2
            y_values[i + 1] = y
        
        return t_values, y_values
    
    def solve_system(self, f_coeffs_list, y0_list, t_span, num_steps=100):
        """
        Solve system of ODEs: dy_i/dt = f_i(y_1, ..., y_n)
        
        Args:
            f_coeffs_list: List of coefficient tensors for each equation
            y0_list: List of initial conditions
            t_span: Tuple (t0, t1)
            num_steps: Number of time steps
        Returns:
            Tuple (t_values, y_values_matrix)
        """
        t0, t1 = t_span
        dt = (t1 - t0) / num_steps
        n_vars = len(y0_list)
        
        dtype = f_coeffs_list[0].dtype
        device = f_coeffs_list[0].device
        
        t_values = torch.linspace(t0, t1, num_steps + 1, dtype=dtype, device=device)
        y_values = torch.zeros(num_steps + 1, n_vars, dtype=dtype, device=device)
        
        y = torch.tensor(y0_list, dtype=dtype, device=device)
        y_values[0] = y
        
        for i in range(num_steps):
            # RK4 for systems
            k1 = torch.stack([self.hilbert.eval_at(f, y[j]) for j, f in enumerate(f_coeffs_list)])
            
            y_mid1 = y + 0.5 * dt * k1
            k2 = torch.stack([self.hilbert.eval_at(f, y_mid1[j]) for j, f in enumerate(f_coeffs_list)])
            
            y_mid2 = y + 0.5 * dt * k2
            k3 = torch.stack([self.hilbert.eval_at(f, y_mid2[j]) for j, f in enumerate(f_coeffs_list)])
            
            y_end = y + dt * k3
            k4 = torch.stack([self.hilbert.eval_at(f, y_end[j]) for j, f in enumerate(f_coeffs_list)])
            
            y = y + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
            y_values[i + 1] = y
        
        return t_values, y_values

class NousAlgebra(nn.Module):
    """
    Differentiable polynomial root solver using the Durand-Kerner method.
    """
    def __init__(self, iterations=60):
        super().__init__()
        self.iterations = iterations

    def solve(self, coeffs):
        """
        Find polynomial roots with gradient support.
        
        Args:
            coeffs: [batch, n+1] tensor of coefficients (highest degree first)
        Returns:
            roots: [batch, n, 2] tensor of real and imaginary components
        """
        roots = DifferentiableRootSolver.apply(coeffs, self.iterations)
        return torch.view_as_real(roots)

class SymbolicChain(nn.Module):
    """
    Define and execute a sequence of symbolic operations.
    Enables multi-step reasoning within a single differentiable graph.
    """
    def __init__(self, model, ops=None):
        super().__init__()
        self.model = model
        self.ops = ops or []
    
    def forward(self, x, **kwargs):
        """
        Execute chain: x -> op1 -> op2 -> ... -> output
        """
        curr = x
        for op in self.ops:
            if isinstance(op, tuple):
                name, op_kwargs = op
                # Dynamic kwargs update
                merged_kwargs = {**kwargs, **op_kwargs}
                curr = self.model.forward(curr, op=name, **merged_kwargs)
            else:
                curr = self.model.forward(curr, op=op, **kwargs)
        return curr


from .geometry import SymbolicGeometry
from .symbolic import SymbolicNode

class NousModel(nn.Module):
    def __init__(self, max_terms=32, solver_iterations=60, solver_tolerance=1e-9):
        super().__init__()
        self.max_terms = max_terms
        self.solver_iterations = solver_iterations
        self.solver_tolerance = solver_tolerance
        
        self.hilbert = NousHilbertCore(max_terms=max_terms)
        self.algebra = NousAlgebra(iterations=solver_iterations)
        self.geometry = SymbolicGeometry()
        # Learnable projection matrix for coefficient discovery
        self.discovery = nn.Parameter(torch.randn(max_terms, max_terms, dtype=torch.float32) * 0.01)

    def expand(self, node, center=0.0):
        """Expand a symbolic node to Taylor coefficients at a given center."""
        if not isinstance(node, SymbolicNode):
            raise ValueError("Input must be a SymbolicNode")
        return node.to_taylor(center, self.max_terms, self.hilbert)

    def forward(self, x, op, **kwargs):
        # Handle symbolic inputs if x is a node
        if isinstance(x, SymbolicNode):
            if op == 'diff':
                return x.diff(kwargs.get('var', 'x'))
            if op == 'expand':
                return self.expand(x, kwargs.get('center', 0.0))
            # Auto-expand at 0.0 for other ops if needed
            x = self.expand(x, center=0.0).unsqueeze(0)

        if op == 'geometry':
            # Geometry dispatcher
            sub_op = kwargs['sub_op']
            if sub_op == 'distance':
                return self.geometry.distance_point_to_circle(x, kwargs['params'])
            if sub_op == 'intersection':
                return self.geometry.intersection_circles(kwargs['c1'], kwargs['c2'])
            if sub_op == 'area_circle':
                return self.geometry.area_circle(kwargs['radius'])
            raise ValueError(f"Unknown geometry sub-op: {sub_op}")
        
        if op == 'get_identity':
            name = kwargs['name']
            if name == 'exp': return self.hilbert.get_taylor_exp()
            if name == 'sin': return self.hilbert.get_taylor_sin()
            if name == 'cos': return self.hilbert.get_taylor_cos()
            if name == 'log1p': return self.hilbert.get_taylor_log1p()
            if name == 'sinh': return self.hilbert.get_taylor_sinh()
            if name == 'cosh': return self.hilbert.get_taylor_cosh()
            if name == 'tanh': return self.hilbert.get_taylor_tanh()
            if name == 'tan': return self.hilbert.get_taylor_tan()
            if name == 'sqrt1p': return self.hilbert.get_taylor_sqrt1p()
            raise ValueError(f"Unknown identity: {name}")
        
        if op == 'solve': 
            return self.algebra.solve(x)
        
        if op == 'derivative': 
            return self.hilbert.derivative(x)
        
        if op == 'nth_derivative':
            return self.hilbert.nth_derivative(x, kwargs.get('n', 1))
            
        if op == 'integrate':
            return self.hilbert.integrate(x)
            
        if op == 'evaluate': 
            return self.hilbert.eval_at(x, kwargs['at'])
            
        if op == 'compose':
            return self.hilbert.compose(x, kwargs['inner'])
        
        if op == 'definite_integrate':
            return self.hilbert.definite_integrate(x, kwargs['a'], kwargs['b'])
        
        if op == 'simplify':
            return self.hilbert.simplify(x, kwargs.get('tolerance', 1e-12))
            
        if op == 'discover':
            # Project input features to Taylor coefficients
            if x is None:
                return self.discovery
            return x @ self.discovery

        if op == 'identity':
            return x

        if op == 'chain':
            # Execute a list of operations
            chain = SymbolicChain(self, ops=kwargs['ops'])
            return chain(x)
            
        if op == 'find_critical_points':
            # 1. Differentiate: [a0, a1, a2] -> [a1, 2*a2, 0]
            deriv = self.hilbert.derivative(x)
            
            # 2. Find effective degree to strip leading zeros
            # Batch support: find max degree across batch for consistency
            mask = torch.abs(deriv) > 1e-9
            nonzero_indices = torch.where(mask)
            if len(nonzero_indices[1]) == 0:
                return torch.zeros(x.shape[0], 1, 2, device=x.device) # No critical points
            
            max_deg = torch.max(nonzero_indices[1]).item()
            # Strip trailing zeros (which become leading zeros when flipped)
            stripped_deriv = deriv[:, :max_deg+1]
            
            # 3. Re-order for solve (descending)
            desc_deriv = torch.flip(stripped_deriv, dims=[-1])
            return self.algebra.solve(desc_deriv)

            
        raise ValueError(f"Unknown operation: {op}")

    def compile(self, mode="reduce-overhead"):
        """
        Compile the Hilbert Engine for faster execution using torch.compile.
        
        Args:
            mode: Compilation mode, one of "default", "reduce-overhead", "max-autotune"
        Returns:
            self (for chaining)
        """
        try:
            self.hilbert = torch.compile(self.hilbert, mode=mode)
        except Exception as e:
            # torch.compile may fail on some platforms/versions
            print(f"Warning: torch.compile failed ({e}). Falling back to eager mode.")
        return self

    def trace(self, node, center=0.0):
        """
        Trace a symbolic graph and return a frozen callable.
        
        This creates a closure that captures the expanded coefficients,
        avoiding repeated symbolic expansion during tight loops.
        
        Args:
            node: SymbolicNode to trace
            center: Expansion center
        Returns:
            Callable that takes a point 'x' and returns f(x)
        """
        coeffs = self.expand(node, center)
        
        def evaluate(x):
            return self.hilbert.eval_at(coeffs, x)
        
        return evaluate

if __name__ == "__main__":
    print("Nous engine initialized.")

