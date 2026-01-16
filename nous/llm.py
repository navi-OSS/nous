"""
Nous LLM Integration Layer

Provides nn.Module components for embedding Nous directly inside
LLM architectures as differentiable layers.

Components:
- EmbeddingBridge: Converts hidden states <-> Taylor coefficients
- NousLayer: Drop-in layer for transformer architectures
- SymbolicHead: Output head for structured symbolic predictions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .engine import NousHilbertCore


class EmbeddingBridge(nn.Module):
    """
    Bidirectional bridge between LLM hidden states and Taylor coefficients.
    
    Learns to project high-dimensional embeddings to polynomial space
    and back, enabling symbolic operations on neural representations.
    """
    def __init__(self, hidden_dim: int, max_terms: int = 32):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_terms = max_terms
        
        # Encoder: hidden_dim -> max_terms
        self.encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, max_terms)
        )
        
        # Decoder: max_terms -> hidden_dim
        self.decoder = nn.Sequential(
            nn.Linear(max_terms, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        
        # Layer norms for stability
        self.encode_norm = nn.LayerNorm(max_terms)
        self.decode_norm = nn.LayerNorm(hidden_dim)
    
    def encode(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Convert hidden states to Taylor coefficients.
        
        Args:
            hidden: [batch, seq_len, hidden_dim] or [batch, hidden_dim]
        Returns:
            coeffs: [batch, seq_len, max_terms] or [batch, max_terms]
        """
        coeffs = self.encoder(hidden)
        return self.encode_norm(coeffs)
    
    def decode(self, coeffs: torch.Tensor) -> torch.Tensor:
        """
        Convert Taylor coefficients back to hidden states.
        
        Args:
            coeffs: [batch, seq_len, max_terms] or [batch, max_terms]
        Returns:
            hidden: [batch, seq_len, hidden_dim] or [batch, hidden_dim]
        """
        hidden = self.decoder(coeffs)
        return self.decode_norm(hidden)


class NousLayer(nn.Module):
    """
    Drop-in symbolic reasoning layer for LLM architectures.
    
    Can be inserted into a transformer's forward pass to inject
    differentiable symbolic operations.
    
    Supports soft operation selection (learns which op to apply)
    or hard operation specification.
    """
    
    # Available symbolic operations
    OPS = ['identity', 'derivative', 'integrate', 'square', 'negate']
    
    def __init__(
        self,
        hidden_dim: int,
        max_terms: int = 32,
        num_ops: int = 5,
        residual: bool = True
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_terms = max_terms
        self.num_ops = num_ops
        self.residual = residual
        
        # Core components
        self.bridge = EmbeddingBridge(hidden_dim, max_terms)
        self.hilbert = NousHilbertCore(max_terms)
        
        # Operation selector: learns which op to apply
        self.op_selector = nn.Linear(hidden_dim, num_ops)
        
        # Learnable scaling for residual connection
        self.gate = nn.Parameter(torch.zeros(1))
    
    def forward(
        self,
        hidden: torch.Tensor,
        op: str = None,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Apply symbolic reasoning to hidden states.
        
        Args:
            hidden: [batch, seq_len, hidden_dim] input hidden states
            op: Optional specific operation ('derivative', 'integrate', etc.)
                If None, uses soft selection based on learned op_selector
            temperature: Softmax temperature for op selection (lower = sharper)
        
        Returns:
            output: [batch, seq_len, hidden_dim] transformed hidden states
        """
        # Store input for residual
        residual = hidden
        
        # Encode to Taylor coefficients
        coeffs = self.bridge.encode(hidden)  # [batch, seq_len, max_terms]
        
        # Apply operation(s)
        if op is not None:
            # Hard operation selection
            result = self._apply_op(op, coeffs)
        else:
            # Soft operation selection (differentiable)
            result = self._soft_apply(hidden, coeffs, temperature)
        
        # Decode back to hidden space
        output = self.bridge.decode(result)
        
        # Residual connection with learnable gate
        if self.residual:
            gate = torch.sigmoid(self.gate)
            output = gate * output + (1 - gate) * residual
        
        return output
    
    def _apply_op(self, op: str, coeffs: torch.Tensor) -> torch.Tensor:
        """Apply a specific symbolic operation."""
        if op == 'identity':
            return coeffs
        elif op == 'derivative':
            return self._batched_derivative(coeffs)
        elif op == 'integrate':
            return self._batched_integrate(coeffs)
        elif op == 'square':
            return self._batched_multiply(coeffs, coeffs)
        elif op == 'negate':
            return -coeffs
        else:
            raise ValueError(f"Unknown operation: {op}")
    
    def _soft_apply(
        self,
        hidden: torch.Tensor,
        coeffs: torch.Tensor,
        temperature: float
    ) -> torch.Tensor:
        """
        Soft-select and blend operations based on learned selector.
        Fully differentiable.
        """
        # Compute operation probabilities from hidden state
        # Use mean pooling over sequence for global decision
        if hidden.dim() == 3:
            pooled = hidden.mean(dim=1)  # [batch, hidden_dim]
        else:
            pooled = hidden
        
        op_logits = self.op_selector(pooled) / temperature  # [batch, num_ops]
        op_probs = F.softmax(op_logits, dim=-1)  # [batch, num_ops]
        
        # Apply each operation and blend
        results = []
        for i, op_name in enumerate(self.OPS[:self.num_ops]):
            op_result = self._apply_op(op_name, coeffs)  # [batch, seq, terms]
            results.append(op_result)
        
        # Stack: [num_ops, batch, seq, terms]
        stacked = torch.stack(results, dim=0)
        
        # Weighted sum: [batch, seq, terms]
        # op_probs: [batch, num_ops] -> [num_ops, batch, 1, 1]
        weights = op_probs.t().unsqueeze(-1).unsqueeze(-1)
        if coeffs.dim() == 2:
            weights = weights.squeeze(-1)
        
        blended = (weights * stacked).sum(dim=0)
        return blended
    
    def _batched_derivative(self, coeffs: torch.Tensor) -> torch.Tensor:
        """Vectorized derivative over batch dimension."""
        n = coeffs.shape[-1]
        # d/dx of a_n x^n = n * a_n x^(n-1)
        # Shift left and scale by index
        indices = torch.arange(n, dtype=coeffs.dtype, device=coeffs.device)
        result = coeffs * indices
        # Shift left
        return F.pad(result[..., 1:], (0, 1))
    
    def _batched_integrate(self, coeffs: torch.Tensor) -> torch.Tensor:
        """Vectorized integration over batch dimension."""
        n = coeffs.shape[-1]
        # ∫ a_n x^n dx = a_n/(n+1) x^(n+1)
        indices = torch.arange(n, dtype=coeffs.dtype, device=coeffs.device) + 1
        result = coeffs / indices
        # Shift right (constant term = 0)
        return F.pad(result[..., :-1], (1, 0))
    
    def _batched_multiply(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Batched polynomial multiplication using FFT."""
        n = a.shape[-1]
        # FFT convolution
        a_fft = torch.fft.fft(a, n=2*n, dim=-1)
        b_fft = torch.fft.fft(b, n=2*n, dim=-1)
        c = torch.fft.ifft(a_fft * b_fft, dim=-1).real[..., :n]
        return c


class SymbolicHead(nn.Module):
    """
    Output head that produces structured symbolic predictions.
    
    Can predict:
    - Operation type (derivative, solve, etc.)
    - Polynomial coefficients
    - Confidence scores
    """
    def __init__(
        self,
        hidden_dim: int,
        max_terms: int = 32,
        num_ops: int = 8
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_terms = max_terms
        self.num_ops = num_ops
        
        # Operation classifier
        self.op_head = nn.Linear(hidden_dim, num_ops)
        
        # Coefficient predictor
        self.coeff_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, max_terms)
        )
        
        # Confidence predictor
        self.confidence_head = nn.Linear(hidden_dim, 1)
    
    def forward(self, hidden: torch.Tensor) -> dict:
        """
        Predict symbolic structure from hidden states.
        
        Args:
            hidden: [batch, hidden_dim] or [batch, seq_len, hidden_dim]
        
        Returns:
            dict with:
                - op_logits: [batch, num_ops] operation probabilities
                - coeffs: [batch, max_terms] predicted coefficients
                - confidence: [batch, 1] confidence score
        """
        # Pool if sequence input
        if hidden.dim() == 3:
            hidden = hidden.mean(dim=1)
        
        return {
            'op_logits': self.op_head(hidden),
            'coeffs': self.coeff_head(hidden),
            'confidence': torch.sigmoid(self.confidence_head(hidden))
        }


class NousTransformerBlock(nn.Module):
    """
    A transformer block with integrated Nous symbolic reasoning.
    
    Structure:
        x -> LayerNorm -> Attention -> + residual
          -> LayerNorm -> FFN -> + residual  
          -> LayerNorm -> NousLayer -> + residual (optional)
    """
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        max_terms: int = 16,
        use_nous: bool = True
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_nous = use_nous
        
        # Standard transformer components
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        # Nous integration
        if use_nous:
            self.ln3 = nn.LayerNorm(hidden_dim)
            self.nous = NousLayer(hidden_dim, max_terms)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor = None,
        nous_op: str = None
    ) -> torch.Tensor:
        """
        Forward pass with optional symbolic reasoning.
        
        Args:
            x: [batch, seq_len, hidden_dim]
            attention_mask: Optional attention mask
            nous_op: Optional specific Nous operation
        """
        # Self-attention
        h = self.ln1(x)
        h, _ = self.attn(h, h, h, key_padding_mask=attention_mask)
        x = x + h
        
        # FFN
        h = self.ln2(x)
        h = self.ffn(h)
        x = x + h
        
        # Nous symbolic reasoning
        if self.use_nous:
            h = self.ln3(x)
            h = self.nous(h, op=nous_op)
            x = x + h
        
        return x


# Demo
if __name__ == "__main__":
    print("=== Nous LLM Integration Demo ===")
    
    batch_size = 4
    seq_len = 16
    hidden_dim = 256
    
    # Test NousLayer
    layer = NousLayer(hidden_dim=hidden_dim, max_terms=16)
    x = torch.randn(batch_size, seq_len, hidden_dim)
    
    print(f"\nInput shape: {x.shape}")
    
    # Hard operation
    y = layer(x, op='derivative')
    print(f"After derivative: {y.shape}")
    
    # Soft operation selection
    y = layer(x)
    print(f"After soft selection: {y.shape}")
    
    # Test gradient flow
    y.sum().backward()
    print(f"Gradients flow: ✓")
    
    # Test SymbolicHead
    head = SymbolicHead(hidden_dim=hidden_dim)
    out = head(x)
    print(f"\nSymbolicHead output:")
    print(f"  op_logits: {out['op_logits'].shape}")
    print(f"  coeffs: {out['coeffs'].shape}")
    print(f"  confidence: {out['confidence'].shape}")
    
    # Test NousTransformerBlock
    block = NousTransformerBlock(hidden_dim=hidden_dim)
    y = block(x)
    print(f"\nNousTransformerBlock output: {y.shape}")
    
    print("\n✓ All components working!")
