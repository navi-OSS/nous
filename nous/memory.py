"""
Nous Neural Memory - Differentiable Heap.

Implements a Neural Turing Machine-style memory bank with soft attention
for reading and writing, enabling algorithms that require addressable memory.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralMemory(nn.Module):
    """
    A differentiable memory bank using attention mechanisms.
    
    Read: Weighted sum over all memory slots using address weights.
    Write: Weighted update to all memory slots using address weights.
    """
    def __init__(self, num_slots=16, slot_size=32):
        super().__init__()
        self.num_slots = num_slots
        self.slot_size = slot_size
        # Memory bank: [num_slots, slot_size]
        self.memory = nn.Parameter(torch.zeros(num_slots, slot_size), requires_grad=False)
        
    def reset(self):
        """Clear memory to zeros."""
        self.memory.data.zero_()
        
    def init_random(self, std=0.1):
        """Initialize memory with random values."""
        self.memory.data.normal_(0, std)
        
    def read(self, address_weights):
        """
        Read from memory using soft attention weights.
        """
        # Normalize weights to probabilities
        weights = F.softmax(address_weights, dim=-1)
        # Weighted sum: sum_i(w_i * M_i)
        return torch.matmul(weights, self.memory)
    
    def top_k_attention(self, query, k=5, beta=10.0):
        """
        Differentiable top-k retrieval approximation.
        Masks all but the top k similarities, then performs softmax.
        """
        # Cosine similarity
        query_norm = F.normalize(query.unsqueeze(0), dim=-1)
        memory_norm = F.normalize(self.memory, dim=-1)
        similarity = torch.matmul(query_norm, memory_norm.T).squeeze(0)
        
        # Soft-masking: keep only top k
        topk_vals, _ = torch.topk(similarity, k=min(k, self.num_slots))
        threshold = topk_vals[-1]
        
        # Differentiable mask: sigmoid(sharpen * (sim - thresh))
        mask = torch.sigmoid(beta * (similarity - threshold))
        
        # Softmax over masked similarities
        return F.softmax(beta * similarity + (1.0 - mask) * -1e9, dim=-1)
    
    def write(self, address_weights, value, erase_strength=0.0):
        """
        Write to memory using soft attention weights.
        
        Args:
            address_weights: [num_slots] tensor of attention weights
            value: [slot_size] tensor of content to write
            erase_strength: optional erase gate (0=pure add, 1=pure replace)
        """
        weights = F.softmax(address_weights, dim=-1)
        
        # Erase step: M = M * (1 - w * e)
        if erase_strength > 0:
            erase = weights.unsqueeze(1) * erase_strength
            self.memory.data = self.memory * (1 - erase)
        
        # Add step: M = M + w * v
        add = weights.unsqueeze(1) * value.unsqueeze(0)
        
        # Functional update to preserve gradients (don't use .data)
        # We overwrite self.memory (it stops being a leaf Parameter and becomes an activation)
        # We must dereference it from nn.Module parameters first to avoid TypeError
        new_mem = self.memory + add
        if 'memory' in self._parameters:
            del self._parameters['memory']
        self.memory = new_mem
        
    def content_addressing(self, query, beta=1.0):
        """
        Compute content-based address weights using cosine similarity.
        
        Args:
            query: [slot_size] tensor to search for
            beta: sharpening factor (higher = more focused)
        Returns:
            [num_slots] tensor of attention weights
        """
        # Cosine similarity
        query_norm = F.normalize(query.unsqueeze(0), dim=-1)
        memory_norm = F.normalize(self.memory, dim=-1)
        similarity = torch.matmul(query_norm, memory_norm.T).squeeze(0)
        
        # Sharpen with beta
        return F.softmax(beta * similarity, dim=-1)
    
    def location_addressing(self, prev_weights, shift_weights, gamma=1.0):
        """
        Compute location-based address weights via convolution shift.
        
        Args:
            prev_weights: [num_slots] previous step's weights
            shift_weights: [3] tensor for shift (-1, 0, +1)
            gamma: sharpening factor
        Returns:
            [num_slots] tensor of shifted weights
        """
        # Circular convolution for shift
        shifted = F.conv1d(
            prev_weights.unsqueeze(0).unsqueeze(0),
            shift_weights.unsqueeze(0).unsqueeze(0),
            padding=1
        ).squeeze()[:self.num_slots]
        
        # Sharpen
        sharpened = shifted ** gamma
        return sharpened / (sharpened.sum() + 1e-8)


class NeuralStack(nn.Module):
    """
    A differentiable stack using Neural Memory primitives.
    Push/Pop use a soft pointer that can be learned.
    """
    def __init__(self, capacity=16, element_size=32):
        super().__init__()
        self.memory = NeuralMemory(num_slots=capacity, slot_size=element_size)
        self.pointer = nn.Parameter(torch.tensor(0.0))  # Soft stack pointer
        
    def push(self, value, strength=1.0):
        """Push a value onto the stack."""
        # Address weights centered at current pointer
        ptr = torch.sigmoid(self.pointer) * (self.memory.num_slots - 1)
        addresses = torch.arange(self.memory.num_slots, dtype=torch.float32)
        weights = torch.exp(-((addresses - ptr) ** 2) / 0.5)  # Gaussian around pointer
        weights = weights / weights.sum()
        
        self.memory.write(weights * strength, value)
        self.pointer.data = self.pointer + strength  # Move pointer up
        
    def pop(self, strength=1.0):
        """Pop a value from the stack."""
        self.pointer.data = self.pointer - strength  # Move pointer down first
        
        ptr = torch.sigmoid(self.pointer) * (self.memory.num_slots - 1)
        addresses = torch.arange(self.memory.num_slots, dtype=torch.float32)
        weights = torch.exp(-((addresses - ptr) ** 2) / 0.5)
        weights = weights / weights.sum()
        
        return self.memory.read(weights)


# Demo
if __name__ == "__main__":
    print("=== Neural Memory Demo ===")
    mem = NeuralMemory(num_slots=8, slot_size=4)
    mem.init_random()
    
    # Write at position 3
    address = torch.zeros(8)
    address[3] = 10.0  # High logit at position 3
    value = torch.tensor([1.0, 2.0, 3.0, 4.0])
    mem.write(address, value)
    
    # Read back
    readout = mem.read(address)
    print(f"Wrote [1,2,3,4] at slot 3, read back: {readout.tolist()}")
    
    # Content addressing
    query = torch.tensor([1.0, 2.0, 3.0, 4.0])
    weights = mem.content_addressing(query, beta=5.0)
    print(f"Content-based lookup for [1,2,3,4]: slot weights = {weights.tolist()}")
