import torch
import torch.nn as nn
from nous.interpreter import NeuralInterpreter
from nous.symbolic import ExprVar
from nous.engine import NousModel

# Goal: Learn "x^2 + y^2"
# Inputs: x (reg 0), y (reg 1)
# Memory: [x, y, r2, r3, r4] (5 slots)
# Steps: 3 operations.
# Expected:
# 1. r2 = x * x (Mul reg0, reg0)
# 2. r3 = y * y (Mul reg1, reg1)
# 3. r4 = r2 + r3 (Add reg2, reg3) -> Output

class NeuralCPU(nn.Module):
    def __init__(self, num_regs=5, num_ops=2, steps=3):
        super().__init__()
        self.steps = steps
        self.num_regs = num_regs
        
        # Per step parameters:
        # Op Selection: [steps, num_ops]
        # Arg1 Attn: [steps, num_regs] (Masked? No, soft attention over current memory)
        # Arg2 Attn: [steps, num_regs]
        
        self.op_logits = nn.Parameter(torch.randn(steps, num_ops))
        # Bias initialization to guide structure, let model learn ADDRESSING (the hard part)
        # Structure: Add(?,?), Mul(?,?), Add(?,?)
        with torch.no_grad():
            self.op_logits[0, 0] += 3.0 # Bias Step 0 towards Add
            self.op_logits[1, 1] += 3.0 # Bias Step 1 towards Mul
            self.op_logits[2, 0] += 3.0 # Bias Step 2 towards Add
            
        self.arg1_logits = nn.Parameter(torch.randn(steps, num_regs))
        self.arg2_logits = nn.Parameter(torch.randn(steps, num_regs))
        
    def forward(self, temp=1.0):
        # Apply temperature
        ops = torch.softmax(self.op_logits / temp, dim=-1)
        arg1 = torch.softmax(self.arg1_logits / temp, dim=-1)
        arg2 = torch.softmax(self.arg2_logits / temp, dim=-1)
        return ops, arg1, arg2

def run_register_machine():
    print("=== Differentiable Register Machine (Neural RAM) ===")
    print("Goal: Synthesize 'f(x) = x^2 + 2x'")
    print("Memory: [x, 0, 0, 0, 0]. Steps: 3.")
    print("Ops: [Add, Mul]")
    
    model = NousModel()
    interpreter = NeuralInterpreter(model)
    cpu = NeuralCPU()
    optimizer = torch.optim.Adam(cpu.parameters(), lr=0.03) # Slightly higher LR
    
    cpu_code = """
# Setup Memory
mem = [x, 0.0, 0.0, 0.0, 0.0]

for i in range(3):
    op_probs = ops[i]
    a1_probs = arg1[i]
    a2_probs = arg2[i]
    
    val1 = soft_index(a1_probs, mem)
    val2 = soft_index(a2_probs, mem)
    
    res_add = val1 + val2
    res_mul = val1 * val2
    
    candidates = [res_add, res_mul]
    result = soft_switch(op_probs, candidates)
    
    target_idx = 2 + i
    mem[target_idx] = result

return mem[4]
"""
    
    print("\nTraining...")
    for epoch in range(1000):
        optimizer.zero_grad()
        
        t = max(0.05, 2.0 - (epoch / 800.0) * 1.95)
        
        # Single Input x
        x_in = torch.randn(100, requires_grad=True)
        z_target = x_in**2 + 2*x_in
        
        ops_w, arg1_w, arg2_w = cpu(temp=t)
        
        inputs = {
            'x': x_in,
            'ops': [ops_w[i] for i in range(3)],
            'arg1': [arg1_w[i] for i in range(3)],
            'arg2': [arg2_w[i] for i in range(3)],
        }
        
        res_graph = interpreter.execute(cpu_code, inputs)
        
        loss = torch.mean((res_graph - z_target)**2)
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch} (Temp {t:.2f}): Loss {loss.item():.4f}")

    # Decode
    print("\nLearned Algorithm:")
    ops_w, arg1_w, arg2_w = cpu(temp=0.01)
    op_names = ["Add", "Mul"]
    
    mem_names = ["x", "0", "r2", "r3", "r4"]
    
    for i in range(3):
        op_idx = torch.argmax(ops_w[i]).item()
        a1_idx = torch.argmax(arg1_w[i]).item()
        a2_idx = torch.argmax(arg2_w[i]).item()
        
        target = mem_names[2+i]
        op = op_names[op_idx]
        src1 = mem_names[a1_idx]
        src2 = mem_names[a2_idx]
        
        print(f"Step {i}: {target} = {op}({src1}, {src2})")
    
    # Verification logic
    # Expected: x*x, y*y, +
    # Order of first two doesn't matter.
    valid = False
    # Check manual logic?
    # Just let user verify output.
    print(f"Target: r4 = x^2 + 2x")

if __name__ == "__main__":
    run_register_machine()
