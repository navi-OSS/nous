import torch
import torch.nn as nn
from nous.interpreter import NeuralInterpreter
from nous.symbolic import ExprVar
from nous.engine import NousModel

# Goal: Learn to compute "2x + 1"
# The controller outputs logits for a sequence of 2 operations.
# Ops Library:
# 0: Add(x, 1)
# 1: Mul(x, 2)
# 2: Square(x)
# 3: Identity(x)
# 4: Add(x, x)

# Target Program: Mul(x, 2) then Add(res, 1) -> 2x+1
# Or Add(x,x) then Add(res, 1) -> 2x+1

class NeuralProgrammer(nn.Module):
    def __init__(self, num_ops=5, prog_len=2):
        super().__init__()
        # Learnable logits for each step in the program.
        # Step 0: [logits] -> Softmax -> Weights
        # Step 1: [logits] -> Softmax -> Weights
        self.prog_logits = nn.Parameter(torch.randn(prog_len, num_ops))
        
    def forward(self, temp=1.0):
        # Return probability distributions for each instruction
        return torch.softmax(self.prog_logits / temp, dim=-1)

def run_program_synthesis():
    print("=== Differentiable Neural Programmer ===")
    print("Goal: Synthesize 'f(x) = 2x + 1' from scratch.")
    print("Available Ops: [Add(x,1), Mul(x,2), Square(x), Identity(x), Add(x,x)]")
    
    model = NousModel()
    interpreter = NeuralInterpreter(model)
    programmer = NeuralProgrammer()
    optimizer = torch.optim.Adam(programmer.parameters(), lr=0.05)
    
    # The Meta-Interpreter Script
    # It takes Probabilities for instructions and applies them.
    # We use a "Register Machine" style or simple composition?
    # Simple Composition:
    # res = x
    # step 0: res = soft_switch(probs[0], [op0(res), op1(res)...])
    # step 1: res = soft_switch(probs[1], [op0(res), op1(res)...])
    
    meta_code = """
# input 'x'
res = x
val_1 = 1.0
val_2 = 2.0

# Instruction Execution Loop
# We manually unroll 2 steps bc 'weights' is list of lists
for i in range(2):
    w = weights[i] # Probs for this step
    
    # Candidate Operations on 'res'
    op0 = res + val_1       # Add 1
    op1 = res * val_2       # Mul 2
    op2 = res ** val_2      # Square
    op3 = res               # Identity
    op4 = res + res         # Add x,x
    
    candidates = [op0, op1, op2, op3, op4]
    
    # Soft Selection
    res = soft_switch(w, candidates)

return res
"""
    
    print("\nTraining...")
    for epoch in range(500):
        optimizer.zero_grad()
        
        # Temperature Annealing: Start soft (2.0), end hard (0.1)
        # 500 epochs.
        # epoch 0: 2.0
        # epoch 500: 0.05
        t = max(0.05, 2.0 - (epoch / 250.0) * 1.95)
        
        # Select random x for training? Or fixed set?
        # Let's use batch of random x.
        # But symbolic graph is built once per execution.
        # To handle batch, we pass tensor for x.
        
        # Training Data
        x_train = torch.randn(20, requires_grad=True)
        y_target = 2 * x_train + 1
        
        # 1. Get Program Weights
        code_probs = programmer(temp=t) # [2, 5]
        
        # 2. Build Graph
        # We need to pass weights as lists of ExprVars or bind directly?
        # weights[i] is a tensor [p0, p1, p2, p3, p4]
        # We need to decompose it into individual scalars for the list?
        # NeuralInterpreter handles lists of tensors if we pass them.
        # But 'weights' in python code will be access via weights[i].
        # So inputs['weights'] should be a list of 2 tensors (or lists).
        # Let's pass list of lists for clarity in python, or tensor?
        # If we pass tensor, weights[i] gives slice.
        # Let's decompose into list of lists of probabilities for max symbolic compat using ExprVar?
        # Actually, let's just pass the tensor. `weights[0]` returns a tensor.
        # `soft_switch(w, cand)` expects w to be a list of scalars.
        # If w is a tensor, `len(w)` works? 
        # Python len(tensor) works (dim 0).
        # Iterating `for i in range(len(w))` works.
        # `w[i]` returns 0-d tensor.
        # `soft_switch` does `w[i] * c[i]`.
        # ExprConst (from w[i]) * SymbolicNode.
        # This should work if `_wrap` handles 0-d tensor -> ExprConst/float.
        # My `_wrap` handles tensors.
        
        inputs = {
            'x': x_train, # Tensor!
            'weights': code_probs # Tensor [2, 5]
        }
        
        # 3. Execute
        # x is tensor, so intermediate 'res' will be tensors (eager evaluation path?)
        # WAIT. If x is tensor, and weights are tensor, is it Symbolic?
        # Interpreter checks inputs using `_wrap`.
        # `_wrap(tensor)` -> `ExprConst` (if numel=1 or handle batch?)
        # My `_wrap`: `if numel()==1 return ExprConst(float)`.
        # If numel > 1 ? It returns the tensor itself (line 58 symbolic.py: `return other`).
        # So `SymbolicNode` ops must handle Tensor operands.
        # `ExprAdd.__add__` -> `ExprAdd(self, other)`.
        # If both are tensors? It falls back to Tensor.__add__.
        # So execution is purely Eager PyTorch!
        # This is fine! We want gradients.
        # But `soft_switch` iterates.
        
        # Wait, if execution is eager tensors, `soft_switch` must work on tensors.
        # My `soft_switch` wrapper takes list `weights`.
        # If `weights` is a 1D tensor, iterating it yields 0-d tensors.
        # `soft_switch` loops.
        # `candidates` list contains result tensors.
        # `w[i] * c[i]`: 0-d tensor * Tensor -> Tensor.
        # Sums up.
        # Perfectly differentiable eager execution.
        
        res_graph = interpreter.execute(meta_code, inputs) 
        # res_graph is a Tensor (result of execution) because all inputs were tensors.
        # No 'SymbolicNode' involved if inputs are raw tensors?
        # Yes, `_wrap` returns raw tensor if numel > 1.
        
        y_pred = res_graph
        
        loss = torch.mean((y_pred - y_target)**2)
        loss.backward()
        optimizer.step()
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch} (Temp {t:.2f}): Loss {loss.item():.4f}")
            
    # Decode Program
    print("\nLearned Program:")
    ops_names = ["Add(1)", "Mul(2)", "Square", "Identity", "Add(x,x)"]
    probs = programmer(temp=0.01).detach() # Hard argmax
    
    for i in range(2):
        p = probs[i]
        best_idx = torch.argmax(p).item()
        print(f"Step {i}: {ops_names[best_idx]} (Conf: {torch.max(p):.2f})")
        
    expected_1 = ["Mul(2)", "Add(1)"] # 2x + 1
    expected_2 = ["Add(x,x)", "Add(1)"] # 2x + 1
    
    step0 = ops_names[torch.argmax(probs[0]).item()]
    step1 = ops_names[torch.argmax(probs[1]).item()]
    
    if (step0 == "Mul(2)" and step1 == "Add(1)") or (step0 == "Add(x,x)" and step1 == "Add(1)"):
        print("SUCCESS: Synthesized '2x + 1' (Hard)")
    elif (step1 == "Mul(2)" and step0 == "Identity") and step0 == "Add(1)": 
        pass 
    elif (step0 == "Add(1)" and step1 == "Mul(2)"): # 2(x+1) = 2x+2
         print("FAIL: Order wrong")
    else:
         print(f"Result: {step0} -> {step1}")

if __name__ == "__main__":
    run_program_synthesis()
