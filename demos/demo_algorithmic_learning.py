"""
demo_algorithmic_learning.py

Demonstrates Nous V2.0 capabilities:
- Learning to control a loop (`soft_while`) to perform a calculation.
- Task: Learn to calculate x^TARGET_POWER without being told "how many times".
- The controller must learn a termination condition based on a hidden counter.
"""

import torch
import torch.nn as nn
from nous.interpreter import NeuralInterpreter
from nous.engine import NousModel

TARGET_POWER = 5  # We want to learn to multiply x, 5 times (x^5)

class LoopController(nn.Module):
    """
    A learned controller that manages the loop state.
    It sees a hidden 'counter' state and decides:
    1. Whether to continue looping (logit).
    2. How to update the counter (delta).
    """
    def __init__(self):
        super().__init__()
        # Input: [1] hidden state
        self.net = nn.Sequential(
            nn.Linear(1, 16),
            nn.Tanh(),
            nn.Linear(16, 2) 
        )
        # Initialize output layer to encourage looping initially
        # Bias the 'logit' (index 0) to be positive (continue looping)
        self.net[-1].bias.data[0] = 2.0 
        
    def forward(self, hidden_state):
        # hidden_state is scalar tensor
        out = self.net(hidden_state.unsqueeze(0)).squeeze(0)
        logit = out[0]
        update = out[1]
        return logit, update

def run_demo():
    print(f"=== Algorithmic Learning Demo: Learning x^{TARGET_POWER} ===")
    print("Goal: Learn to loop exactly N times using soft_while.")
    
    model = NousModel()
    interpreter = NeuralInterpreter(model)
    controller = LoopController()
    optimizer = torch.optim.Adam(controller.parameters(), lr=0.01)
    
    # Python Algorithm to be learned
    # State: (accumulator, hidden_counter)
    # Body: acc = acc + x; hidden = hidden + learned_update
    # Cond: controller(hidden) -> logit
    
    code = """
# Inputs: x, initial_hidden, initial_acc
# State tuple: (accumulator, hidden_counter)
state = (initial_acc, initial_hidden)

def condition(state):
    acc, hidden = state
    logit, update = controller(hidden)
    return logit

def body(state):
    acc, hidden = state
    logit, update = controller(hidden)
    
    # Update logic
    new_acc = acc * x
    new_hidden = hidden + update
    
    return (new_acc, new_hidden)

# Run soft_while loop
final_state = soft_while(condition, body, state, max_iters=10)
return final_state[0] # Return accumulator
"""

    print("\nTraining...")
    for step in range(2000):
        optimizer.zero_grad()
        
        # Train on range [0.8, 1.2] to prevent explosion
        x = torch.rand(1) * 0.4 + 0.8
        target = x ** TARGET_POWER
        
        inputs = {
            'x': x.requires_grad_(True),
            'initial_hidden': torch.tensor([0.0], requires_grad=True),
            'controller': controller,
            'initial_acc': torch.tensor([1.0], requires_grad=True)
        }
        
        # Execute
        y_pred = interpreter.execute(code, inputs)
        
        # Loss
        loss = (y_pred - target) ** 2
        loss.backward()
        
        # Gradient Clipping to prevent explosion
        torch.nn.utils.clip_grad_norm_(controller.parameters(), 1.0)
        
        optimizer.step()
        
        if step % 50 == 0:
            # Check interpretation
            err = abs(y_pred.item() - target.item())
            print(f"Step {step}: Loss {loss.item():.6f}, Error {err:.6f}, Pred {y_pred.item():.4f} vs Tgt {target.item():.4f}")

    print("\nVerification:")
    x_test = torch.tensor([1.1], requires_grad=True)
    inputs = {
        'x': x_test,
        'initial_hidden': torch.tensor([0.0], requires_grad=True),
        'controller': controller,
        'initial_acc': torch.tensor([1.0], requires_grad=True)
    }
    y_final = interpreter.execute(code, inputs)
    tgt = 1.1**TARGET_POWER
    print(f"Input: 1.1")
    print(f"Target: 1.1^{TARGET_POWER} = {tgt:.4f}")
    print(f"Output: {y_final.item():.4f}")
    
    if abs(y_final.item() - tgt) < 0.05:
        print("🎉 SUCCESS: Model learned to loop the correct number of times!")
    else:
        print("❌ FAIL: Did not converge.")

if __name__ == "__main__":
    run_demo()
