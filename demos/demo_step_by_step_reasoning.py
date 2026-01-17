import torch
import torch.nn as nn
from nous.workspace import NousWorkspace
from nous.symbolic import ExprVar

def demo_reasoning():
    print("=== Nous: Differentiable Chain of Thought Demo ===")
    ws = NousWorkspace()
    
    # 1. Define the Reasoning Goal
    # We want the agent to start at [0,0] and reach [1,1] 
    # by taking discrete logical steps.
    target_state = torch.tensor([1.0, 1.0])
    
    # 2. Define the "Thinker" (The Agent's params)
    # It has a "step" vector and a "gate" logic.
    # We want it to learn: "Take a step" only when conditions are met.
    # For this simple demo, we let it learn a sequence of additions.
    
    # Learnable parameters
    params = {
        'thought_step': torch.randn(2, requires_grad=True),  # What to add
        'gate_logit': torch.randn(1, requires_grad=True),    # Whether to add it
        'initial_thought': torch.zeros(2, requires_grad=True) # Starting point
    }
    
    # Optimizer
    optimizer = torch.optim.Adam(params.values(), lr=0.1)
    
    # 3. The Reasoning Loop (Script)
    # This logic represents the agent's internal monologue.
    # "For 5 steps, consider updating my thought."
    code = """
state = initial_thought
step_val = thought_step
gate = gate_logit

def condition(s):
    # Differentiable loop condition (not used in fixed unroll, but good practice)
    return 1.0

def step_logic(s):
    # The "Core Reasoning Step":
    # state_new = state_old + step * sigmoid(gate)
    # This is a soft conditional update.
    # "If gate is high, add the step."
    
    # Using soft_if for explicit branching logic if needed, 
    # or just algebraic blending. Let's use algebraic for smooth optimization first.
    # But we can also use soft_if:
    # return soft_if(gate, s + step_val, s)
    
    # Let's use the algebraic form for clarity in the trace:
    update = step_val * sigmoid(gate)
    return s + update

# We use soft_while to unroll this reasoning chain 5 times.
# In Nous V8.0, this unrolls into a differentiable graph.
final_state = soft_while(condition, step_logic, state, max_iters=5)

return final_state
"""

    print(f"\n[Goal] Reach state {target_state.numpy()}")
    print("-" * 40)
    
    # 4. Training Loop (Meta-Reasoning)
    # We optimize the agent's "thinking process" to reach the conclusion.
    for i in range(50):
        optimizer.zero_grad()
        
        # Execute the reasoning chain
        # Nous builds the graph and returns the differentiable result (a SymbolicNode)
        final_thought_node = ws.run(code, params)
        
        # Evaluate the symbolic graph to get a tensor
        # We use max_terms=1 because we only care about the value (0-th derivative) here.
        # The inputs are already tensors, so 'center' is dummy.
        coeffs = final_thought_node.to_taylor(center=0.0, max_terms=1, hilbert=ws.model.hilbert)
        
        # Extract the value (0-th term). Terms are in the LAST dimension.
        # Works for [max_terms] scalar or [..., max_terms] vector.
        final_thought = coeffs[..., 0] 
        
        # Compute Loss: Distance from target conclusion
        loss = torch.norm(final_thought - target_state)
        
        loss.backward()
        optimizer.step()
        
        if i % 10 == 0:
            print(f"Iter {i:02d} | Loss: {loss.item():.4f} | Final Thought: {final_thought.detach().numpy()}")
            
    print("-" * 40)
    print("✓ Optimization Complete.")
    print(f"Final Conclusion: {final_thought.detach().numpy()}")
    print(f"Target: {target_state.numpy()}")
    
    # 5. Verify the "Logic" Learned
    # Did it learn to take steps?
    gate_prob = torch.sigmoid(params['gate_logit']).item()
    step_vec = params['thought_step'].detach().numpy()
    print(f"\n[Learned Logic]")
    print(f"- Step Vector: {step_vec}")
    print(f"- Confidence (Gate): {gate_prob:.4f}")
    
    if gate_prob > 0.5 and loss.item() < 0.1:
        print("\nSUCCESS: Agent learned a coherent reasoning chain to reach the target.")
    else:
        print("\nNOTE: Optimization might need more steps or tuning.")

if __name__ == "__main__":
    demo_reasoning()
