import torch
import torch.nn as nn
import math
from nous.interpreter import NeuralInterpreter
from nous.symbolic import ExprVar
from nous.engine import NousModel

# 1. The "Tiny LLM" (Policy Network)
# Input: State (Position, Velocity)
# Output: A flight plan (Sequence of 20 thrusts)
# Open-loop planning based on initial state.
class PlanningPolicy(nn.Module):
    def __init__(self, steps=20):
        super().__init__()
        self.steps = steps
        # Simple policy: Linear -> Tanh -> Linear -> Output Trajectory
        self.net = nn.Sequential(
            nn.Linear(2, 32),
            nn.Tanh(),
            nn.Linear(32, steps), # Outputs: [Thrust_0, ..., Thrust_19]
            nn.Tanh() 
        )
    
    def forward(self, state):
        # state is tensor [pos, vel]
        out = self.net(state)
        # Thrust: 0 to 40. (tanh+1)*20
        thrust_seq = (out + 1.0) * 20.0 
        return thrust_seq

def visualize_trajectory(trajectory, velocity_trace=None, thrust_trace=None):
    """Prints an ASCII chart of the landing."""
    print("\n=== Simulation Telemetry ===")
    max_h = max(p for p in trajectory)
    scale = 20.0 / (max_h + 1e-5)
    
    print(f"{'Time':<5} {'Alt (m)':<10} {'Vel (m/s)':<10} {'Thrust':<10} {'Visual':<20}")
    print("-" * 60)
    
    for t, pos in enumerate(trajectory):
        vel_str = f"{velocity_trace[t]:6.2f}" if velocity_trace else "N/A"
        thrust_str = f"{thrust_trace[t]:6.2f}" if thrust_trace else "N/A"
        
        # Altitude Bar
        bars = int(pos * scale)
        bar_str = '|' * bars
        if pos < 0.1: bar_str = '_' # Ground contact
        
        print(f"T{t:02d}  {pos:6.2f}     {vel_str}     {thrust_str}     {bar_str}")

def run_neural_landing():
    print("=== System 2 Experiment: Neural Rocket Landing (Dynamic) ===")
    print("Goal: Train a Neural Network to generate a 30-step flight plan to soft-land.")
    
    steps = 30
    model = NousModel()
    interpreter = NeuralInterpreter(model)
    policy = PlanningPolicy(steps=steps)
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.05) 
    
    physics_code = """
pos = start_pos
vel = start_vel
dt = 0.1
gravity = 9.8

for i in range(30):
    val = float(thrusts[i])
    accel = val - gravity
    vel = vel + accel * dt
    pos = pos + vel * dt

return pos, vel
"""
    
    # Needs to match symbolic code length
    physics_code_symbolic = """
pos = start_pos
vel = start_vel
dt = 0.1
gravity = 9.8

for i in range(30):
    thrust = thrusts[i]
    accel = thrust - gravity
    vel = vel + accel * dt
    pos = pos + vel * dt

return pos, vel
"""

    print("\nTraining...")
    
    for epoch in range(201):
        optimizer.zero_grad()
        start_pos_val = 20.0
        start_vel_val = 0.0
        state_tensor = torch.tensor([start_pos_val, start_vel_val])
        
        thrust_seq = policy(state_tensor)
        thrust_vars = [ExprVar(f't{i}') for i in range(steps)]
        inputs = {'start_pos': start_pos_val, 'start_vel': start_vel_val, 'thrusts': thrust_vars}
        
        context = {f't{i}': thrust_seq[i] for i in range(steps)}
        
        res_pos, res_vel = interpreter.execute(physics_code_symbolic, inputs)
        
        final_pos = model.expand(res_pos, center=context)[0]
        final_vel = model.expand(res_vel, center=context)[0]
        
        loss = final_pos**2 + final_vel**2
        loss.backward()
        optimizer.step()
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch}: Pos={final_pos.item():.2f}, Vel={final_vel.item():.2f}, Loss={loss.item():.2f}")

    print("\n=== Validation Flight ===")
    final_plan = policy(state_tensor).detach()
    params = {
        'start_pos': start_pos_val,
        'start_vel': start_vel_val,
        'thrusts': [float(t) for t in final_plan]
    }
    
    # Visualization Code with trace
    vis_code = """
pos = start_pos
vel = start_vel
dt = 0.1
gravity = 9.8
hist_pos = []
hist_vel = []

for i in range(30):
    val = float(thrusts[i])
    accel = val - gravity
    vel = vel + accel * dt
    pos = pos + vel * dt
    hist_pos.append(pos)
    hist_vel.append(vel)

return hist_pos, hist_vel
"""
    traj_pos, traj_vel = interpreter.execute(vis_code, params)
    
    def clean(res_list):
        out = []
        for x in res_list:
            if hasattr(x, 'value'): out.append(x.value)
            else: out.append(x)
        return out
        
    visualize_trajectory(clean(traj_pos), clean(traj_vel), [float(t) for t in final_plan])
    
    last_pos = clean(traj_pos)[-1]
    last_vel = clean(traj_vel)[-1]
    
    print(f"\nFinal State: Altitude {last_pos:.2f}m, Velocity {last_vel:.2f} m/s")
    if abs(last_pos) < 0.5 and abs(last_vel) < 2.0:
        print("SUCCESS: Soft Landing Achieved! 🚀")
    else:
        print("PARTIAL: Hard Landing / Crash.")

if __name__ == "__main__":
    run_neural_landing()
