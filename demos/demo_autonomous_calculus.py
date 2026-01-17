import torch
from nous.workspace import NousWorkspace
from nous.symbolic import ExprVar

def demo_autonomous():
    print("=== Nous V8.1: System 2 Autonomous Solver ===")
    ws = NousWorkspace()
    
    # 1. Define Problem
    print("[1] Problem: Find critical points of f(x,y) = x^3 + y^3 - 3xy")
    
    # 2. Symbolic Definition
    code = """
x = ExprVar('x')
y = ExprVar('y')
f = x**3 + y**3 - 3*x*y

# Compute Gradient (System of Equations)
# V8.2: Use simplify() to remove redundant ops (e.g. 1*x, x+0)
df_dx = f.diff('x').simplify()
df_dy = f.diff('y').simplify()

# "System 2" Call: Solve for roots of the gradient system
# The engine autonomously runs the optimization and snaps to exact values.
roots = solve_system(equations=[df_dx, df_dy], vars=['x', 'y'])

return roots
"""
    
    # 3. Autonomous Solution
    print("[2] Thinking... (Optimizing & Snapping)")
    solutions_symbolic = ws.run(code)
    
    # Unwrap symbolic results
    def unwrap(x):
        if hasattr(x, 'value'): return unwrap(x.value)
        if isinstance(x, list): return [unwrap(v) for v in x]
        return x
        
    solutions = unwrap(solutions_symbolic)
    
    # 4. Result
    print(f"\n[3] Solution: Found {len(solutions)} critical points.")
    for i, point in enumerate(solutions):
        print(f"    Point {i+1}: {point}")
        
    # Validation
    expected = [[0.0, 0.0], [1.0, 1.0]]
    # Sort for comparison
    solutions.sort()
    
    # Check match
    success = True
    if len(solutions) != 2:
        success = False
    else:
        for s, e in zip(solutions, expected):
            # approximate check in case snapping missed (but it shouldn't)
            if not (abs(s[0] - e[0]) < 1e-4 and abs(s[1] - e[1]) < 1e-4):
                success = False
    
    if success:
        print("\nSUCCESS: Exact critical points found autonomously.")
    else:
        print("\nFAILURE: Did not find expected roots (0,0) and (1,1).")

if __name__ == "__main__":
    demo_autonomous()
