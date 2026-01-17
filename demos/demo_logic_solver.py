"""
Nous V8.3: System 4 Logical Solver Demo

Demonstrates solving discrete boolean satisfiability (SAT) problems
using the differentiable `solve_logic` primitive.
"""
from nous.workspace import NousWorkspace

def demo():
    print("=== System 4: Logical Solver ===")
    ws = NousWorkspace()
    
    # Problem:
    # Find binary x, y such that:
    # 1. (x OR y) is True
    # 2. (NOT x)  is True
    # Expected: x=0, y=1
    
    import textwrap
    code = textwrap.dedent("""
    x = ExprVar('x')
    y = ExprVar('y')
    
    # Boolean Logic as Arithmetic (Soft Logic)
    # A OR B  => x + y - x*y
    # NOT A   => 1 - x
    # A AND B => x * y
    
    constraint1 = x + y - x*y  # x OR y
    constraint2 = 1 - x        # NOT x
    
    # Solve for variables that satisfy constraints (must evaluate to 1.0)
    # The engine automatically enforces x, y in {0, 1}
    solutions = solve_logic([constraint1, constraint2], vars=['x', 'y'])
    
    return solutions
    """)
    
    print("Solving SAT Problem: (x OR y) AND (NOT x)...")
    results = ws.run(code)
    
    print(f"Solutions: {results}")
    
    # Verify
    if [0, 1] in results or [0.0, 1.0] in results:
        print("SUCCESS: Found valid assignment x=0, y=1.")
    else:
        print("FAILURE: Did not find independent solution.")

if __name__ == "__main__":
    demo()
