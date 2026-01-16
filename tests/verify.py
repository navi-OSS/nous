"""
Updated verification tests for the enhanced Nous engine.
"""
import torch
import math
from nous.engine import NousModel, NousHilbertCore, MultivariatePolynomial, TaylorODESolver

def test_algebra():
    """Test polynomial root solving across various degrees and types."""
    print("=" * 60)
    print("ALGEBRA TESTS")
    print("=" * 60)
    
    model = NousModel()
    tests_passed = 0
    tests_total = 0
    
    # Test 1: Simple quadratic with real roots
    print("\n1. Quadratic with real roots: x^2 - 5x + 6 = 0")
    coeffs = torch.tensor([[1.0, -5.0, 6.0]], dtype=torch.float64)
    roots = model.forward(coeffs, op='solve')
    expected = {2.0, 3.0}
    actual = {roots[0, 0, 0].item(), roots[0, 1, 0].item()}
    error = max(abs(roots[0, 0, 1].item()), abs(roots[0, 1, 1].item()))
    tests_total += 1
    if error < 1e-10:
        tests_passed += 1
        print("   ✓ PASS")
    else:
        print("   ✗ FAIL")
    
    # Test 2: Double root (improved with Newton polishing)
    print("\n2. Double root: (x-2)^2 = x^2 - 4x + 4")
    coeffs = torch.tensor([[1.0, -4.0, 4.0]], dtype=torch.float64)
    roots = model.forward(coeffs, op='solve')
    error = abs(roots[0, 0, 0].item() - 2.0) + abs(roots[0, 1, 0].item() - 2.0)
    print(f"   Error: {error:.2e}")
    tests_total += 1
    if error < 1e-12:  # Achieves hardware precision after clustering and Vieta-snapping
        tests_passed += 1
        print("   ✓ PASS")
    else:
        print("   ✗ FAIL")
    
    print(f"\n{'='*60}")
    print(f"Algebra: {tests_passed}/{tests_total} tests passed")
    return tests_passed, tests_total

def test_transcendental():
    """Test new transcendental function support."""
    print("\n" + "=" * 60)
    print("TRANSCENDENTAL FUNCTION TESTS")
    print("=" * 60)
    
    model = NousModel()
    tests_passed = 0
    tests_total = 0
    
    x = torch.tensor(0.5)
    
    # Test cos
    print("\n1. cos(0.5)")
    cos_coeffs = model.forward(None, op='get_identity', name='cos')
    cos_val = model.hilbert.eval_at(cos_coeffs, x)
    expected = torch.cos(x)
    error = abs(cos_val.item() - expected.item())
    print(f"   Expected: {expected.item():.10f}, Got: {cos_val.item():.10f}")
    print(f"   Error: {error:.2e}")
    tests_total += 1
    if error < 1e-10:
        tests_passed += 1
        print("   ✓ PASS")
    else:
        print("   ✗ FAIL")
    
    # Test log1p
    print("\n2. log(1+0.5)")
    log_coeffs = model.forward(None, op='get_identity', name='log1p')
    log_val = model.hilbert.eval_at(log_coeffs, x)
    expected = torch.log1p(x)
    error = abs(log_val.item() - expected.item())
    print(f"   Expected: {expected.item():.10f}, Got: {log_val.item():.10f}")
    print(f"   Error: {error:.2e}")
    tests_total += 1
    if error < 1e-6:  # log1p converges slower
        tests_passed += 1
        print("   ✓ PASS")
    else:
        print("   ✗ FAIL")
    
    # Test sinh
    print("\n3. sinh(0.5)")
    sinh_coeffs = model.forward(None, op='get_identity', name='sinh')
    sinh_val = model.hilbert.eval_at(sinh_coeffs, x)
    expected = torch.sinh(x)
    error = abs(sinh_val.item() - expected.item())
    print(f"   Expected: {expected.item():.10f}, Got: {sinh_val.item():.10f}")
    print(f"   Error: {error:.2e}")
    tests_total += 1
    if error < 1e-10:
        tests_passed += 1
        print("   ✓ PASS")
    else:
        print("   ✗ FAIL")
    
    # Test cosh
    print("\n4. cosh(0.5)")
    cosh_coeffs = model.forward(None, op='get_identity', name='cosh')
    cosh_val = model.hilbert.eval_at(cosh_coeffs, x)
    expected = torch.cosh(x)
    error = abs(cosh_val.item() - expected.item())
    print(f"   Expected: {expected.item():.10f}, Got: {cosh_val.item():.10f}")
    print(f"   Error: {error:.2e}")
    tests_total += 1
    if error < 1e-10:
        tests_passed += 1
        print("   ✓ PASS")
    else:
        print("   ✗ FAIL")
    
    print(f"\n{'='*60}")
    print(f"Transcendental: {tests_passed}/{tests_total} tests passed")
    return tests_passed, tests_total

def test_definite_integration():
    """Test definite integration."""
    print("\n" + "=" * 60)
    print("DEFINITE INTEGRATION TESTS")
    print("=" * 60)
    
    model = NousModel()
    tests_passed = 0
    tests_total = 0
    
    # Test: ∫[0,1] x^2 dx = 1/3
    print("\n1. ∫[0,1] x^2 dx = 1/3")
    coeffs = torch.tensor([0.0, 0.0, 1.0] + [0.0]*29, dtype=torch.float32)
    a = torch.tensor(0.0)
    b = torch.tensor(1.0)
    result = model.forward(coeffs, op='definite_integrate', a=a, b=b)
    expected = 1/3
    error = abs(result.item() - expected)
    print(f"   Expected: {expected:.10f}, Got: {result.item():.10f}")
    print(f"   Error: {error:.2e}")
    tests_total += 1
    if error < 1e-6:
        tests_passed += 1
        print("   ✓ PASS")
    else:
        print("   ✗ FAIL")
    
    # Test: ∫[0,π] sin(x) dx = 2
    print("\n2. ∫[0,π] sin(x) dx ≈ 2")
    sin_coeffs = model.forward(None, op='get_identity', name='sin')
    a = torch.tensor(0.0)
    b = torch.tensor(math.pi, dtype=torch.float32)
    result = model.hilbert.definite_integrate(sin_coeffs, a, b)
    expected = 2.0
    error = abs(result.item() - expected)
    print(f"   Expected: {expected:.10f}, Got: {result.item():.10f}")
    print(f"   Error: {error:.2e}")
    tests_total += 1
    if error < 1e-4:  # Taylor approximation for sin
        tests_passed += 1
        print("   ✓ PASS")
    else:
        print("   ✗ FAIL")
    
    print(f"\n{'='*60}")
    print(f"Definite Integration: {tests_passed}/{tests_total} tests passed")
    return tests_passed, tests_total

def test_multivariate():
    """Test multivariate polynomial operations."""
    print("\n" + "=" * 60)
    print("MULTIVARIATE POLYNOMIAL TESTS")
    print("=" * 60)
    
    tests_passed = 0
    tests_total = 0
    
    # Test: f(x,y) = x^2 + 2xy + y^2 = (x+y)^2
    print("\n1. f(x,y) = x^2 + 2xy + y^2 at (1, 2)")
    mv = MultivariatePolynomial(num_vars=2, max_degree=3)
    coeffs = torch.zeros(3, 3)
    coeffs[2, 0] = 1.0  # x^2
    coeffs[1, 1] = 2.0  # 2xy
    coeffs[0, 2] = 1.0  # y^2
    
    points = torch.tensor([[1.0, 2.0]])
    result = mv.evaluate(coeffs, points)
    expected = (1 + 2)**2  # = 9
    error = abs(result.item() - expected)
    print(f"   Expected: {expected}, Got: {result.item():.6f}")
    tests_total += 1
    if error < 1e-6:
        tests_passed += 1
        print("   ✓ PASS")
    else:
        print("   ✗ FAIL")
    
    # Test partial derivative
    print("\n2. ∂f/∂x at f(x,y) = x^2 + 2xy + y^2")
    df_dx = mv.partial_derivative(coeffs, var_index=0)
    # ∂f/∂x = 2x + 2y
    points = torch.tensor([[1.0, 2.0]])
    result = mv.evaluate(df_dx, points)
    expected = 2*1 + 2*2  # = 6
    error = abs(result.item() - expected)
    print(f"   Expected: {expected}, Got: {result.item():.6f}")
    tests_total += 1
    if error < 1e-6:
        tests_passed += 1
        print("   ✓ PASS")
    else:
        print("   ✗ FAIL")
    
    print(f"\n{'='*60}")
    print(f"Multivariate: {tests_passed}/{tests_total} tests passed")
    return tests_passed, tests_total

def test_ode_solver():
    """Test ODE solver."""
    print("\n" + "=" * 60)
    print("ODE SOLVER TESTS")
    print("=" * 60)
    
    tests_passed = 0
    tests_total = 0
    
    # Test: dy/dt = y, y(0) = 1 => y(t) = e^t
    print("\n1. dy/dt = y, y(0) = 1 => y(1) = e")
    solver = TaylorODESolver(max_terms=16)
    # f(y) = y => coefficients [0, 1, 0, ...]
    f_coeffs = torch.zeros(16, dtype=torch.float32)
    f_coeffs[1] = 1.0  # y term
    
    y0 = torch.tensor(1.0)
    t, y = solver.solve(f_coeffs, y0, (0.0, 1.0), num_steps=100)
    
    result = y[-1].item()
    expected = math.e
    error = abs(result - expected) / expected
    print(f"   Expected y(1): {expected:.6f}, Got: {result:.6f}")
    print(f"   Relative error: {error:.2e}")
    tests_total += 1
    if error < 0.01:  # 1% tolerance
        tests_passed += 1
        print("   ✓ PASS")
    else:
        print("   ✗ FAIL")
    
    # Test: dy/dt = -y, y(0) = 1 => y(t) = e^(-t)
    print("\n2. dy/dt = -y, y(0) = 1 => y(1) = 1/e")
    f_coeffs = torch.zeros(16, dtype=torch.float32)
    f_coeffs[1] = -1.0  # -y term
    
    t, y = solver.solve(f_coeffs, y0, (0.0, 1.0), num_steps=100)
    
    result = y[-1].item()
    expected = 1/math.e
    error = abs(result - expected) / expected
    print(f"   Expected y(1): {expected:.6f}, Got: {result:.6f}")
    print(f"   Relative error: {error:.2e}")
    tests_total += 1
    if error < 0.01:
        tests_passed += 1
        print("   ✓ PASS")
    else:
        print("   ✗ FAIL")
    
    print(f"\n{'='*60}")
    print(f"ODE Solver: {tests_passed}/{tests_total} tests passed")
    return tests_passed, tests_total

def test_simplify():
    """Test polynomial simplification."""
    print("\n" + "=" * 60)
    print("SIMPLIFICATION TESTS")
    print("=" * 60)
    
    model = NousModel()
    tests_passed = 0
    tests_total = 0
    
    # Test: Simplify coefficients with near-zero terms
    print("\n1. Simplify [1.0, 2.0, 1e-15, 0.0, 0.0]")
    coeffs = torch.tensor([1.0, 2.0, 1e-15, 0.0, 0.0] + [0.0]*27, dtype=torch.float32)
    simplified, degree = model.forward(coeffs, op='simplify')
    print(f"   Effective degree: {degree}")
    print(f"   Non-zero count before: {(torch.abs(coeffs) > 1e-12).sum().item()}")
    print(f"   Non-zero count after: {(torch.abs(simplified) > 1e-12).sum().item()}")
    tests_total += 1
    if degree == 1:  # Only constant and x term should remain
        tests_passed += 1
        print("   ✓ PASS")
    else:
        print("   ✗ FAIL")
    
    print(f"\n{'='*60}")
    print(f"Simplification: {tests_passed}/{tests_total} tests passed")
    return tests_passed, tests_total

def test_configurable_params():
    """Test configurable model parameters."""
    print("\n" + "=" * 60)
    print("CONFIGURABLE PARAMETERS TESTS")
    print("=" * 60)
    
    tests_passed = 0
    tests_total = 0
    
    # Test: Create model with custom parameters
    print("\n1. Create model with max_terms=64")
    model = NousModel(max_terms=64, solver_iterations=100, solver_tolerance=1e-12)
    print(f"   max_terms: {model.max_terms}")
    print(f"   solver_iterations: {model.solver_iterations}")
    print(f"   discovery shape: {model.discovery.shape}")
    tests_total += 1
    if model.max_terms == 64 and model.discovery.shape == (64, 64):
        tests_passed += 1
        print("   ✓ PASS")
    else:
        print("   ✗ FAIL")
    
    print(f"\n{'='*60}")
    print(f"Configurable Parameters: {tests_passed}/{tests_total} tests passed")
    return tests_passed, tests_total

def test_multi_step():
    """Test multi-step reasoning pipelines."""
    print("\n" + "=" * 60)
    print("MULTI-STEP REASONING TESTS")
    print("=" * 60)
    
    model = NousModel()
    tests_passed = 0
    tests_total = 0
    
    # Test: Find critical points of f(x) = x^2 - 4x + 4
    # f'(x) = 2x - 4, critical point at x = 2
    print("\n1. Find critical points of f(x) = x^2 - 4x + 4")
    # Coefficients (ascending): [4.0, -4.0, 1.0]
    coeffs = torch.tensor([[4.0, -4.0, 1.0] + [0.0]*29], dtype=torch.float64)
    roots = model.forward(coeffs, op='find_critical_points')
    
    # roots for linear 2x-4 should be [2.0, 0.0]
    result = roots[0, 0, 0].item()
    expected = 2.0
    error = abs(result - expected)
    print(f"   Expected critical point: {expected}, Got: {result:.6f}")
    tests_total += 1
    if error < 1e-10:
        tests_passed += 1
        print("   ✓ PASS")
    else:
        print("   ✗ FAIL")
    
    # Test: Symbolic Chain: integrate -> derivative (should be identity)
    print("\n2. Symbolic Chain: integrate -> derivative (Identity)")
    chain_ops = ['integrate', 'derivative']
    # Use only 16 terms to avoid truncation at 32 during integration
    x = torch.zeros(1, 32)
    x[0, :16] = torch.randn(1, 16)
    result = model.forward(x, op='chain', ops=chain_ops)
    # The integrate -> derivative should restore at least the terms that were there
    error = torch.mean(torch.abs(x[0, :16] - result[0, :16]))
    print(f"   Mean error (first 16 terms): {error:.2e}")
    tests_total += 1
    if error < 1e-7:
        tests_passed += 1
        print("   ✓ PASS")
    else:
        print("   ✗ FAIL")

    print(f"\n{'='*60}")
    print(f"Multi-step: {tests_passed}/{tests_total} tests passed")
    return tests_passed, tests_total

def test_geometry():
    """Test symbolic geometry operations."""
    print("\n" + "=" * 60)
    print("GEOMETRY TESTS")
    print("=" * 60)
    
    model = NousModel()
    tests_passed = 0
    tests_total = 0
    
    # Test 1: Distance point to circle
    print("\n1. Distance from (6,8) to Circle(0,0,5)")
    p = torch.tensor([6.0, 8.0])
    params = (0.0, 0.0, 5.0)
    dist = model.forward(p, op='geometry', sub_op='distance', params=params)
    expected = 5.0
    error = abs(dist.item() - expected)
    print(f"   Expected: {expected}, Got: {dist.item():.4f}")
    tests_total += 1
    if error < 1e-6:
        tests_passed += 1
        print("   ✓ PASS")
    else:
        print("   ✗ FAIL")
        
    # Test 2: Circle Intersection
    print("\n2. Intersection of Circle(0,0,5) and Circle(4,0,3)")
    c1 = (0.0, 0.0, 5.0)
    c2 = (4.0, 0.0, 3.0)
    pts = model.forward(None, op='geometry', sub_op='intersection', c1=c1, c2=c2)
    # Expected: (4,3) and (4,-3)
    p1, p2 = pts
    print(f"   Point 1: ({p1[0].item():.1f}, {p1[1].item():.1f})")
    print(f"   Point 2: ({p2[0].item():.1f}, {p2[1].item():.1f})")
    tests_total += 1
    # Check if (4,3) or (4,-3) is found
    found_4_3 = any(abs(p[0]-4)<1e-1 and abs(p[1]-3)<1e-1 for p in pts)
    found_4_n3 = any(abs(p[0]-4)<1e-1 and abs(p[1]+3)<1e-1 for p in pts)
    if found_4_3 and found_4_n3:
        tests_passed += 1
        print("   ✓ PASS")
    else:
        print("   ✗ FAIL")

    print(f"\n{'='*60}")
    print(f"Geometry: {tests_passed}/{tests_total} tests passed")
    return tests_passed, tests_total


def test_neural_interpreter():
    """Test the Neural Python Interpreter."""
    print("\n" + "=" * 60)
    print("NEURAL INTERPRETER TESTS")
    print("=" * 60)
    
    from nous.interpreter import NeuralInterpreter
    from nous.symbolic import ExprVar
    model = NousModel()
    interpreter = NeuralInterpreter(model)
    tests_passed = 0
    tests_total = 0
    
    # 1. Arithmetic & Scope
    print("\n1. Arithmetic & Scope: x=10; y=x+5; return y*2")
    code = "x=10; y=x+5; return y*2"
    res = interpreter.execute(code)
    val = model.expand(res)[0].item()
    print(f"   Result: {val} (Expected 30.0)")
    tests_total += 1
    if abs(val - 30.0) < 1e-9:
        tests_passed += 1
        print("   ✓ PASS")
    else:
        print("   ✗ FAIL")
        
    # 2. Loop Accumulation
    print("\n2. Loop: Sum 0..4")
    code = """
total = 0
for i in range(5):
    total = total + i
return total
"""
    res = interpreter.execute(code)
    val = model.expand(res)[0].item()
    print(f"   Result: {val} (Expected 10.0)")
    tests_total += 1
    if abs(val - 10.0) < 1e-9:
        tests_passed += 1
        print("   ✓ PASS")
    else:
        print("   ✗ FAIL")

    # 3. Function & Backprop
    print("\n3. Function & Backprop: f(x) = x^2 + 2x + 1 at x=2")
    code = """
def f(x):
    return x*x + 2*x + 1

return f(x_in)
"""
    x_var = ExprVar('x')
    res = interpreter.execute(code, inputs={'x_in': x_var})
    deriv = res.diff('x')
    val = model.expand(deriv, center=2.0)[0].item()
    print(f"   df/dx(2): {val} (Expected 6.0)")
    tests_total += 1
    if abs(val - 6.0) < 1e-9:
        tests_passed += 1
        print("   ✓ PASS")
    else:
        print("   ✗ FAIL")

    print(f"\n{'='*60}")
    print(f"Interpreter: {tests_passed}/{tests_total} tests passed")
    return tests_passed, tests_total

def test_interpreter_advanced():
    """Test advanced features: Tuple unpacking, f-strings, constants."""
    print("\n" + "=" * 60)
    print("INTERPRETER ADVANCED TESTS")
    print("=" * 60)
    
    from nous.interpreter import NeuralInterpreter
    model = NousModel()
    interpreter = NeuralInterpreter(model)
    tests_passed = 0
    tests_total = 0
    
    # 1. Tuple Unpacking & Strings
    print("\n1. Tuple Unpacking: a, b = 1, 2; return a+b")
    code = "a, b = 1, 2; return f'{a}+{b}={a+b}'"
    res = interpreter.execute(code)
    print(f"   Result: {res}")
    tests_total += 1
    # Native execution might use ints, creating "1+2=3"
    # Old interpreter used floats, creating "1.0+2.0=3.0"
    # Both are acceptable, but native uses Ints by default for 1, 2.
    if res == "1.0+2.0=3.0" or res == "1+2=3":
        tests_passed += 1
        print("   ✓ PASS")
    else:
        print("   ✗ FAIL")
        
    print(f"\n{'='*60}")
    print(f"Interpreter Advanced: {tests_passed}/{tests_total} tests passed")
    return tests_passed, tests_total

def test_interpreter_lists():
    """Test list operations and kwargs."""
    print("\n" + "=" * 60)
    print("INTERPRETER LISTS TESTS")
    print("=" * 60)
    
    from nous.interpreter import NeuralInterpreter
    model = NousModel()
    interpreter = NeuralInterpreter(model)
    tests_passed = 0
    tests_total = 0
    
    # 1. List Creation and Append
    print("\n1. List Operations: l=[1]; l.append(2); return l[1]")
    code = """
l = [1]
l.append(2)
return l[1]
"""
    res = interpreter.execute(code)
    # res should be ExprConst or int/float
    val = res.value if hasattr(res, 'value') else res
    print(f"   Result: {val} (Expected 2.0)")
    tests_total += 1
    if abs(float(val) - 2.0) < 1e-9:
        tests_passed += 1
        print("   ✓ PASS")
    else:
        print("   ✗ FAIL")
    
    # 2. Kwargs and Functions
    print("\n2. Kwargs: def f(a, b=1): return a+b; return f(a=2, b=3)")
    code = """
def f(a, b=1):
    return a + b
return f(a=2, b=3)
"""
    res = interpreter.execute(code)
    val = res.value if hasattr(res, 'value') else res
    print(f"   Result: {val} (Expected 5.0)")
    tests_total += 1
    if abs(float(val) - 5.0) < 1e-9:
        tests_passed += 1
        print("   ✓ PASS")
    else:
        print("   ✗ FAIL")

    print(f"\n{'='*60}")
    print(f"Interpreter Lists: {tests_passed}/{tests_total} tests passed")
    return tests_passed, tests_total

def test_interpreter_eager_math():
    """Test eager evaluation of math functions on constants."""
    print("\n" + "=" * 60)
    print("INTERPRETER EAGER MATH TESTS")
    print("=" * 60)
    
    from nous.interpreter import NeuralInterpreter
    model = NousModel()
    interpreter = NeuralInterpreter(model)
    tests_passed = 0
    tests_total = 0
    
    # 1. Eager exp(constant)
    print("\n1. Eager exp(1.0)")
    code = "return exp(1.0)"
    res = interpreter.execute(code)
    # res should be ExprConst
    val = float(res)
    expected = math.e
    print(f"   Result: {val} (Expected {expected:.4f})")
    tests_total += 1
    if abs(val - expected) < 1e-9:
        tests_passed += 1
        print("   ✓ PASS")
    else:
        print("   ✗ FAIL")
    
    print(f"\n{'='*60}")
    print(f"Interpreter Eager Math: {tests_passed}/{tests_total} tests passed")
    return tests_passed, tests_total

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("ENHANCED NOUS ENGINE VERIFICATION")
    print("=" * 60)
    
    total_passed = 0
    total_tests = 0
    
    # Run all test suites
    p, t = test_neural_interpreter()
    total_passed += p
    total_tests += t
    
    p, t = test_interpreter_advanced()
    total_passed += p
    total_tests += t

    p, t = test_interpreter_lists()
    total_passed += p
    total_tests += t
    
    p, t = test_interpreter_eager_math()
    total_passed += p
    total_tests += t

    p, t = test_algebra()
    total_passed += p
    total_tests += t
    
    p, t = test_transcendental()
    total_passed += p
    total_tests += t
    
    p, t = test_definite_integration()
    total_passed += p
    total_tests += t
    
    p, t = test_multivariate()
    total_passed += p
    total_tests += t
    
    p, t = test_ode_solver()
    total_passed += p
    total_tests += t
    
    p, t = test_simplify()
    total_passed += p
    total_tests += t
    
    p, t = test_multi_step()
    total_passed += p
    total_tests += t
    
    p, t = test_geometry()
    total_passed += p
    total_tests += t
    
    p, t = test_configurable_params()
    total_passed += p
    total_tests += t
    
    # Final summary
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Total: {total_passed}/{total_tests} tests passed")
    print(f"Success rate: {100*total_passed/total_tests:.1f}%")
    print("=" * 60)
