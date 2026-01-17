import torch
import unittest
import math
from nous.engine import NousModel
from nous.interpreter import NeuralInterpreter
from nous.symbolic import ExprConst, SymbolicNode

class TestV3DSL(unittest.TestCase):
    def setUp(self):
        self.model = NousModel()
        self.interpreter = NeuralInterpreter(self.model)

    def _get_val(self, node):
        if isinstance(node, (int, float)):
            return float(node)
        if isinstance(node, ExprConst):
            return float(node.value)
        # Evaluate using Taylor at 0
        coeffs = node.to_taylor(center=0.0, max_terms=1, hilbert=self.model.hilbert)
        return float(coeffs[0].item())

    def test_soft_map(self):
        code = """
def double(x): return x * 2
items = [ExprConst(1), ExprConst(2), ExprConst(3)]
return soft_map(double, items)
"""
        result = self.interpreter.execute(code)
        vals = [self._get_val(x) for x in result]
        self.assertEqual(vals, [2.0, 4.0, 6.0])

    def test_soft_filter(self):
        code = """
def positive(x): return x  # logit: positive if x > 0
items = [ExprConst(-1.0), ExprConst(1.0), ExprConst(-2.0), ExprConst(2.0)]
return soft_filter(positive, items)
"""
        result = self.interpreter.execute(code)
        vals = [self._get_val(x) for x in result]
        # x=1.0, prob = sigmoid(1.0) approx 0.731. res approx 0.731
        # x=-1.0, prob = sigmoid(-1.0) approx 0.269. res approx -0.269
        self.assertLess(vals[0], 0)
        self.assertGreater(vals[1], 0)
        expected_v1 = 1.0 * (1 / (1 + math.exp(-1.0)))
        self.assertAlmostEqual(vals[1], expected_v1, places=3)

    def test_soft_sort(self):
        code = "return soft_sort([ExprConst(3), ExprConst(1), ExprConst(2)])"
        result = self.interpreter.execute(code)
        vals = [self._get_val(x) for x in result]
        # Soft sort should roughly order them
        self.assertLess(vals[0], vals[1])
        self.assertLess(vals[1], vals[2])
        # Check values are roughly correct (Sinkhorn/Attention might not be exact)
        self.assertAlmostEqual(vals[0], 1.0, delta=0.5)
        self.assertAlmostEqual(vals[2], 3.0, delta=0.5)

    def test_soft_enumerate(self):
        code = "return soft_enumerate([ExprConst(10), ExprConst(20)])"
        result = self.interpreter.execute(code)
        # returns [(0, ExprConst(10)), (1, ExprConst(20))]
        self.assertEqual(result[0][0], 0)
        self.assertEqual(self._get_val(result[0][1]), 10.0)

    def test_comparisons(self):
        code = """
x = ExprConst(10.0)
y = ExprConst(20.0)
return [x < y, x > y, x == y]
"""
        result = self.interpreter.execute(code)
        # x < y -> returns 20 - 10 = 10 (high positive logit)
        # x > y -> returns 10 - 20 = -10 (high negative logit)
        # x == y -> returns -abs(10-20) = -10 (high negative logit)
        logits = [self._get_val(x) for x in result]
        self.assertGreater(logits[0], 0)
        self.assertLess(logits[1], 0)
        self.assertLess(logits[2], 0)

if __name__ == "__main__":
    unittest.main()
