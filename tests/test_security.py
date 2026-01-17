import unittest
import torch
from nous.engine import NousModel
from nous.interpreter import NeuralInterpreter, CodeSafetyError

class TestSecurity(unittest.TestCase):
    def setUp(self):
        self.model = NousModel()
        self.interpreter = NeuralInterpreter(self.model)

    def test_import_blocking(self):
        code = "import os; return os.name"
        with self.assertRaises(CodeSafetyError) as cm:
            self.interpreter.execute(code)
        self.assertIn("Forbidden syntax", str(cm.exception))

    def test_import_from_blocking(self):
        code = "from os import path; return path.sep"
        with self.assertRaises(CodeSafetyError):
            self.interpreter.execute(code)

    def test_eval_blocking(self):
        code = "return eval('1+1')"
        with self.assertRaises(CodeSafetyError) as cm:
            self.interpreter.execute(code)
        self.assertIn("forbidden built-in 'eval'", str(cm.exception))

    def test_open_blocking(self):
        code = "f = open('test.txt', 'w'); f.write('secret')"
        with self.assertRaises(CodeSafetyError):
            self.interpreter.execute(code)

    def test_private_attribute_blocking(self):
        # Attempt to access internals
        code = "return interpreter._base_ctx"
        with self.assertRaises(CodeSafetyError) as cm:
            self.interpreter.execute(code, {'interpreter': self.interpreter})
        self.assertIn("private attribute", str(cm.exception))

    def test_class_definition_blocking(self):
        code = "class Malicious: pass; return 1"
        with self.assertRaises(CodeSafetyError):
            self.interpreter.execute(code)

    def test_with_blocking(self):
        code = "with open('t.txt') as f: pass"
        with self.assertRaises(CodeSafetyError):
            self.interpreter.execute(code)

    def test_introspection_exploit_blocking(self):
        # Classic escape: ().__class__.__base__.__subclasses__()
        code = "return ().__class__.__base__"
        with self.assertRaises(CodeSafetyError):
            self.interpreter.execute(code)

    def test_globals_blocking(self):
        code = "return globals()"
        with self.assertRaises(CodeSafetyError):
            self.interpreter.execute(code)

    def test_legitimate_code_works(self):
        code = "x = 10; y = 20; return x + y"
        result = self.interpreter.execute(code)
        self.assertEqual(result.value, 30.0)

    def test_soft_logic_works(self):
        code = "return soft_if(1.0, 10.0, 20.0)"
        result = self.interpreter.execute(code)
        # 1.0 is prob ~0.73, res = 0.73*10 + 0.27*20 = 7.3 + 5.4 = 12.7
        # sigmoid(1.0) = 1 / (1 + e^-1) = 0.731058
        # 0.731058 * 10 + (1-0.731058) * 20 = 7.31058 + 5.37884 = 12.68942
        self.assertAlmostEqual(result.value, 12.689414, places=5)

if __name__ == "__main__":
    unittest.main()
