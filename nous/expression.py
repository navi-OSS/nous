"""
Native Symbolic DSL for Nous.
Parses mathematical expressions using Python's 'ast' module and 
converts them directly to Nous Taylor coefficients without SymPy.
"""
import ast
import torch
import math
from engine import NousModel
from symbolic import ExprVar, ExprConst, ExprAdd, ExprSub, ExprMul, ExprDiv, ExprPow, ExprFunc

class ExpressionParser:
    """
    Parses string-based math expressions into Symbolic DAG nodes.
    Example: "x**2 - 2*x + 1" -> ExprSub(ExprAdd(ExprPow(x, 2), ...), ...)
    """
    def __init__(self, model):
        self.model = model
        self.max_terms = model.max_terms
        
    def parse(self, expr_str, var_name='x'):
        """
        Main entry point for parsing an expression string.
        """
        tree = ast.parse(expr_str, mode='eval')
        return self._evaluate_node(tree.body, var_name)

    def _evaluate_node(self, node, var_name):
        # 1. Constants
        if isinstance(node, ast.Constant):
            return ExprConst(float(node.value))

        # 2. Variable (x)
        if isinstance(node, ast.Name):
            if node.id == var_name:
                return ExprVar(node.id)
            elif node.id == 'pi':
                return ExprConst(math.pi)
            elif node.id == 'e':
                return ExprConst(math.e)
            raise ValueError(f"Unknown variable: {node.id}")

        # 3. Binary Operations (+, -, *, /, **)
        if isinstance(node, ast.BinOp):
            left = self._evaluate_node(node.left, var_name)
            right = self._evaluate_node(node.right, var_name)
            
            if isinstance(node.op, ast.Add): return ExprAdd(left, right)
            if isinstance(node.op, ast.Sub): return ExprSub(left, right)
            if isinstance(node.op, ast.Mult): return ExprMul(left, right)
            if isinstance(node.op, ast.Div): return ExprDiv(left, right)
            if isinstance(node.op, ast.Pow): return ExprPow(left, right)

        # 4. Unary Operations (-x)
        if isinstance(node, ast.UnaryOp):
            operand = self._evaluate_node(node.operand, var_name)
            if isinstance(node.op, ast.USub):
                return ExprMul(ExprConst(-1.0), operand)
            return operand

        # 5. Functions (exp(x), sin(x), etc.)
        if isinstance(node, ast.Call):
            func_name = node.func.id
            arg = self._evaluate_node(node.args[0], var_name)
            return ExprFunc(func_name, arg)
            
        raise ValueError(f"Unsupported expression component: {type(node)}")
            
        raise ValueError(f"Unsupported expression component: {type(node)}")

def install_dsl(model):
    """Monkey-patch 'parse' method into NousModel for convenience."""
    parser = ExpressionParser(model)
    model.parse = parser.parse
    return model

if __name__ == "__main__":
    print("Zero-Friction DSL Module initialized.")
