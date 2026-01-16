"""
Nous Visual Debugger - Symbolic DAG Visualization.

Provides utilities for visualizing and debugging symbolic computation graphs.
"""
from .symbolic import SymbolicNode, ExprConst, ExprVar, ExprAdd, ExprSub, ExprMul, ExprDiv, ExprPow, ExprFunc


def graph_to_dot(node, name="graph"):
    """
    Export a symbolic DAG to GraphViz DOT format.
    
    Args:
        node: Root SymbolicNode
        name: Graph name
    Returns:
        String in DOT format
    """
    lines = [f"digraph {name} {{"]
    lines.append("  rankdir=TB;")
    lines.append("  node [shape=box, style=filled, fillcolor=lightblue];")
    
    visited = set()
    node_ids = {}
    counter = [0]
    
    def get_id(n):
        if id(n) not in node_ids:
            node_ids[id(n)] = f"n{counter[0]}"
            counter[0] += 1
        return node_ids[id(n)]
    
    def visit(n, parent_id=None, edge_label=None):
        nid = get_id(n)
        
        if id(n) not in visited:
            visited.add(id(n))
            
            # Node label
            if isinstance(n, ExprConst):
                label = f"{n.value:.4g}"
                color = "lightyellow"
            elif isinstance(n, ExprVar):
                label = n.name
                color = "lightgreen"
            elif isinstance(n, ExprAdd):
                label = "+"
                color = "lightblue"
            elif isinstance(n, ExprSub):
                label = "-"
                color = "lightblue"
            elif isinstance(n, ExprMul):
                label = "×"
                color = "lightblue"
            elif isinstance(n, ExprDiv):
                label = "÷"
                color = "lightblue"
            elif isinstance(n, ExprPow):
                label = "^"
                color = "lightblue"
            elif isinstance(n, ExprFunc):
                label = n.name
                color = "lightcoral"
            else:
                label = type(n).__name__
                color = "white"
            
            lines.append(f'  {nid} [label="{label}", fillcolor={color}];')
            
            # Recurse into children
            if hasattr(n, 'left') and hasattr(n, 'right'):
                visit(n.left, nid, "L")
                visit(n.right, nid, "R")
            elif hasattr(n, 'base') and hasattr(n, 'exp'):
                visit(n.base, nid, "base")
                visit(n.exp, nid, "exp")
            elif hasattr(n, 'arg'):
                visit(n.arg, nid, "arg")
        
        # Draw edge from parent
        if parent_id:
            edge_str = f'  {parent_id} -> {nid}'
            if edge_label:
                edge_str += f' [label="{edge_label}"]'
            edge_str += ";"
            lines.append(edge_str)
    
    visit(node)
    lines.append("}")
    return "\n".join(lines)


def export_to_file(node, filename="graph.dot"):
    """Export DAG to a DOT file."""
    dot_str = graph_to_dot(node)
    with open(filename, 'w') as f:
        f.write(dot_str)
    print(f"Exported to {filename}")
    return filename


def print_tree(node, indent=0):
    """Print a simple text representation of the DAG."""
    prefix = "  " * indent
    
    if isinstance(node, ExprConst):
        print(f"{prefix}Const({node.value})")
    elif isinstance(node, ExprVar):
        print(f"{prefix}Var({node.name})")
    elif isinstance(node, ExprFunc):
        print(f"{prefix}{node.name}(")
        print_tree(node.arg, indent + 1)
        print(f"{prefix})")
    elif hasattr(node, 'left') and hasattr(node, 'right'):
        op_name = type(node).__name__.replace("Expr", "")
        print(f"{prefix}{op_name}(")
        print_tree(node.left, indent + 1)
        print_tree(node.right, indent + 1)
        print(f"{prefix})")
    elif hasattr(node, 'base') and hasattr(node, 'exp'):
        print(f"{prefix}Pow(")
        print_tree(node.base, indent + 1)
        print_tree(node.exp, indent + 1)
        print(f"{prefix})")
    else:
        print(f"{prefix}{node}")


# Demo
if __name__ == "__main__":
    print("=== Visual Debugger Demo ===")
    
    # Build a simple expression: sin(x^2 + 1)
    x = ExprVar('x')
    expr = ExprFunc('sin', x * x + 1)
    
    print("\nText Tree:")
    print_tree(expr)
    
    print("\nDOT Format:")
    print(graph_to_dot(expr, "sin_example"))
