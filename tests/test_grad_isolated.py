import torch
from nous.symbolic import ExprConst

x = torch.tensor(3.0, requires_grad=True)
node = ExprConst(x)
taylor = node.to_taylor(center=0.0, max_terms=4, hilbert=None)
print(f"Taylor: {taylor}")
print(f"Requires grad: {taylor.requires_grad}")
taylor[0].backward()
print(f"Grad: {x.grad}")
