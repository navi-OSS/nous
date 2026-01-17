import torch
x = torch.tensor(5.0, requires_grad=True)
z = torch.zeros(3)
c = torch.cat([x.unsqueeze(0), z])
print(f"Cat: {c}")
print(f"Requires grad: {c.requires_grad}")
print(f"Grad fn: {c.grad_fn}")
loss = c[0]**2
loss.backward()
print(f"Grad: {x.grad}")
