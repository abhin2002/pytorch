import torch
import numpy as np

# x = torch.rand(5,3)
# print(x)
# print(x[0,0].item())

# print(torch.cuda.is_available())

#numpy and tensor in cpu

# a = torch.ones(5)
# print(a)
# b = a.numpy()
# print(b)

# a.add_(1)
# print(a)
# print(b)

# in gpu
print(torch.cuda.is_available())

device = torch.device("cuda")
x = torch.ones(5, device= device)
y = torch.ones(5)
y = y.to(device)

z = x + y
z = z.to("cpu")

z.numpy()

print(x)
print(y)
print(z)
