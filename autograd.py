import torch

# x = torch.randn(3, requires_grad=True)
# print(x)

# y = x+2
# print(y)

# z = y*y*2
# # z = z.mean()
# print(z)

# z.backward()
# print(x.grad)

##dummy training example

weights = torch.ones(4, requires_grad=True)

for epoch in range(2):
    model_output = (weights*3).sum()
    # print(model_output)

    model_output.backward()

    print(weights.grad)
    weights.grad.zero_()
    print("")