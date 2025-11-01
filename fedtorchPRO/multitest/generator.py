def fab(max):
    n, a, b = 0, 0, 1
    while n < max:
        yield b
        a, b = b, a + b
        n = n + 1

# print(fab(100))

# for i in fab(100):
#     print(i)

import torch

# net = torch.nn.ModuleList([
#     torch.nn.Linear(5,10) for i in range(10)
# ])

# for pm in net.parameters(True) :
#     print(pm.shape)

net = torch.nn.Linear(5,10)

for pm in net.parameters(True) :
    print(pm.grad)

adam = torch.optim.Adam(net.parameters(),0.01)


x = torch.randn((10,5))
adam.zero_grad()
Fi = torch.nn.functional.mse_loss(net(x),torch.randn((10,10)))
Fi.backward()
adam.step()

for pm in net.parameters(True) :
    # pm = pm - 0.01 * pm.grad
    pm.add_(pm.grad,alpha=0.01)
