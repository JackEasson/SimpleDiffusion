import torch

x = torch.tensor(1)
print(x.size())
y = x.reshape(1, )
print(y.size())