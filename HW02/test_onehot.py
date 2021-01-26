import torch
"""
x = torch.arange(0,2)
y = torch.nn.functional.one_hot(x)
print(y[0])
print(y[1])
"""
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)