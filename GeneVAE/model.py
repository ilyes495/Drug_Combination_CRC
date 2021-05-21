import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import torch.nn.functional as F


class Nonlinearity(torch.nn.Module):
    def __init__(self):
        super(Nonlinearity, self).__init__()

    def forward(self, x):
        return F.leaky_relu(x)

class Net(nn.Module):

    def __init__(self, input_size):
        super(Net, self).__init__()
        k = 1024
        # k = 20000
        self.h1 = nn.Linear(input_size, k, bias=False)
        self.activation = Nonlinearity()
        self.out = nn.Linear(k, input_size, bias=False)        

    def forward(self, x):
        z = self.activation(self.h1(x))
        out = self.out(z)
        return out, z