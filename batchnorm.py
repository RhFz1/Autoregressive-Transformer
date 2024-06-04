import torch
import torch.nn as nn

class BatchNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.gamma = torch.ones((1, ))
        self.beta = torch.zeros((1, ))
        self.running_mean = torch.tensor()
        self.running_std = torch.tensor()
    def forward(self, x):
        mean = torch.mean(x, 0, keepdim=True)
        variance = torch.std(x, 0, keepdim=True)
        x = (x - mean) / variance
        with torch.no_grad():
            self.running_mean = 0.999 * self.running_mean + 0.001 * mean
            self.running_std = 0.999 * self.running_std + 0.001 * variance
        x = self.gamma * x + self.beta

        return x
