import torch
import torch.nn as nn
import numpy as np

def scm(theta, alpha=1.0, k=1.0, omega=0.16): return alpha * theta + omega + (k/(2 * np.pi)) * np.sin(2 * np.pi * theta)

class SCMNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 128, 3)
        self.linear = nn.Linear(256, 2)
    
    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 256)
        return self.linear(x)

torch.manual_seed(0)
model = SCMNet()
inp = np.array([[[[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]]]])
x = torch.tensor(inp, dtype=torch.float)