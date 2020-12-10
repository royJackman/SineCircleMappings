import torch
import torch.nn as nn
import numpy as np

def scm(theta, alpha=1.0, k=1.0, omega=0.16): return alpha * theta + omega + (k/(2 * np.pi)) * np.sin(2 * np.pi * theta)

class SCMNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(1, 20, 2)
        self.linear = nn.Linear(20, 3)
        self.params = []
    
    def forward(self, x, hidden=torch.zeros(2, 1, 20).to(torch.device('cpu'))):
        theta_n = x
        x, hidden = self.rnn(x)
        x = x.view(-1, 20)
        x = self.linear(x)
        return scm(theta_n, x[0][0], x[0][1], x[0][2]), hidden

torch.manual_seed(0)
model = SCMNet()
inp = np.array([[[0]]])
x = torch.tensor(inp, dtype=torch.float)
out, hidden = model(x)
print(out)