import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

device = torch.device('cpu')

def scm(theta, alpha=1.0, k=1.0, omega=0.16): return alpha * theta + omega + (k/(2 * np.pi)) * np.sin(2 * np.pi * theta)

class SCMNet(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(SCMNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_size * 3)
        self.params = []
    
    def forward(self, x, hidden):
        theta_n = x
        x, hidden = self.rnn(x, hidden)
        
        outs = []
        for time_step in range(x.size(1)):
            self.params = self.linear(x[:, time_step, :]).flatten()
            outs.append(scm(theta_n, self.params[0], self.params[1], self.params[2]))
        return torch.stack(outs, dim=1), hidden

    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device)
        return hidden

torch.manual_seed(0)
model = SCMNet(1, 1, 12, 1)
model = model.to(device)

# steps = np.linspace(0, 2 * np.pi, 100, dtype=np.float32)
# x_np = np.sin(steps)
# y_np = np.cos(steps)
# plt.plot(steps, x_np, 'b-', label='input')
# plt.plot(steps, y_np, 'r-', label='target')
# plt.legend(loc='best')
# plt.show()

hidden = None
epochs = 270
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

plt.figure(1, figsize=(12, 5))
plt.ion()

for step in range(100):
    start, end = step * np.pi, (step+1)*np.pi
    steps = np.linspace(start, end, 10, dtype=np.float32, endpoint=False)
    x_np = np.sin(steps)
    y_np = np.cos(steps)
    x = torch.from_numpy(x_np[np.newaxis, :, np.newaxis])
    y = torch.from_numpy(y_np[np.newaxis, :, np.newaxis])

    prediction, hidden = model(x, hidden)
    hidden = hidden.data

    loss = criterion(prediction, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    plt.plot(steps, y_np.flatten(), 'r-')
    plt.plot(steps, np.mean(prediction.data.numpy()[0, :, :, 0], axis=0), 'b-')
    plt.draw(); plt.pause(0.05)

plt.ioff()
plt.show()