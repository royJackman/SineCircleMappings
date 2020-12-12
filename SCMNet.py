import torch
import torch.nn as nn
import numpy as np

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
    
    def forward(self, x):
        batch_size = x.size(0)
        hidden = self.init_hidden(batch_size)
        theta_n = x
        x, hidden = self.rnn(x, hidden)
        x = x.contiguous().view(-1, self.hidden_dim)
        x = self.linear(x)
        self.params = x
        return scm(theta_n, x[0][0], x[0][1], x[0][2]), hidden
        # return x, hidden
    
    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device)
        return hidden

torch.manual_seed(0)
model = SCMNet(1, 1, 12, 1)
model = model.to(device)
inp = np.array([[[0], [1], [2], [3], [4], [5], [6], [7]]])
# outp = np.array([[[1], [2], [3], [4], [5], [6], [7], [8]]])
outp = np.array([[8]])
x = torch.tensor(inp, dtype=torch.float)
y = torch.tensor(outp, dtype=torch.float)

steps = np.linspace(0, 2 * np.pi, 300, dtype=np.float32)
x_np = np.sin(steps)
y_np = np.cos(steps)
plt.plot(steps, x_np, 'b-', label='input')
plt.plot(steps, y_np, 'r-', label='target')
plt.legend(loc='best')
plt.show()

epochs = 270
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
x = x.to(device)
for e in range(1, epochs + 1):
    optimizer.zero_grad()
    output, hidden = model(x)
    output = output.to(device)
    y = y.to(device)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()

    if e%15 == 0:
        print('Epoch: {}/{}.............'.format(e, epochs), end=' ')
        print("Loss: {:.6f}".format(loss.item()))
