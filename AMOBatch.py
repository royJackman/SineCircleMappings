import argparse
import torch

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import MultiSCMNet as msn
import HarmonicNN as hnn

from tqdm import trange
from statistics import mean
from TestRNN import testRNN

nodes = [64]
linear_nodes = [32, 16]
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
data = pd.read_csv('Data/amo_monthly.csv', delim_whitespace=True)
data = torch.tensor(data[['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']].values).flatten()
sp = np.fft.fft(data.clone().detach().cpu().numpy())
data = data[:, np.newaxis].to(device)
t = np.arange(len(data))
freq = np.fft.fftfreq(t.shape[-1])

middle = len(sp) // 2
empty = np.zeros(sp.shape)
node_half = (sum(nodes) + sum(linear_nodes)) // 2
empty[:node_half] = sp[:node_half]
empty[-node_half:] = sp[-node_half:]
ifft = np.fft.ifft(empty)
fft_score = sum((data.detach().cpu().numpy().flatten() - ifft.real) ** 2)
print('FFT score:', fft_score)

parser = argparse.ArgumentParser('Learn AMO data with with HNN')
parser.add_argument('-e', '--epochs', type=int, dest='epochs', default=2000, help='Number of epochs')
parser.add_argument('-n', '--nodes', type=int, dest='nodes', default=3, help='Number of internal nodes')
args = parser.parse_args()

torch.manual_seed(2)
model = hnn.MultilayerHarmonicNN(1,1,nodes, linear_nodes).to(device).double()
# model = msn.MultiSCMNet(1, 1, nodes).to(device).double()
print(f'Model Params: {sum([len(p.flatten()) for p in model.parameters()])}')
crit = torch.nn.MSELoss()
opti = torch.optim.Adam(model.parameters(), lr=1e-2)

torch.autograd.set_detect_anomaly(True)
losses = []
inp = torch.arange(1, 1981)[:, np.newaxis].to(device)
guess = torch.mul(torch.mean(data), torch.ones((1980, 1)).to(device))
hiscore = crit(data, guess).item()
print(f'Hi score to beat: {hiscore}')
broken = False

for r in trange(args.epochs):
    opti.zero_grad()
    pred = model(inp.double())
    loss = crit(pred, data)
    losses.append(loss.item())
    if not broken and loss.item() < fft_score:
        print(f'FFT score broken on epoch {int(r) + 1} with loss {loss.item()}')
        broken = True

    if not (int(r) + 1) % 200:
        print(f'Epoch {int(r)+1} MSE: {loss.item()}')
    
    loss.backward(retain_graph=True)
    opti.step()

print(np.mean(np.array(losses[-50:])))

plt.plot(losses)
plt.yscale("log")
plt.figure()

vals = [i for i in model(inp.double())]
plt.plot(data.detach().cpu().numpy())
plt.plot(vals)
plt.plot(guess.detach().cpu().numpy())
plt.plot(ifft)
plt.show()