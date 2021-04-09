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

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

data = pd.read_csv('Data/amo_monthly.csv', delim_whitespace=True)
data = torch.tensor(data[['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']].values).flatten()
data = data[:, np.newaxis]

parser = argparse.ArgumentParser('Learn AMO data with with HNN')
parser.add_argument('-e', '--epochs', type=int, dest='epochs', default=100, help='Number of epochs')
parser.add_argument('-n', '--nodes', type=int, dest='nodes', default=3, help='Number of internal nodes')
args = parser.parse_args()

torch.manual_seed(2)
model = hnn.MultilayerHarmonicNN(1,1,[32, 24, 16, 8])
print(f'HNN Params: {sum([len(p.flatten()) for p in model.parameters()])}')
crit = torch.nn.MSELoss()
opti = torch.optim.Adam(model.parameters(), lr=0.01)

torch.autograd.set_detect_anomaly(True)
losses = []
inp = torch.arange(1, 1981)[:, np.newaxis]
guess = torch.mul(torch.mean(data), torch.ones((1980, 1)))
hiscore = crit(data, guess).item()
print(f'Guess to beat: {hiscore}')
broken = False

for r in trange(args.epochs):
    opti.zero_grad()
    pred = model(inp)
    loss = crit(pred, data)
    losses.append(loss.item())
    if not broken and loss.item() < hiscore:
        print(f'Record broken on epoch {int(r) + 1}')
        broken = True
    loss.backward(retain_graph=True)
    opti.step()

print(np.mean(np.array(losses[-50:])))

plt.plot(losses[-1000])
plt.figure()

plt.plot(pred.detach().numpy())
plt.plot(data)
plt.plot(guess)
plt.show()