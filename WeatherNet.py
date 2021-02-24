import torch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from HarmonicNN import MultilayerHarmonicNN
from tqdm import trange

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

data = pd.read_csv('Data/amo_monthly.csv')
data = torch.tensor(data[months].values)

train_count = 140

torch.manual_seed(0)
torch.autograd.set_detect_anomaly(True)
models = {}
crits = {}
optis = {}
for month in months:
    models[month] = torch.jit.script(MultilayerHarmonicNN(1, 1))
    crits[month] = torch.nn.MSELoss()
    optis[month] = torch.optim.Adam(models[month].parameters(), lr=0.01)
# model = torch.jit.script(MultilayerHarmonicNN(1, 12, [4, 4]))

# crit = torch.nn.MSELoss()
# opti = torch.optim.Adam(model.parameters(), lr=0.01)

total_loss = 0.0

plt.figure(1)
plt.get_current_fig_manager().window.state('zoomed')
plt.ion()

for i in range(100):
    round_loss = 0.0
    for step in range(train_count):
        for j, month in enumerate(months):
            pred = models[month](torch.tensor(step).reshape((1,1)))
            loss = crits[month](pred, data[step,j].reshape((1,1)))
            round_loss += loss.item()/12.0
            optis[month].zero_grad()
            loss.backward(retain_graph=True)
            optis[month].step()
        # pred = model(torch.tensor(step).reshape((1,1)))
        # loss = crit(pred, data[step][np.newaxis, :])
        # round_loss += abs(loss.item())
        # opti.zero_grad()
        # loss.backward(retain_graph=True)
        # opti.step()
    test_loss = 0.0
    for step in range(data.shape[0]-train_count):
        example_loss = 0.0
        for j, month in enumerate(months):
            pred = models[month](torch.tensor(step + train_count).reshape((1,1)))
            example_loss += abs(crits[month](pred, data[step + train_count, j]).item())/12.0
        # pred = model(torch.tensor(step + train_count).reshape((1,1)))
        # test_loss += abs(crit(pred, data[step + train_count]).item())
        test_loss += example_loss
    if (i+1)%10 == 0:
        print(f'Round {i} avg training loss {round_loss/float(train_count)}')
        print(f'Rount {i} avg test loss {test_loss/float(data.shape[0]-train_count)}\n------------')

print(f'{models["Jan"](torch.tensor(150).reshape((1,1)))}\n{data[150,0]}')