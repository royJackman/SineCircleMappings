import torch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from HarmonicNN import MultilayerHarmonicNN
from MixedReservoirNet import MultiMix
from tqdm import trange
from Models import MixedNet

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

data = pd.read_csv('Data/amo_monthly.csv')
data = torch.tensor(data[months].values)

train_count = 120

torch.manual_seed(0)
torch.autograd.set_detect_anomaly(True)
models = {}
crits = {}
optis = {}
for month in months:
    # models[month] = torch.jit.script(MultilayerHarmonicNN(1, 1, [6, 6, 6, 6]))
    # models[month] = torch.jit.script(MultiMix(1, 1, [20], [20]).double())
    models[month] = MixedNet(1, 1, [32, 32], [[24, 8], [16, 16]]).double()
    crits[month] = torch.nn.MSELoss()
    optis[month] = torch.optim.Adam(models[month].parameters(), lr=0.03)

total_loss = 0.0

fig, axs = plt.subplots(3, 4) 
plt.get_current_fig_manager().window.state('zoomed')
plt.ion()

X = torch.range(0, train_count)[:, np.newaxis].double()
test_X = torch.range(train_count+1, len(data))[:, np.newaxis].double()
for i in range(250):
    round_loss = 0.0
    predictions = []
    for j, month in enumerate(months):
        optis[month].zero_grad()
        pred = models[month](X)
        predictions.append(pred.clone().detach().numpy())
        loss = crits[month](pred, data[:train_count,j])
        round_loss += loss.item()/12.0
        loss.backward(retain_graph=True)
        optis[month].step()

    test_loss = 0.0
    test_predictions = []
    for j, month in enumerate(months):
        pred = models[month](test_X)
        test_predictions.append(pred.clone().detach().numpy())
        loss = crits[month](pred.flatten(), data[train_count:, j].flatten())
        test_loss += loss.item()/12.0
    if (i+1)%10 == 0:
        print(f'Round {i} avg training loss {round_loss/float(train_count)}')
        print(f'Rount {i} avg test loss {test_loss/float(data.shape[0]-train_count)}\n------------')
        
    for j in range(3):
        for k in range(4):
            idx = 4 * j + k
            axs[j][k].cla()
            axs[j][k].set_title(f'{months[idx]}')
            axs[j][k].plot(data[:, idx])
            axs[j][k].plot(X, predictions[idx])
            axs[j][k].plot(test_X, test_predictions[idx])
    fig.suptitle(f'Round {i+1} training loss: {round_loss} testing loss: {test_loss}')
    plt.gcf().canvas.draw()
    plt.gcf().canvas.flush_events()

plt.ioff()
plt.show()

print(f'{models["Jan"](torch.tensor(150).reshape((1,1)))}\n{data[150,0]}')