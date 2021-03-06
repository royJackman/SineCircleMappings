import torch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from HarmonicNN import MultilayerHarmonicNN
from MixedReservoirNet import MultiMix
from tqdm import trange

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

data = pd.read_csv('Data/amo_monthly.csv')
data = torch.tensor(data[months].values)

# plt.plot(data[:, 0])
# plt.show()
# sys.exit()

train_count = 120

torch.manual_seed(0)
torch.autograd.set_detect_anomaly(True)
models = {}
crits = {}
optis = {}
for month in months:
    # models[month] = torch.jit.script(MultilayerHarmonicNN(1, 1, [6, 6, 6, 6]))
    models[month] = torch.jit.script(MultiMix(1, 1, [7], [3]).double())
    crits[month] = torch.nn.MSELoss()
    optis[month] = torch.optim.Adam(models[month].parameters(), lr=0.01)

total_loss = 0.0

fig, axs = plt.subplots(3, 4) 
plt.get_current_fig_manager().window.state('zoomed')
plt.ion()


for i in range(1000):
    round_loss = 0.0
    predictions = np.zeros(data.shape)
    for step in range(train_count):
        for j, month in enumerate(months):
            pred = models[month](torch.tensor(step).reshape((1,1)))
            predictions[step, j] = pred.item()
            loss = crits[month](pred.reshape((1,1)), data[step,j].reshape((1,1)))
            round_loss += loss.item()/12.0
            optis[month].zero_grad()
            loss.backward(retain_graph=True)
            optis[month].step()

    test_loss = 0.0
    for step in range(data.shape[0]-train_count):
        example_loss = 0.0
        prediction = []
        for j, month in enumerate(months):
            pred = models[month](torch.tensor(step + train_count).reshape((1,1)))
            predictions[step + train_count, j] = pred.item()
            example_loss += abs(crits[month](pred.reshape((1,1)), data[step + train_count, j].reshape((1,1))).item())/12.0
            prediction.append(pred.item())
        test_loss += example_loss
    if (i+1)%10 == 0:
        print(f'Round {i} avg training loss {round_loss/float(train_count)}')
        print(f'Rount {i} avg test loss {test_loss/float(data.shape[0]-train_count)}\n------------')
        
    for j in range(3):
        for k in range(4):
            axs[j][k].cla()
            axs[j][k].set_title(f'{months[4*j + k]}')
            axs[j][k].plot(data[:, 4*j + k])
            axs[j][k].plot(predictions[:train_count, 4*j + k])
            axs[j][k].plot(list(range(train_count, data.shape[0])), predictions[train_count:, 4*j + k])
    fig.suptitle(f'Layers: {models["Dec"].reservoir_sizes}, Linears: {models["Dec"].linear_nodes},  Round {i+1}')
    plt.gcf().canvas.draw()
    plt.gcf().canvas.flush_events()

plt.ioff()
plt.show()

print(f'{models["Jan"](torch.tensor(150).reshape((1,1)))}\n{data[150,0]}')