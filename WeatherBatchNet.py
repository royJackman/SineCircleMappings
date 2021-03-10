import torch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from colour import Color
from HarmonicNN import MultilayerHarmonicNN
from MixedReservoirNet import MultiMix
from tqdm import trange
# from Models import MixedNet
from MixedNet import MixedNet

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

data = pd.read_csv('Data/amo_monthly.csv')
data = torch.tensor(data[months].values)

train_count = 120

# torch.manual_seed(0)
# torch.autograd.set_detect_anomaly(True)
# models = {}
# crits = {}
# optis = {}
# for month in months:
#     # models[month] = torch.jit.script(MultilayerHarmonicNN(1, 1, [6, 6, 6, 6]))
#     # models[month] = torch.jit.script(MultiMix(1, 1, [20], [20]).double())
#     models[month] = MixedNet(1, 1, [64], [[64]]).double()
#     crits[month] = torch.nn.MSELoss()
#     optis[month] = torch.optim.Adam(models[month].parameters(), lr=0.03)

total_loss = 0.0
layer_width = 32
colors = 1000
architectures = []
for i in range(0, layer_width):
    rem = layer_width - i
    for j in range(0, rem):
        mod = rem - j
        architectures.append([[i, j, 0, 0, layer_width - (i+j)], [i, j, 0, 0, layer_width - (i+j)], [layer_width, layer_width]])

# fig, axs = plt.subplots(3, 4) 
# plt.get_current_fig_manager().window.state('zoomed')
# plt.ion()

plot_data = [[], [], []]
best = []

torch.manual_seed(0)
torch.autograd.set_detect_anomaly(True)
X = torch.arange(0, train_count)[:, np.newaxis].double()
test_X = torch.arange(train_count, len(data))[:, np.newaxis].double()
for a, arch in enumerate(architectures):
    res = arch.pop()
    models = {}
    crits = {}
    optis = {}
    for month in months:
        # models[month] = torch.jit.script(MultilayerHarmonicNN(1, 1, [6, 6, 6, 6]))
        # models[month] = torch.jit.script(MultiMix(1, 1, [20], [20]).double())
        models[month] = MixedNet(1, 1, res, arch).double()
        crits[month] = torch.nn.MSELoss()
        optis[month] = torch.optim.Adam(models[month].parameters(), lr=0.01)

    arch_train_loss = 0.0
    arch_test_loss = 0.0
    epochs = 100
    for i in range(epochs):
        round_loss = 0.0
        # predictions = []
        for j, month in enumerate(months):
            optis[month].zero_grad()
            pred = models[month](X)
            # predictions.append(pred.clone().detach().numpy())
            loss = crits[month](pred.flatten(), data[:train_count,j].flatten())
            round_loss += loss.item()/12.0
            loss.backward(retain_graph=True)
            optis[month].step()

        test_loss = 0.0
        # test_predictions = []
        for j, month in enumerate(months):
            pred = models[month](test_X)
            # test_predictions.append(pred.clone().detach().numpy())
            loss = crits[month](pred.flatten(), data[train_count:, j].flatten())
            test_loss += loss.item()/12.0
        
        arch_train_loss += round_loss
        arch_test_loss += test_loss

        if (i+1)%10 == 0:
            print(f'Round {i} avg training loss {round_loss}')
            print(f'Rount {i} avg test loss {test_loss}\n------------')
            
        # for j in range(3):
        #     for k in range(4):
        #         idx = 4 * j + k
        #         axs[j][k].cla()
        #         axs[j][k].set_title(f'{months[idx]}')
        #         axs[j][k].plot(data[:, idx])
        #         axs[j][k].plot(X, predictions[idx])
        #         axs[j][k].plot(test_X, test_predictions[idx])
        # fig.suptitle(f'Round {i+1} training loss: {round_loss} testing loss: {test_loss}')
        # plt.gcf().canvas.draw()
        # plt.gcf().canvas.flush_events()
        # plt.draw(); plt.pause(0.005)
    plot_data[0].append(arch[0][0])
    plot_data[1].append(arch[0][1])
    plot_data[2].append(arch[0][4])
    best.append(arch_test_loss)
    print(f'Architecture {arch} Training loss: {arch_train_loss /float(epochs)} Testing loss: {arch_test_loss /float(epochs)}')
    

# plt.ioff()
# plt.show()

gradient = list(Color('blue').range_to(Color('orange'), colors))
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

print(min(best), max(best))
mn = min(best)
best = [b - mn for b in best]
mx = max(best)
best = [int((b*colors)/mx) for b in best]
best = [gradient[max(0, min(b, colors-1))].hex for b in best]

ax.scatter(*plot_data, c=best)
# ax.suptitle(f'Lowest loss after 300 epochs in network with 2 layers with [30, 30] nodes')
ax.set_xlabel('Sin nodes')
ax.set_ylabel('Tanh nodes')
ax.set_zlabel('Sigmoid nodes')

# grad = fig.add_subplot(1, 4, 4)
# grad.imshow([[]])

plt.show()