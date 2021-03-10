import json
import torch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from colour import Color
from MixedNet import MixedNet
from mpl_toolkits.mplot3d import Axes3D

torch.manual_seed(0)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']

data = pd.read_csv('Data/heart.csv')
data = torch.tensor(data[cols].values).double()

target = data[:, -1:]
data = data[:, :-1]
plot_data = [[], [], []]
best = []

examples = data.shape[0]
epochs = 300
colors = 1000
layer_width = 16
eighty = int(0.8 * examples)
twenty = examples - eighty

performance_statistics = {}
architectures = []
for i in range(0, layer_width):
    rem = layer_width - i
    for j in range(0, rem):
        mod = rem - j
        architectures.append([[i, j, layer_width - (i+j)], [i, j, layer_width - (i+j)], [layer_width, layer_width]])

for i, a in enumerate(architectures):
    res = a.pop()
    print(f'Model {i + 1}: {len(res)} layer(s) with size(s) {a}')
    # for j, l in enumerate(a):
    #     print(f'Layer {j+1}: {l[0]} sin, {l[1]} tanh, {l[2]} log')
    model = MixedNet(13, 1, res, a).double()
    crit = torch.nn.MSELoss()
    opti = torch.optim.Adam(model.parameters(), lr = 0.01)

    arch_loss = 0.0
    min_arch = 1000.0
    arch_test = 0.0
    for e in range(epochs):
        i = 0
        batch = 0
        epoch_loss = 0.0

        while i < eighty:
            opti.zero_grad()
            batch += 1
            last = i
            i = min(last + 64, eighty)
            pred = model(data[last:i, :].double())
            loss = crit(pred, target[last:i].double())
            epoch_loss += loss.item()
            loss.backward(retain_graph=True)
            opti.step()
        arch_loss += epoch_loss/float(epochs)
        
        pred = model(data[eighty:].double())
        loss = crit(pred, target[eighty:].double())
        if loss.item() < min_arch:
            min_arch = loss.item()
        arch_test += loss.item()/float(epochs)
    plot_data[0].append(a[0][0])
    plot_data[1].append(a[0][1])
    plot_data[2].append(a[0][2])
    best.append(min_arch)
    performance_statistics[f'Layer {a[0][0]} sin, {a[0][1]} tanh, {a[0][2]} log'] = min_arch
    # print(f'Avg train MSE: {arch_loss}, Avg test MSE: {arch_test}, Best test MSE: {min_arch}\n----------------------')

# with open('heartnet.json', 'w') as f:
#     json.dump(performance_statistics, f)

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
ax.set_zlabel('Log nodes')

# grad = fig.add_subplot(1, 4, 4)
# grad.imshow([[]])

plt.show()