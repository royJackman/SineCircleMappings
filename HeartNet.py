import torch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from MixedNet import MixedNet

torch.manual_seed(0)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']

data = pd.read_csv('Data/heart.csv')
data = torch.tensor(data[cols].values).double()

target = data[:, -1:]
print(target.shape)
data = data[:, :-1]

examples = data.shape[0]
epochs = 100
eighty = int(0.8 * examples)
twenty = examples - eighty

architectures = [
    [[15, 15, 0], [15, 15, 0], [30, 30]],
    [[25, 5, 0], [25, 5, 0], [30, 30]],
    [[0, 15, 15], [0, 15, 15], [30, 30]],
    [[0, 25, 5], [0, 25, 5], [30, 30]],
    [[15, 0, 15], [15, 0, 15], [30, 30]],
    [[25, 0, 5], [25, 0, 5], [30, 30]],
    [[15, 15, 0], [15, 15, 0], [30, 30]],
    [[25, 5, 0], [25, 5, 0], [30, 30]],
    [[10, 10, 10], [10, 10, 10], [30, 30]]
]

for i, a in enumerate(architectures):
    res = a.pop()
    print(f'Model {i + 1}: {len(res)} layer(s) with size(s) {res}')
    for j, l in enumerate(a):
        print(f'Layer {j+1}: {l[0]} sin, {l[1]} tanh, {l[2]} relu')
    model = MixedNet(13, 1, res, a).double()
    crit = torch.nn.MSELoss()
    opti = torch.optim.Adam(model.parameters(), lr = 0.01)

    arch_loss = 0.0
    min_arch = 1000.0
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
            if loss.item() < min_arch:
                min_arch = loss.item()
            loss.backward(retain_graph=True)
            opti.step()
        arch_loss += epoch_loss/float(epochs)
    print(f'Avg MSE: {arch_loss}, Best MSE: {min_arch}\n----------------------')
