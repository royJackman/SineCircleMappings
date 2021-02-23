import torch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from HarmonicNN import MultilayerHarmonicNN
from tqdm import trange

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

data = pd.read_csv('Data/BostonWeather.csv')
data = data.dropna(axis='columns')
dates = data[['DATE']]
data = torch.tensor(data[['TMAX', 'TMIN']].values)

training = 20000

torch.manual_seed(0)
torch.autograd.set_detect_anomaly(True)

model = torch.jit.script(MultilayerHarmonicNN(1, 2, [3, 4, 3]))
crit = torch.nn.MSELoss()
opti = torch.optim.Adam(model.parameters())

total_loss = 0.0
for i in trange(10):
    round_loss = 0.0
    for j in range(training):
        pred = model(torch.tensor(j).reshape((1,1)))
        loss = crit(pred, data[j][np.newaxis, :])
        round_loss += abs(loss.item())
        opti.zero_grad()
        loss.backward(retain_graph=True)
        opti.step()
    test_loss = 0.0
    for j in range(len(dates) - training):
        pred = model(torch.tensor(j + training).reshape((1,1)))
        test_loss += abs(crit(pred, data[j][np.newaxis, :]))
    print(f'Round {i} avg training loss {round_loss/float(training)}')
    print(f'Rount {i} avg test loss {test_loss/float(len(dates)-training)}\n------------')

rando = np.random.randint(len(dates))
print(f'Example {rando}: {model(torch.tensor(rando).reshape((1,1)))}\n{data[rando]}')