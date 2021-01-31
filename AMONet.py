import torch

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import MultiSCMNet as msn

from tqdm import trange
from statistics import mean

data = pd.read_csv('Data/amo_monthly.csv')
data = torch.tensor(data[['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']].values).flatten()
# print(data[:24])
# sys.exit()
# data = data/data.max(0, keepdim=True)[0]

torch.manual_seed(10)
model = torch.jit.script(msn.MultiSCMNet(1, 1, [6, 6, 6, 6]))
crit = torch.nn.MSELoss()
opti = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
roll = []

fig, axs = plt.subplots(1, 1)
plt.ion()

torch.autograd.set_detect_anomaly(True)

total_loss = 0.0
for repetition in range(1):
    plt.cla()
    x = data[0].reshape(1,1)
    y = data[1].reshape(1,1)
    for step in trange(data.shape[0] - 1):
        pred = model(x)
        roll.append(pred.item())
        loss = crit(pred.double(), y)
        total_loss += loss.item()
        opti.zero_grad()
        loss.backward(retain_graph=True)
        opti.step()

        plotpred = pred.detach().numpy()

        axs.plot(int(step), plotpred[0], 'b+')
        axs.plot(int(step), y[0], 'bo')

        if len(roll) > 12:
            roll = roll[1:]
        axs.plot(int(step), mean(roll), 'bx')

        axs.set_xlim(max(0, int(step) - 100), int(step))

        x = y
        y = data[int(step) + 1].reshape(1,1)
        
        plt.draw(); plt.pause(0.01)

plt.ioff()
plt.show()

