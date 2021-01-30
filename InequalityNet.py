import torch

import pandas as pd
import matplotlib.pyplot as plt
import MultiSCMNet as msn

from tqdm import trange

data = pd.read_csv('Data/WID/WID.csv', delimiter=';', header=1)
data = torch.tensor(data[['USAb50', 'USAt1']].values)

torch.manual_seed(0)
model = torch.jit.script(msn.MultiSCMNet(2, 2, [6, 6, 6, 6]))
crit = torch.nn.MSELoss()
opti = torch.optim.SGD(model.parameters(), lr=0.03, momentum=0.9)

fig, axs = plt.subplots(1, 1)
plt.ion()

torch.autograd.set_detect_anomaly(True)

total_loss = 0.0
for repetition in range(5):
    plt.cla()
    x = data[0]
    y = data[1]
    for step in trange(data.shape[0] - 1):
        pred = model(x)
        loss = crit(pred.double(), y)
        total_loss += loss.item()
        opti.zero_grad()
        loss.backward(retain_graph=True)
        opti.step()

        plotpred = pred.detach().numpy()

        axs.plot(int(step), plotpred[0], 'bo')
        axs.plot(int(step), plotpred[1], 'go')
        axs.plot(int(step), y[0], 'ko')
        axs.plot(int(step), y[1], 'ro')
        axs.set_ylim(0.1, 0.3)

        x = y
        y = data[int(step) + 1]
        
        plt.draw(); plt.pause(0.01)

plt.ioff()
plt.show()

