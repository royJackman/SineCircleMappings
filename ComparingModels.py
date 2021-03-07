import torch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Models import SCMNet, MixedNet, SCMReservoir
from TestNN import testNN
from TestRNN import testRNN
from tqdm import trange

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

data = pd.read_csv('Data/amo_monthly.csv')
data = torch.tensor(data[months].values)
flat = data.flatten()

torch.manual_seed(0)
torch.autograd.set_detect_anomaly(True)
models = {}
crits = {}
optis = {}

names = ['SCMNet', 'MixedNet', 'TestNN']

models['SCMNet'] = torch.jit.script(SCMNet(1, 1, [8, 8]).double())
models['MixedNet'] = torch.jit.script(MixedNet(1, 1, [8, 8]).double())
models['SCMReservoir'] = torch.jit.script(SCMReservoir(1, 1, [8, 8]).double())
models['TestRNN'] = testRNN(1, 1, 8, 2).double()
models['TestNN'] = testNN(1, 1, 8).double()

# names.append('SCMReservoir')
# names.append('TestRNN')
# for n in names:
#     print(n, sum([len(p) for p in models[n].parameters()]))

# sys.exit()

loss_data = {'SCMNet': [], 'MixedNet': [], 'TestNN': []}

for n in names:
    crits[n] = torch.nn.MSELoss()
    optis[n] = torch.optim.Adam(models[n].parameters(), lr=0.01) if n != 'SCMReservoir' else torch.optim.SGD(models[n].parameters(), lr=0.01, momentum=0.9)

# fig, axs = plt.subplots(1, 1) 
# plt.get_current_fig_manager().window.state('zoomed')
# plt.ion()

hidden = torch.zeros((2, 1, 8))
colors = {'SCMNet': 'b', 'MixedNet': 'r', 'SCMReservoir': 'g', 'TestRNN': 'y', 'TestNN': 'c'}
losses = {'SCMNet': 0.0, 'MixedNet': 0.0, 'SCMReservoir': 0.0, 'TestRNN': 0.0, 'TestNN': 0.0}
data_points = float(len(flat))
for step, datum in enumerate(flat):
    predictions = {}
    for n in names:
        if n != 'TestRNN':
            predictions[n] = models[n](torch.tensor(step).reshape((1,1)).double())
        else:
            predictions[n], hidden = models[n](torch.tensor(step).reshape((1,1,1)).double(), hidden.clone().double())
        loss = crits[n](predictions[n], datum)
        losses[n] += loss.item()/data_points
        loss_data[n].append(loss.item())
        optis[n].zero_grad()
        loss.backward(retain_graph=True)
        optis[n].step()
        # axs.plot(step, predictions[n].item(), colors[n] + 'o', label=n)

    # axs.plot(step, datum, 'gx', label='Actual')
    # axs.set_title('Atlantic Multi-Decadal Oscillation')
    # axs.set_xlim(max(-1, step - 100), step)
    # axs.set_ylim(-0.25, 0.25)

    # if step == 0:
    #     axs.legend(loc='lower left')
 
    if (step + 1) % 100 == 0:
        print(f'Step {step + 1} model Losses: {losses}')
    
    # plt.draw(); plt.pause(0.02)

# plt.ioff()
plt.show()
plt.clf()

for n in names:
    plt.plot(loss_data[n], label=n)
# plt.plot(loss_data['MixedNet'], '#ff7f0e')
plt.legend(loc='upper left')
plt.suptitle('Loss vs Month of AMO Data')
plt.xlabel('Months')
plt.ylabel('MSE Loss')
plt.show()