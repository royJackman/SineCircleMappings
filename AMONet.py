import argparse
import torch

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import MultiSCMNet as msn
import HarmonicNN as hnn

from tqdm import trange
from statistics import mean
from TestRNN import testRNN
from TestNN import testNN

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

data = pd.read_csv('Data/amo_monthly.csv')
data = torch.tensor(data[['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']].values).flatten()

parser = argparse.ArgumentParser('Compare Harmonic NN to RNN of equal volume')
parser.add_argument('-n', '--nodes', type=int, dest='nodes', default=3, help='Number of internal nodes')
args = parser.parse_args()

torch.manual_seed(2)
# model = torch.jit.script(msn.MultiSCMNet(1, 1, [8, 8, 8]))
model = torch.jit.script(hnn.MultilayerHarmonicNN(1, 1, None))
print('HNN params:', sum([len(p.flatten()) for p in model.parameters()]))
crit = torch.nn.MSELoss()
# opti = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
opti = torch.optim.Adam(model.parameters(), lr=0.01)

RNN = testRNN(1, 1, args.nodes, 1)
hidden = torch.zeros(1, 1, args.nodes)
print('RNN params:', sum([len(p.flatten()) for p in RNN.parameters()]))
RNN_crit = torch.nn.MSELoss()
RNN_opti = torch.optim.Adam(RNN.parameters(), lr = 0.01)

NN = testNN(1, 1, args.nodes)
print('NN params:', sum([len(p.flatten()) for p in NN.parameters()]))
NN_crit = torch.nn.MSELoss()
NN_opti = torch.optim.Adam(NN.parameters(), lr = 0.01)

roll = []
RNN_roll = []
NN_roll = []
actual_roll = []

fig, axs = plt.subplots(2, 1)
plt.ion()

torch.autograd.set_detect_anomaly(True)

total_loss = 0.0
RNN_total = 0.0
NN_total = 0.0
for repetition in range(1):
    plt.cla()
    x = data[0].reshape(1,1).float()
    y = data[1].reshape(1,1)
    for step in trange(data.shape[0] - 1):
        pred = model(x)
        RNN_pred, hidden = RNN(x[np.newaxis, :].float(), hidden.float())
        hidden = hidden.data
        NN_pred = NN(x.float())

        roll.append(pred.item())
        RNN_roll.append(RNN_pred.item())
        NN_roll.append(NN_pred.item())
        actual_roll.append(y.item())

        loss = crit(pred.double(), y)
        RNN_loss = RNN_crit(RNN_pred.double(), y.reshape([]))
        NN_loss = NN_crit(NN_pred.double(), y)

        if int(step) > 100:
            total_loss += loss.item()
            RNN_total += RNN_loss.item()
            NN_total += NN_loss.item()

        opti.zero_grad()
        RNN_opti.zero_grad()
        NN_opti.zero_grad()

        loss.backward(retain_graph=True)
        RNN_loss.backward(retain_graph=True)
        NN_loss.backward(retain_graph=True)

        opti.step()
        RNN_opti.step()
        NN_opti.step()

        plotpred = pred.detach().numpy()
        plotRNN = RNN_pred.detach().numpy()
        plotNN = NN_pred.detach().numpy()
        
        axs[0].set_title('Atlantic Multi-Decadal Oscillation')

        axs[0].plot(int(step), plotpred[0], 'rx', label='Prediction')
        axs[0].plot(int(step), plotRNN, 'bx', label='RNN Benchmark')
        axs[0].plot(int(step), plotNN, 'kx', label='NN Benchmark')
        axs[0].plot(int(step), y[0], 'gx', label='Actual')

        axs[0].set_xlim(max(-1, int(step) - 100), int(step))
        axs[0].set_ylim(-0.5, 0.5)

        if len(roll) > 12:
            roll = roll[1:]
            RNN_roll = RNN_roll[1:]
            NN_roll = NN_roll[1:]
            actual_roll = actual_roll[1:]

        axs[1].plot(int(step), mean(roll), 'ro', label='Pred Rolling Avg')
        axs[1].plot(int(step), mean(RNN_roll), 'bo', label='RNN Rolling Avg')
        axs[1].plot(int(step), mean(NN_roll), 'ko', label='NN Rolling Avg')
        axs[1].plot(int(step), mean(actual_roll), 'go', label='Actual Rolling Avg')

        axs[1].set_xlim(max(-1, int(step) - 100), int(step))
        axs[1].set_ylim(-0.25, 0.25)

        x = y
        y = data[int(step) + 1].reshape(1,1)

        if int(step) == 0:
            axs[0].legend(loc='lower left')
            axs[1].legend(loc='lower left')
        
        if (int(step) - 1) % 100 == 0:
            print('Average error HNN:', total_loss/int(step))
            print('Average error RNN:', RNN_total/int(step))
            print('Average error NN:', NN_total/int(step))
        
        plt.draw(); plt.pause(0.001)

plt.ioff()
plt.show()

