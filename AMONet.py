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

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

data = pd.read_csv('Data/amo_monthly.csv')
data = torch.tensor(data[['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']].values).flatten()

parser = argparse.ArgumentParser('Compare Harmonic NN to RNN of equal volume')
parser.add_argument('-n', '--nodes', type=int, dest='nodes', default=3, help='Number of internal nodes')
args = parser.parse_args()

torch.manual_seed(1)
# model = torch.jit.script(msn.MultiSCMNet(1, 1, [8, 8, 8]))
model = torch.jit.script(hnn.MultilayerHarmonicNN(1, 1, None))
print('Model params:', sum([len(p.flatten()) for p in model.parameters()]))
crit = torch.nn.MSELoss()
# opti = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
opti = torch.optim.Adam(model.parameters(), lr=0.01)

benchmark = testRNN(1, 1, args.nodes, 1)
hidden = torch.zeros(1, 1, args.nodes)
print('Benchmark params:', sum([len(p.flatten()) for p in benchmark.parameters()]))
bench_crit = torch.nn.MSELoss()
bench_opti = torch.optim.Adam(benchmark.parameters(), lr = 0.01)

roll = []
bench_roll = []
actual_roll = []

fig, axs = plt.subplots(1, 1)
plt.ion()

torch.autograd.set_detect_anomaly(True)

total_loss = 0.0
bench_total = 0.0
for repetition in range(1):
    plt.cla()
    x = data[0].reshape(1,1).float()
    y = data[1].reshape(1,1)
    for step in trange(data.shape[0] - 1):
        pred = model(x)
        bench_pred, hidden = benchmark(x[np.newaxis, :].float(), hidden.float())
        hidden = hidden.data

        roll.append(pred.item())
        bench_roll.append(bench_pred.item())
        actual_roll.append(y.item())

        loss = crit(pred.double(), y)
        bench_loss = bench_crit(bench_pred.double(), y.reshape([]))

        total_loss += loss.item()
        bench_total += bench_loss.item()

        opti.zero_grad()
        bench_opti.zero_grad()

        loss.backward(retain_graph=True)
        bench_loss.backward(retain_graph=True)

        opti.step()
        bench_opti.step()

        plotpred = pred.detach().numpy()
        plotbench = bench_pred.detach().numpy()

        axs.plot(int(step), plotpred[0], 'rx', label='Prediction')
        axs.plot(int(step), plotbench, 'bx', label='RNN Benchmark')
        axs.plot(int(step), y[0], 'gx', label='Actual')

        if len(roll) > 12:
            roll = roll[1:]
            bench_roll = bench_roll[1:]
            actual_roll = actual_roll[1:]

        axs.plot(int(step), mean(roll), 'ro', label='Pred Rolling Avg')
        axs.plot(int(step), mean(bench_roll), 'bo', label='RNN Rolling Avg')
        axs.plot(int(step), mean(actual_roll), 'go', label='Actual Rolling Avg')

        axs.set_xlim(max(-1, int(step) - 100), int(step))
        axs.set_ylim(-0.5, 0.5)

        x = y
        y = data[int(step) + 1].reshape(1,1)

        if int(step) == 0:
            axs.legend(loc='lower left')
        
        if (int(step) - 1) % 100 == 0:
            print('Average error HNN:', total_loss/int(step))
            print('Average error RNN:', bench_total/int(step))
        
        plt.draw(); plt.pause(0.001)

plt.ioff()
plt.show()

