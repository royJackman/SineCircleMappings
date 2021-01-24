import argparse
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import odeint
from sympy.parsing.sympy_parser import (parse_expr, standard_transformations, implicit_multiplication_application)
from tqdm import trange
from SCMNet import SCMNet

parser = argparse.ArgumentParser('Continuously train SCMs on functions')
parser.add_argument('-e', '--epochs', type=int, dest='epochs', default=250, help='Number of epochs')
parser.add_argument('-f', '--function', type=str, dest='func', default='sine', help='Type of function to learn')
parser.add_argument('-i', '--input_spread', type=int, dest='ins', default=2, help='Number of input channels')
parser.add_argument('-n', '--nodes', type=int, dest='nodes', default=6, help='Number of nodes in the reservoir')
parser.add_argument('-o', '--output_spread', type=int, dest='outs', default=4, help='Number of output channels')
parser.add_argument('-p', '--parse_function', type=str, dest='func_string', default=None, help='Custom function to learn, will override built-in functions')
parser.add_argument('-r', '--range', nargs='+', type=int, dest='range', default=[0, 100], help='Range of data to generate')
parser.add_argument('-w', '--window', type=int, dest='window', default=50, help='Window length into the past')
args = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def lorenz(state, t, sigma=10.0, beta=8.0/3.0, rho=28.0):
    x, y, z = state
    return sigma * (y - x), x * (rho - z) - y, x * y - beta * z

state0 = [1.0, 1.0, 1.0]
t = np.arange(0.0, 100.1, 0.1)
states = odeint(lorenz, state0, t)

def tensorLorenzMap(inp, sigma=10.0, beta=8.0/3.0, p=28.0):
    item = inp.detach().numpy()
    return torch.tensor([
        item[0] + sigma * (item[1] - item[0]), 
        item[1] + item[0]*(p - item[2]) - item[1], 
        item[2] + item[0]*item[1] - beta * item[2]
    ]).double()

def generate_data(start, end, points):
    full = np.linspace(start, end, points+1)
    if args.func_string is None:
        if args.func == 'sine':
            full = np.sin(full)
    else:
        exp = parse_expr(args.func_string, transformations=(standard_transformations + (implicit_multiplication_application,)))
        test = [exp.evalf(subs={'x': i}) for i in full]
        full = np.asarray([exp.evalf(subs={'x': i}) for i in full]).astype('float')
    return full[:-1], full[1:]

torch.manual_seed(0)
# model = torch.jit.script(SCMNet(3, 3, args.nodes, args.ins, args.outs)).to(device)
from MultiSCMNet import MultiSCMNet
model = torch.jit.script(MultiSCMNet(3, 3, [6, 6])).to(device)
reservoir0 = torch.rand(args.nodes).to(device)
crit = nn.MSELoss()
opti = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

fig = plt.figure()
axs = []
axs.append(fig.add_subplot(121, projection='3d'))
axs.append(fig.add_subplot(122, projection='3d'))
axs[0].set_title('Prediction')
axs[1].set_title('Actual')
plt.suptitle(f'SCM Model, {args.nodes} node reservoir, {args.ins} input channel{"" if args.ins == 1 else "s"}, {args.outs} output channel{"" if args.outs == 1 else "s"}')
plt.ion()

torch.autograd.set_detect_anomaly(True)

x = torch.tensor(states[0])
y = torch.tensor(states[1])

prediction_past = []
actual_past = []

total_loss = 0.0
for step in trange(999):
    # pred, reservoir0 = model(x.double(), reservoir0.clone())
    pred = model(x)
    loss = crit(pred.double(), y.flatten())
    total_loss += loss.item()
    opti.zero_grad()
    loss.backward(retain_graph=True)
    opti.step()

    plotpred = pred.detach().numpy()
    prediction_past.append(axs[0].plot(plotpred[0], plotpred[1], plotpred[2], 'bo'))
    actual_past.append(axs[1].plot(y[0], y[1], y[2], 'ro'))

    if len(prediction_past) > args.window or len(actual_past) > args.window:
        prediction_past[0][0].remove()
        prediction_past = prediction_past[1:]
        actual_past[0][0].remove()
        actual_past = actual_past[1:]

    x = y
    y = torch.tensor(states[step + 2])

    axs[0].set_xlim(*axs[1].get_xlim())
    axs[0].set_ylim(*axs[1].get_ylim())
    axs[0].set_zlim(*axs[1].get_zlim())

    plt.draw(); plt.pause(0.02)

plt.ioff()
plt.show()

print('Alphas:   ', model.alphas.detach().numpy(), 
      '\nKs:       ', model.ks.detach().numpy(), 
      '\nOmegas:   ', model.omegas.detach().numpy(), 
      '\nReservoir:', reservoir0.detach().flatten().numpy(), 
      f'\nAvg Loss:  {total_loss/args.epochs}')