import argparse
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from sympy.parsing.sympy_parser import (parse_expr, standard_transformations, implicit_multiplication_application)
from tqdm import trange
from SCMNet import SCMNet
from MultiSCMNet import MultiSCMNet

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

def tensorHenonMap(inp, alpha=1.4, beta=0.3): 
    item = inp.detach().numpy()
    return torch.tensor([-1 * alpha * item[0] * item[0] + item[1] + 1, beta * item[0]]).double()

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
# model = torch.jit.script(SCMNet(2, 2, args.nodes, args.ins, args.outs)).to(device)
# model = torch.jit.script(MultiSCMNet(2, 2, [6, 6, 6]))
from HarmonicNN import MultilayerHarmonicNN
model = torch.jit.script(MultilayerHarmonicNN(1, 1, [2]))
reservoir0 = torch.rand(2, 6).to(device)
crit = nn.MSELoss()
# opti = torch.optim.SGD(model.parameters(), lr=0.03, momentum=0.9)
opti = torch.optim.Adam(model.parameters(), lr=0.1)

fig, axs = plt.subplots(1, 2)
axs[0].set_xlim(-2, 2)
axs[0].set_ylim(-2, 2)
axs[0].set_title('Prediction')
axs[1].set_xlim(-2, 2)
axs[1].set_ylim(-2, 2)
axs[1].set_title('Actual')
plt.suptitle(f'SCM Model, {args.nodes} node reservoir, {args.ins} input channel{"" if args.ins == 1 else "s"}, {args.outs} output channel{"" if args.outs == 1 else "s"}')
plt.ion()

torch.autograd.set_detect_anomaly(True)

x = torch.tensor([0.0, 0.0])
y = tensorHenonMap(x)

prediction_past = []
actual_past = []

total_loss = 0.0
for step in trange(args.epochs):
    # pred, reservoir0 = model(x.clone(), reservoir0.clone())
    pred = model(x[:, np.newaxis])
    loss = crit(pred.double().flatten(), y.flatten())
    total_loss += loss.item()
    opti.zero_grad()
    loss.backward(retain_graph=True)
    opti.step()

    plotpred = pred.detach().numpy()
    prediction_past.append(axs[0].plot(plotpred[0], plotpred[1], 'bo'))
    actual_past.append(axs[1].plot(y[0], y[1], 'ro'))

    if len(prediction_past) > args.window or len(actual_past) > args.window:
        prediction_past[0][0].remove()
        prediction_past = prediction_past[1:]
        actual_past[0][0].remove()
        actual_past = actual_past[1:]

    x = y
    y = tensorHenonMap(x)

    plt.draw(); plt.pause(0.02)

plt.ioff()
plt.show()

print('Alphas:   ', model.alphas.detach().numpy(), 
      '\nKs:       ', model.ks.detach().numpy(), 
      '\nOmegas:   ', model.omegas.detach().numpy(), 
      '\nReservoir:', reservoir0.detach().flatten().numpy(), 
      f'\nAvg Loss:  {total_loss/args.epochs}')