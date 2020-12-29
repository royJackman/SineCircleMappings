import argparse
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

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
parser.add_argument('-w', '--window', type=int, dest='window', default=1, help='Window width')
args = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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
model = torch.jit.script(SCMNet(1, 1, args.nodes, args.ins, args.outs)).to(device)
reservoir0 = torch.rand(args.nodes).to(device)
crit = nn.MSELoss()
opti = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
data = generate_data(args.range[0], args.range[1], args.window * args.epochs)

plt.figure(1)
plt.title(f'SCM Model, {args.nodes} node reservoir, {args.ins} input channel{"" if args.ins == 1 else "s"}, {args.outs} output channel{"" if args.outs == 1 else "s"}')
plt.ion()

torch.autograd.set_detect_anomaly(True)

total_loss = 0.0
for step in trange(args.epochs):
    start, end = step * args.window, (step + 1) * args.window
    x_np = data[0][start:end]
    y_np = data[1][start:end]
    x = torch.from_numpy(x_np[np.newaxis, :, np.newaxis]).float().to(device)
    y = torch.from_numpy(y_np[np.newaxis, :, np.newaxis]).to(device)

    pred, reservoir0 = model(x, reservoir0.clone())
    loss = crit(pred.double(), y.flatten())
    total_loss += loss.item()
    opti.zero_grad()
    loss.backward(retain_graph=True)
    opti.step()
    
    steps = [*range(start, end)]
    if len(steps) == 1:
        plt.plot(steps[0], y.item(), 'ro', label='Target')
        plt.plot(steps[0], pred.item(), 'bo', label='Prediction')
    else:
        plt.plot(steps, y_np.flatten(), 'r-', label='Target')
        plt.plot(steps, pred.data.numpy(), 'b-', label='Prediction')

    plt.draw(); plt.pause(0.02)

plt.ioff()
plt.show()

print('Alphas:   ', model.alphas.detach().numpy(), 
      '\nKs:       ', model.ks.detach().numpy(), 
      '\nOmegas:   ', model.omegas.detach().numpy(), 
      '\nReservoir:', reservoir0.detach().flatten().numpy(), 
      f'\nAvg Loss:  {total_loss/args.epochs}')