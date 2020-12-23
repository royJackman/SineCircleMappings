import argparse
import random
import torch
import torch.nn as nn
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from sympy.parsing.sympy_parser import (parse_expr, standard_transformations, implicit_multiplication_application)
from tqdm import trange

parser = argparse.ArgumentParser('Continuously train SCMs on functions')
parser.add_argument('-e', '--epochs', type=int, dest='epochs', default=1000, help='Number of epochs')
parser.add_argument('-f', '--function', type=str, dest='func', default='sine', help='Type of function to learn')
parser.add_argument('-i', '--input_spread', type=int, dest='ins', default=2, help='Number of input channels')
parser.add_argument('-l', '--hidden_layers', type=int, dest='hidden_dim', default=12, help='Number of hidden recurrent nodes')
parser.add_argument('-n', '--nodes', type=int, dest='nodes', default=6, help='Number of nodes in the reservoir')
parser.add_argument('-o', '--output_spread', type=int, dest='outs', default=4, help='Number of output channels')
parser.add_argument('-p', '--parse_function', type=str, dest='func_string', default=None, help='Custom function to learn, will override built-in functions')
parser.add_argument('-r', '--range', nargs='+', type=int, dest='range', default=[0, 100], help='Range of data to generate')
parser.add_argument('-w', '--window', type=int, dest='window', default=1, help='Window width')
args = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

torch.pi = torch.acos(torch.zeros(1)).item() * 2

def scm(theta, alpha=1.0, k=1.0, omega=0.16): return alpha * theta + omega + (k/(2 * np.pi)) * np.sin(2 * np.pi * theta)

def tensor_scm(thetas, alphas, ks, omegas): return torch.mul(alphas.clone(), thetas.clone()) + omegas.clone() + torch.mul(torch.mul(ks.clone(), (1.0/(2.0 * torch.pi))), torch.sin(torch.mul(2.0 * torch.pi, thetas.clone())))

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

class SCMNet(nn.Module):
    def __init__(self, input_size, output_size, reservoir_size, input_spread=1, output_spread=1):
        super(SCMNet, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.reservoir_size = reservoir_size
        self.input_spread = input_spread
        self.output_spread = output_spread

        self.inputs = random.sample(range(reservoir_size), self.input_spread)
        self.outputs = random.sample(range(reservoir_size), self.output_spread)

        self.driver = torch.zeros(reservoir_size)
        self.driver[self.inputs] = 1.0

        self.alphas = nn.Parameter(torch.rand(reservoir_size))
        self.ks = nn.Parameter(torch.rand(reservoir_size))
        self.omegas = nn.Parameter(torch.rand(reservoir_size))
        self.reservoir = torch.from_numpy(nx.to_numpy_array(nx.generators.random_graphs.watts_strogatz_graph(reservoir_size, 4, 0.5))).float()
    
    def forward(self, x, reservoir_state):
        driven_state = reservoir_state + x * self.driver
        dsmax = driven_state.max()
        dsmin = driven_state.min()
        driven_state = torch.div(torch.sub(driven_state.clone(), torch.ones(self.reservoir_size), alpha=dsmin.item()), dsmax.item() - dsmin.item())
        thetas = torch.matmul(driven_state, self.reservoir)
        thetas = tensor_scm(thetas.clone(), self.alphas, self.ks, self.omegas)
        return torch.sum(thetas.clone()[0, :, self.outputs], 1), thetas.clone()

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
    plt.draw(); plt.pause(0.05)
    if step % 50 == 0:
        print(total_loss/(step + 1))

plt.ioff()
plt.show()
print('Alphas:', model.alphas.detach().numpy(), '\nKs:', model.ks.detach().numpy(), '\nOmegas:', model.omegas.detach().numpy(), '\nReservoir:', reservoir0.detach().numpy(), f'\nAvg Loss: {total_loss/args.epochs}')