import argparse
import random
import torch
import torch.nn as nn
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from sympy.parsing.sympy_parser import (parse_expr, standard_transformations, implicit_multiplication_application)

parser = argparse.ArgumentParser('Continuously train SCMs on functions')
parser.add_argument('-f', '--function', type=str, dest='func', default='line', help='Type of function to learn')
parser.add_argument('-i', '--hidden_layers', type=int, dest='hidden_dim', default=12, help='Number of hidden recurrent nodes')
parser.add_argument('-p', '--parse_function', type=str, dest='func_string', default=None, help='Custom function to learn, will override built-in functions')
parser.add_argument('-r', '--range', nargs='+', type=int, dest='range', default=[0, 100], help='Range of data to generate')
args = parser.parse_args()

device = torch.device('cpu')

torch.pi = torch.acos(torch.zeros(1)).item() * 2

def scm(theta, alpha=1.0, k=1.0, omega=0.16): return alpha * theta + omega + (k/(2 * np.pi)) * np.sin(2 * np.pi * theta)

def tensor_scm(thetas, alphas, ks, omegas): return torch.mul(alphas, thetas) + omegas + torch.mul(torch.mul(ks, (1.0/(2.0 * torch.pi))), torch.sin(torch.mul(2.0 * torch.pi, thetas)))

class SCMNet(nn.Module):
    def __init__(self, input_size, output_size, reservoir_size, input_spread=1, output_spread=1):
        super(SCMNet, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
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
        thetas = torch.matmul(driven_state, self.reservoir)
        thetas = tensor_scm(thetas, self.alphas, self.ks, self.omegas)
        return torch.sum(thetas[self.outputs]), thetas

torch.manual_seed(0)
model = SCMNet(1, 1, 10, 2, 4)
reservoir0 = torch.rand(10)
crit = nn.MSELoss()
opti = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(1):
    running_loss =0.0
    for i in range(100):
        x = i/100.0
        y = (i + 1)/100.0
        opti.zero_grad()
        out, reservoir0 = model(x, reservoir0)
        loss = crit(out, torch.tensor(y))
        opti.step()
        print(loss.item())

print('Done')