import networkx as nx
import numpy as np
import random
import torch
import torch.nn as nn

torch.pi = torch.acos(torch.zeros(1)).item() * 2

# def scm(theta, alpha=1.0, k=1.0, omega=0.16): return alpha * theta + omega + (k/(2 * np.pi)) * np.sin(2 * np.pi * theta)

def tensor_scm(thetas, alphas, ks, omegas): return torch.mul(alphas.clone(), thetas.clone()) + omegas.clone() + torch.mul(torch.mul(ks.clone(), (1.0/(2.0 * torch.pi))), torch.sin(torch.mul(2.0 * torch.pi, thetas.clone())))

class MultilayerSCMNet(nn.Module):
    def __init__(self, input_size, output_size, reservoir_sizes, input_spread, output_spread):
        super(MultilayerSCMNet, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.reservoir_sizes = reservoir_sizes
        self.input_spread = input_spread
        self.output_spread = output_spread

        self.inputs = []
        for i in range(self.input_size):
            self.inputs.append(random.sample(range(reservoir_sizes[0]), self.input_spread))
        
        self.outputs = []
        for o in range(self.output_size):
            self.outputs.append(random.sample(range(reservoir_sizes[-1]), self.output_spread))

        self.driver = torch.zeros(self.input_size, self.reservoir_sizes[0])
        for i, v in enumerate(self.inputs):
            self.driver[i, v] = 1.0

        self.transitions = [self.driver]
        for i, r in enumerate(reservoir_sizes[1:]):
            self.transitions.append(torch.ones(reservoir_sizes[i], r))
        
        self.mask = torch.zeros(self.reservoir_sizes[-1], self.output_size).double()
        for o, v in enumerate(self.outputs):
            self.mask[v, o] = 1.0
        
        self.alphas = nn.ParameterList([])
        self.ks = nn.ParameterList([])
        self.omegas = nn.ParameterList([])
        self.reservoirs = []
        self.states = []

        for r in reservoir_sizes:
            self.alphas.append(nn.Parameter(torch.rand(r), requires_grad=True))
            self.ks.append(nn.Parameter(torch.rand(r), requires_grad=True))
            self.omegas.append(nn.Parameter(torch.rand(r), requires_grad=True))
            self.reservoirs.append(torch.from_numpy(nx.to_numpy_array(nx.generators.watts_strogatz_graph(r, 4, 0.5))))
            self.states.append(torch.rand(r))

    def forward(self, x):
        for i, r in enumerate(self.reservoirs):
            x = torch.matmul(x.clone().double(), self.transitions[i].double())
            x = torch.add(x.clone(), self.states[i])
            mx = x.max()
            mn = x.min()
            x = torch.div(torch.sub(x.clone(), torch.ones(self.reservoir_sizes[i]), alpha=mn.item()), mx.item() - mn.item())
            x = torch.matmul(x.clone().double(), r.double())
            self.states[i] = x.clone()
        return torch.matmul(x, self.mask)
