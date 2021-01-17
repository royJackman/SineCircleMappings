import networkx as nx
import random
import torch
import torch.nn as nn

torch.pi = torch.acos(torch.zeros(1)).item() * 2

# def scm(theta, alpha=1.0, k=1.0, omega=0.16): return alpha * theta + omega + (k/(2 * np.pi)) * np.sin(2 * np.pi * theta)

def tensor_scm(thetas, alphas, ks, omegas): return torch.mul(alphas.clone(), thetas.clone()) + omegas.clone() + torch.mul(torch.mul(ks.clone(), (1.0/(2.0 * torch.pi))), torch.sin(torch.mul(2.0 * torch.pi, thetas.clone())))

class SCMLayer(nn.Module):
    def __init__(self, reservoir_size, prev_size, transition=None):
        super(SCMLayer, self).__init__()
        if transition is None:
            self.transition = torch.ones(prev_size, reservoir_size)
        else:
            self.transition = transition
        
        self.reservoir_size = reservoir_size
        self.alphas = nn.Parameter(torch.rand(reservoir_size))
        self.ks = nn.Parameter(torch.rand(reservoir_size))
        self.omegas = nn.Parameter(torch.rand(reservoir_size))
        self.reservoir = torch.from_numpy(nx.to_numpy_array(nx.generators.random_graphs.watts_strogatz_graph(reservoir_size, 3, 0.5))).double()

    def forward(self, x, state):
        driven_state = state + torch.matmul(x.double(), self.transition.double())
        dsmax = driven_state.max()
        dsmin = driven_state.min()
        driven_state = torch.div(torch.sub(driven_state.clone(), torch.ones(self.reservoir_size), alpha=dsmin.item()), dsmax.item() - dsmin.item())
        thetas = torch.matmul(driven_state.double(), self.reservoir.double())
        thetas = tensor_scm(thetas.clone(), self.alphas, self.ks, self.omegas)
        return thetas

class MultiSCMNet(nn.Module):
    def __init__(self, input_size, output_size, reservoir_sizes, input_spread=1, output_spread=1):
        super(MultiSCMNet, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.reservoir_sizes = reservoir_sizes
        self.input_spread = input_spread
        self.output_spread = output_spread
        self.layers = nn.ModuleList([])
        self.states = []

        self.inputs = []
        for i in range(self.input_size):
            self.inputs.append(random.sample(range(reservoir_sizes[0]), self.input_spread))
        
        self.outputs = []
        for o in range(self.output_size):
            self.outputs.append(random.sample(range(reservoir_sizes[-1]), self.output_spread))

        self.driver = torch.zeros(self.input_size, self.reservoir_sizes[0])
        for i, v in enumerate(self.inputs):
            self.driver[i, v] = 1.0
        
        self.mask = torch.zeros(self.reservoir_sizes[-1], self.output_size).double()
        for o, v in enumerate(self.outputs):
            self.mask[v, o] = 1.0

        self.layers.append(SCMLayer(self.reservoir_sizes[0], 1, self.driver))
        self.states.append(torch.rand(self.reservoir_sizes[0]))

        for i, r in enumerate(self.reservoir_sizes[1:]):
            self.layers.append(SCMLayer(r, reservoir_sizes[i]))
            self.states.append(torch.rand(r))
    
    def forward(self, x):
        for i, l in enumerate(self.layers):
            x = l(x, self.states[i])
            self.states[i] = x.clone()
        return torch.matmul(x, self.mask)