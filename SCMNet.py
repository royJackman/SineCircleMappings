import networkx as nx
import random
import torch
import torch.nn as nn

torch.pi = torch.acos(torch.zeros(1)).item() * 2

# def scm(theta, alpha=1.0, k=1.0, omega=0.16): return alpha * theta + omega + (k/(2 * np.pi)) * np.sin(2 * np.pi * theta)

def tensor_scm(thetas, alphas, ks, omegas): return torch.mul(alphas.clone(), thetas.clone()) + omegas.clone() + torch.mul(torch.mul(ks.clone(), (1.0/(2.0 * torch.pi))), torch.sin(torch.mul(2.0 * torch.pi, thetas.clone())))

class SCMNet(nn.Module):
    def __init__(self, input_size, output_size, reservoir_size, input_spread=1, output_spread=1):
        super(SCMNet, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.reservoir_size = reservoir_size
        self.input_spread = input_spread
        self.output_spread = output_spread

        self.inputs = []
        for i in range(self.input_size):
            self.inputs.append(random.sample(range(reservoir_size), self.input_spread))
        
        self.outputs = []
        for o in range(self.output_size):
            self.outputs.append(random.sample(range(reservoir_size), self.output_spread))

        self.driver = torch.zeros(self.input_size, self.reservoir_size)
        for i, v in enumerate(self.inputs):
            self.driver[i, v] = 1.0
        
        self.mask = torch.zeros(self.reservoir_size, self.output_size).double()
        for o, v in enumerate(self.outputs):
            self.mask[v, o] = 1.0

        self.alphas = nn.Parameter(torch.rand(reservoir_size))
        self.ks = nn.Parameter(torch.rand(reservoir_size))
        self.omegas = nn.Parameter(torch.rand(reservoir_size))
        self.reservoir = torch.from_numpy(nx.to_numpy_array(nx.generators.random_graphs.watts_strogatz_graph(reservoir_size, 4, 0.5))).float()
    
    def forward(self, x, reservoir_state):
        driven_state = reservoir_state + torch.matmul(x.double(), self.driver.double())
        dsmax = driven_state.max()
        dsmin = driven_state.min()
        driven_state = torch.div(torch.sub(driven_state.clone(), torch.ones(self.reservoir_size), alpha=dsmin.item()), dsmax.item() - dsmin.item())
        thetas = torch.matmul(driven_state.double(), self.reservoir.double())
        thetas = tensor_scm(thetas.clone(), self.alphas, self.ks, self.omegas)
        return torch.matmul(thetas.clone(), self.mask), thetas.clone()