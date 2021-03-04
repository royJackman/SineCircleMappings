import networkx as nx
import numpy as np
import torch

torch.pi = torch.acos(torch.zeros(1)).item() * 2

class MixedLayer(torch.nn.Module):
    def __init__(self, reservoir_size, prev_size, transition=None, linear_nodes=0):
        super(MixedLayer, self).__init__()
        if transition is None:
            self.transition = torch.rand(prev_size, reservoir_size)
        else:
            self.transition = transition
        
        self.reservoir_size = reservoir_size
        self.linear_nodes = linear_nodes
        self.kappas = torch.nn.Parameter(torch.rand(reservoir_size))
        self.gammas = torch.nn.Parameter(torch.rand(reservoir_size))
        self.deltas = torch.nn.Parameter(torch.rand(reservoir_size))
        self.reservoir = torch.from_numpy(nx.to_numpy_array(nx.generators.random_graphs.watts_strogatz_graph(reservoir_size, 3, 0.5))).double()
    
    def forward(self, x):
        driven_state = torch.matmul(x.double(), self.transition.double())
        driven_state = torch.matmul(driven_state, self.reservoir)
        # driven_state = torch.add(torch.mul(self.gammas, driven_state), self.deltas)

        if len(driven_state.shape) == 1:
            driven_state = driven_state[np.newaxis, :]

        if self.linear_nodes <= 0:
            driven_state = torch.sin(driven_state)
        else:
            driven_state = torch.cat((torch.sin(driven_state[0, :-1 * self.linear_nodes]), torch.tanh(driven_state[0, -1 * self.linear_nodes:])), 0)

        return torch.mul(self.kappas, driven_state)
        
class MultiMix(torch.nn.Module):
    def __init__(self, input_size, output_size, reservoir_sizes, linear_nodes):
        super(MultiMix, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.reservoir_sizes = reservoir_sizes
        self.linear_nodes = linear_nodes
        self.layers = torch.nn.ModuleList([])

        last = input_size
        for i, r in enumerate(reservoir_sizes):
            self.layers.append(MixedLayer(r, last, linear_nodes=linear_nodes[i]))
            last = r
        
        self.linear = torch.nn.Linear(last, output_size)
    
    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return self.linear(x)