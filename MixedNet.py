import random
import sys
import torch

import numpy as np

class NodesLayer(torch.nn.Module):
    def __init__(self, alphas):
        super(NodesLayer, self).__init__()
        self.alphas = alphas
    
    def forward(self, x):
        return torch.mul(self.alphas, x)

class MixedNet(torch.nn.Module):
    def __init__(self, input_size, output_size, layers, distributions=None, dist_order=['sin', 'tanh', 'log']):
        super(MixedNet, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.layers = layers
        self.transitions = []
        self.distributions = [] if distributions is None else distributions
        self.dist_order = dist_order

        self.alphas = torch.nn.ModuleList([])

        last = input_size
        for i, l in enumerate(self.layers):
            self.alphas.append(NodesLayer(torch.nn.Parameter(torch.rand(l))))

            if distributions is None:
                self.distributions.append([l])
            elif sum(self.distributions[i]) != l:
                sys.exit(f'Distribution at layer {i + 1} does not match layer size')
            elif len(self.distributions[i]) > len(dist_order):
                sys.exit(f'Distributing over more functions than available on layer {i+1}')
            
            self.transitions.append(torch.rand((last, l)))
            last = l
            
        self.transitions.append(torch.rand((last, self.output_size)))
    
    def forward(self, x):
        retval = torch.zeros(x.shape[0], self.output_size)
        for e, example in enumerate(x):
            temp_x = example.clone()
            for i, a in enumerate(self.alphas):
                temp_x = torch.matmul(temp_x.double(), self.transitions[i].double())
                d = self.distributions[i]
                updates = []
                for i, s in enumerate(torch.split(temp_x, d, dim=0)):
                    if self.dist_order[i] == 'sin':
                        updates.append(torch.sin(s))
                    elif self.dist_order[i] == 'tanh':
                        updates.append(torch.tanh(s))
                    elif self.dist_order[i] == 'log':
                        updates.append(torch.log(torch.pow(s, 2)))
                    else:
                        updates.append(s)

                temp_x = torch.cat(updates, 0)
                temp_x = a(temp_x.double())
            retval[e, :] = torch.matmul(temp_x.double(), self.transitions[-1].double())
        return retval.double()

class MixedReservoir(torch.nn.Module):
    def __init__(self, input_size, output_size, reservoir_sizes, distributions=None, dist_order=['sin', 'tanh', 'log']):
        super(MixedReservoir, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.reservoir_sizes = reservoir_sizes
        self.reservoirs = []
        self.states = []
        self.transitions = []
        self.distributions = [] if distributions is None else distributions
        self.dist_order = dist_order

        self.alphas = torch.nn.ModuleList([])

        last = input_size
        for i, l in enumerate(self.reservoir_sizes):
            self.alphas.append(NodesLayer(torch.nn.Parameter(torch.rand(l))))
            self.reservoirs.append(torch.normal(0, 1/np.sqrt(l), (l,l)))
            self.states.append(torch.zeros(l))

            if distributions is None:
                self.distributions.append([l])
            elif sum(self.distributions[i]) != l:
                sys.exit(f'Distribution at layer {i + 1} does not match layer size')
            elif len(self.distributions[i]) > len(dist_order):
                sys.exit(f'Distributing over more functions than available on layer {i+1}')
            
            self.transitions.append(torch.rand((last, l)))
            last = l
            
        self.linear = torch.nn.Linear(last, output_size)

    def forward(self, x):
        retval = torch.zeros((x.shape[0], x.shape[1], self.output_size))
        batch_states = x.shape[0] * [self.states]
        for t in range(x.shape[1]):
            for example in range(x.shape[0]):
                for i, a in enumerate(self.alphas):
                    temp_x = torch.matmul(x[example, t, :].clone().double(), self.transitions[i].double())
                    temp_x = torch.add(temp_x.double(), batch_states[example][i].double())
                    temp_x = torch.matmul(temp_x.double(), self.reservoirs[i].double())

                    updates = []
                    for j, s in enumerate(torch.split(temp_x.double(), self.distributions[i], dim=0)):
                        if self.dist_order[j] == 'sin':
                            updates.append(torch.sin(s))
                        elif self.dist_order[j] == 'tanh':
                            updates.append(torch.tanh(s))
                        elif self.dist_order[j] == 'log':
                            updates.append(torch.log(torch.pow(s, 2)))
                        else:
                            updates.append(s)
                    temp_x = torch.cat(updates)
                    temp_x = a(temp_x)
                    batch_states[example][i] = temp_x
                retval[example, t, :] = self.linear(temp_x)
        
        return retval
    
    def reset_states(self):
        for i, l in enumerate(self.reservoir_sizes):
            self.states[i] = torch.zeros(l)