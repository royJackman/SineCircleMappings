import random
import sys
import torch

import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class NodesLayer(torch.nn.Module):
    def __init__(self, alphas):
        super(NodesLayer, self).__init__()
        self.alphas = alphas.to(device)
    
    def forward(self, x):
        return torch.mul(self.alphas, x.to(device))

class MixedNet(torch.nn.Module):
    def __init__(self, input_size, output_size, layers, distributions=None, dist_order=['sin', 'tanh', 'log', 'relu', 'sigmoid']):
        super(MixedNet, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.layers = layers
        self.transitions = []
        self.distributions = [] if distributions is None else distributions
        self.dist_order = dist_order

        self.alphas = torch.nn.ModuleList([]).to(device)

        last = input_size
        for i, l in enumerate(self.layers):
            self.alphas.append(NodesLayer(torch.nn.Parameter(torch.rand(l))).to(device))

            if distributions is None:
                self.distributions.append([l])
            elif sum(self.distributions[i]) != l:
                sys.exit(f'Distribution at layer {i + 1} does not match layer size')
            elif len(self.distributions[i]) > len(dist_order):
                sys.exit(f'Distributing over more functions than available on layer {i+1}')
            
            self.transitions.append(torch.rand((last, l)).to(device))
            last = l
            
        self.linear = torch.nn.Linear(last, self.output_size).to(device)
    
    def forward(self, x):
        # Loop through layers of network
        for i, a in enumerate(self.alphas):
            x = torch.matmul(x.double(), self.transitions[i].double())

            # Perform mixed update
            updates = []
            for j, s in enumerate(torch.split(x, self.distributions[i], dim=-1)):
                if self.dist_order[j] == 'sin':
                    updates.append(torch.sin(s))
                elif self.dist_order[j] == 'tanh':
                    updates.append(torch.tanh(s))
                elif self.dist_order[j] == 'log':
                    updates.append(torch.log(torch.pow(s, 2)))
                elif self.dist_order[j] == 'relu':
                    updates.append(torch.relu(s))
                elif self.dist_order[j] == 'sigmoid':
                    updates.append(torch.sigmoid(s))
                else:
                    updates.append(s)

            x = torch.cat(updates, dim=2)
            x = a(x.double())
        return self.linear(x)

class MixedReservoir(torch.nn.Module):
    def __init__(self, input_size, output_size, reservoir_sizes, distributions=None, dist_order=['sin', 'tanh', 'log', 'relu', 'sigmoid']):
        super(MixedReservoir, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.reservoir_sizes = reservoir_sizes
        self.distributions = [] if distributions is None else distributions
        self.dist_order = dist_order

        self.reservoirs = []
        self.states = []
        self.transitions = []
        
        self.alphas = torch.nn.ModuleList([])

        last = input_size
        for i, l in enumerate(self.reservoir_sizes):
            self.alphas.append(NodesLayer(torch.nn.Parameter(torch.rand(l))))
            self.reservoirs.append(torch.normal(0, 1/np.sqrt(l), (l,l)))
            self.states.append(torch.zeros(l))

            if distributions is None:
                self.distributions.append([l])
            elif sum(self.distributions[i]) != l:
                raise ValueError(f'Distribution at layer {i + 1} does not match layer size')
            elif len(self.distributions[i]) > len(dist_order):
                raise ValueError(f'Distributing over more functions than available on layer {i+1}')
            
            self.transitions.append(torch.rand((last, l)))
            last = l

        self.linear = torch.nn.Linear(last, output_size)

    def forward(self, x):
        # Initialize return value and empty batch internal states
        retval = torch.zeros((x.shape[0], x.shape[1], self.output_size))
        batch_states = [torch.zeros((x.shape[0], r)) for r in self.reservoir_sizes]

        # Loop through timesteps 
        for t in range(x.shape[1]):
            temp_x = x[:, t, :].clone().double()

            # Loop through layers of network
            for i, a in enumerate(self.alphas):
                temp_x = torch.matmul(temp_x.double(), self.transitions[i].double())
                temp_x = torch.add(temp_x.double(), batch_states[i].double())
                temp_x = torch.matmul(temp_x.double(), self.reservoirs[i].double())

                # Perform mixed update
                updates = []
                for j, s in enumerate(torch.split(temp_x.double(), self.distributions[i], dim=1)):
                    if self.dist_order[j] == 'sin':
                        updates.append(torch.sin(s))
                    elif self.dist_order[j] == 'tanh':
                        updates.append(torch.tanh(s))
                    elif self.dist_order[j] == 'log':
                        updates.append(torch.log(torch.pow(s, 2)))
                    elif self.dist_order[j] == 'relu':
                        updates.append(torch.relu(s))
                    elif self.dist_order[j] == 'sigmoid':
                        updates.append(torch.sigmoid(s))
                    else:
                        updates.append(s)

                temp_x = torch.cat(updates, dim=1)
                temp_x = a(temp_x.clone())
                batch_states[i] = temp_x.clone()
            retval[:, t, :] = self.linear(temp_x)
        return retval
    
    def reset_states(self):
        for i, l in enumerate(self.reservoir_sizes):
            self.states[i] = torch.zeros(l)