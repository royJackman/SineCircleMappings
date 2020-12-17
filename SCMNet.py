import argparse
import torch
import torch.nn as nn
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

def scm(theta, alpha=1.0, k=1.0, omega=0.16): return alpha * theta + omega + (k/(2 * np.pi)) * np.sin(2 * np.pi * theta)

class SCMNet(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(SCMNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_size * 3)
        self.params = []
    
    def forward(self, x, hidden):
        theta_n = x
        x, hidden = self.rnn(x, hidden)
        
        outs = []
        for time_step in range(x.size(1)):
            self.params = self.linear(x[:, time_step, :]).flatten()
            outs.append(scm(theta_n, self.params[0], self.params[1], self.params[2]))
        return torch.stack(outs, dim=1), hidden

    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device)
        return hidden

torch.manual_seed(0)