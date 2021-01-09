import argparse
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from sympy.parsing.sympy_parser import (parse_expr, standard_transformations, implicit_multiplication_application)
from tqdm import trange
from SCMNet import SCMNet

parser = argparse.ArgumentParser('Compare an SCM to an RNN of equal volume')
parser.add_argument('-e', '--epochs', type=int, dest='epochs', default=250, help='Number of epochs')
parser.add_argument('-f', '--function', type=str, dest='func', default='sine', help='Type of function to learn')
parser.add_argument('-i', '--input_spread', type=int, dest='ins', default=2, help='Number of input channels')
parser.add_argument('-n', '--nodes', type=int, dest='nodes', default=6, help='Number of nodes in the reservoir')
parser.add_argument('-o', '--output_spread', type=int, dest='outs', default=4, help='Number of output channels')
parser.add_argument('-p', '--parse_function', type=str, dest='func_string', default=None, help='Custom function to learn, will override built-in functions')
parser.add_argument('-r', '--range', nargs='+', type=int, dest='range', default=[0, 100], help='Range of data to generate')
parser.add_argument('-w', '--window', type=int, dest='window', default=50, help='Window length into the past')
args = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(0)
SCM = torch.jit.script(SCMNet(1, 1, args.nodes, args.ins, args.outs)).to(device)
RNN = nn.RNN(1, args.nodes, 1, batch_first=True).to(device)

print('Number of parameters for SCM:', sum([len(p.flatten()) for p in SCM.parameters()]))
print('Number of parameters for RNN:', sum([len(p.flatten()) for p in RNN.parameters()]))