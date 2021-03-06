import argparse
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from sympy.parsing.sympy_parser import (parse_expr, standard_transformations, implicit_multiplication_application)
from tqdm import trange
from SCMNet import SCMNet
from MultilayerSCMNet import MultilayerSCMNet
from MultiSCMNet import MultiSCMNet
from HarmonicNN import MultilayerHarmonicNN

parser = argparse.ArgumentParser('Compare an SCM to an RNN of equal volume')
parser.add_argument('-d', '--delta', type=float, dest='delta', default=0.25, help='Distance between points')
parser.add_argument('-e', '--epochs', type=int, dest='epochs', default=250, help='Number of epochs')
parser.add_argument('-f', '--function', type=str, dest='func', default='sine', help='Type of function to learn')
parser.add_argument('-i', '--input_spread', type=int, dest='ins', default=2, help='Number of input channels')
parser.add_argument('-l', '--layers', nargs='+', type=int, dest='layers', default=None, help='Reservoir layers')
parser.add_argument('-n', '--nodes', type=int, dest='nodes', default=1, help='Number of hidden nodes in the RNN')
parser.add_argument('-o', '--output_spread', type=int, dest='outs', default=4, help='Number of output channels')
parser.add_argument('-p', '--parse_function', type=str, dest='func_string', default='sin(x)', help='Custom function to learn, will override built-in functions')
parser.add_argument('-r', '--range', nargs='+', type=int, dest='range', default=[0, 100], help='Range of data to generate')
args = parser.parse_args()

def generate_data(start, end, points):
    full = np.linspace(start, end, points+1)
    if args.func_string is None:
        if args.func == 'sine':
            full = np.sin(full)
    else:
        exp = parse_expr(args.func_string, transformations=(standard_transformations + (implicit_multiplication_application,)))
        full = np.asarray([exp.evalf(subs={'x': i}) for i in full]).astype('float')
    return full[:-1], full[1:]

class testRNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(testRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_size)
    
    def forward(self, x, hidden):
        x, hidden = self.rnn(x, hidden)
        return torch.mean(x), hidden

    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device)
        return hidden

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(0)

# SCM = torch.jit.script(SCMNet(1, 1, args.nodes, args.ins, args.outs)).to(device)
# SCM = torch.jit.script(MultilayerSCMNet(1, 1, [4, 5], 4, 4))
# SCM = torch.jit.script(MultiSCMNet(1, 1, args.layers))
SCM = torch.jit.script(MultilayerHarmonicNN(1, 1, args.layers))
RNN = testRNN(1, 1, args.nodes, 1)

print('Number of parameters for SCM:', sum([len(p.flatten()) for p in SCM.parameters()]))
print('Number of parameters for RNN:', sum([len(p.flatten()) for p in RNN.parameters()]))

reservoir = torch.rand(args.nodes).to(device)
hidden = None

SCMcrit = nn.MSELoss()
RNNcrit = nn.MSELoss()

# SCMopti = torch.optim.SGD(SCM.parameters(), lr = 0.01, momentum=0.9)
SCMopti = torch.optim.Adam(SCM.parameters(), lr = 0.01)
RNNopti = torch.optim.Adam(RNN.parameters(), lr = 0.01)

plt.figure(1)
plt.get_current_fig_manager().window.state('zoomed')
plt.ion()

combined_axis = plt.subplot(212)
actual_axis = plt.subplot(231)
SCM_axis = plt.subplot(232)
RNN_axis = plt.subplot(233)

actual_axis.set_title('Actual')
SCM_axis.set_title('HNN')
RNN_axis.set_title('RNN')

lindata = generate_data(args.range[0], args.range[1], args.epochs)

delta = torch.tensor(args.delta).float().reshape(1, 1, 1)
x = -1.0 * delta
exp = parse_expr(args.func_string, transformations=(standard_transformations + (implicit_multiplication_application,)))

total_SCM = 0.0
total_RNN = 0.0

torch.autograd.set_detect_anomaly(True)
for step in trange(args.epochs):
    # ycx = torch.from_numpy(lindata[0][int(step):int(step)+1][np.newaxis, :, np.newaxis]).float().to(device)
    # ycy = torch.from_numpy(lindata[1][int(step):int(step)+1][np.newaxis, :, np.newaxis]).to(device)
    x += delta
    y = torch.tensor(exp.evalf(subs={'x': x.item()}), dtype=torch.float64).reshape(1,1,1)

    SCMpred = SCM(x / (np.pi/90.0))
    RNNpred, hidden = RNN(x, hidden)
    hidden = hidden.data

    SCMloss = SCMcrit(SCMpred.double(), y)
    RNNloss = RNNcrit(RNNpred.double().reshape(1,1,1), y)

    total_SCM += abs(SCMloss.item())
    total_RNN += abs(RNNloss.item())

    SCMopti.zero_grad()
    RNNopti.zero_grad()

    SCMloss.backward(retain_graph=True)
    RNNloss.backward(retain_graph=True)

    SCMopti.step()
    RNNopti.step()

    combined_axis.plot(x.item(), y.item(), 'go')
    actual_axis.plot(x.item(), y.item(), 'go')

    combined_axis.plot(x.item(), SCMpred.item(), 'ro')
    SCM_axis.plot(x.item(), SCMpred.item(), 'ro')
    SCM_axis.set_title(f'SCM Loss: {round(SCMloss.item(), 6)}')
    
    combined_axis.plot(x.item(), RNNpred.item(), 'bo')
    RNN_axis.plot(x.item(), RNNpred.item(), 'bo')
    RNN_axis.set_title(f'RNN Loss: {round(RNNloss.item(), 6)}')

    combined_axis.set_xlim(x.item()-150 * delta, x.item())
    combined_axis.set_ylim(-2,2)
    
    actual_axis.set_xlim(x.item()-50 * delta, x.item())
    actual_axis.set_ylim(-2,2)
    
    SCM_axis.set_xlim(x.item()-50 * delta, x.item())
    SCM_axis.set_ylim(-2,2)
    
    RNN_axis.set_xlim(x.item()-50 * delta, x.item())
    RNN_axis.set_ylim(-2,2)
    
    if (int(step) - 1) % 50 == 0:
        print('Average error SCM:', total_SCM/int(step))
        print('Average error RNN:', total_RNN/int(step))
        total_SCM = 0.0
        total_RNN = 0.0

    plt.draw(); plt.pause(0.02)

    del SCMpred
    del RNNpred

plt.ioff()
plt.show()

print(f'SCM Loss: {SCMloss}\nRNN loss: {RNNloss}')