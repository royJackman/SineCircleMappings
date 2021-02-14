import torch
import torch.nn as nn

torch.pi = torch.acos(torch.zeros(1)).item() * 2

class HarmonicNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(HarmonicNN, self).__init__()
        self.input_size, self.output_size = input_size, output_size
        self.alphas = nn.Parameter(torch.rand(input_size))
        self.betas = nn.Parameter(torch.rand(input_size))
        self.kappas = nn.Parameter(torch.rand(input_size))
        self.gammas = nn.Parameter(torch.rand(input_size))
        self.deltas = nn.Parameter(torch.rand(input_size))
        self.mask = torch.rand(input_size, output_size).double()
    
    def forward(self, x):
        forget = torch.mul(self.alphas.clone(), x.clone())
        linear = self.betas.clone()
        intsin = torch.mul(2.0 * torch.pi * self.gammas.clone(), x.clone()) + self.deltas.clone()
        extsin = torch.mul(torch.mul(self.kappas.clone(), (1.0/(2.0 * torch.pi))), torch.sin(intsin))
        updated = forget + linear + extsin
        if len(updated.shape) == 3:
            updated = updated[0]
        # return torch.mm(updated.double(), torch.ones(self.input_size, self.output_size).double())
        return torch.mm(updated.double(), self.mask)

class MultilayerHarmonicNN(nn.Module):
    def __init__(self, input_size, output_size, layers=None):
        super(MultilayerHarmonicNN, self).__init__()
        if layers is None:
            self.layers = nn.ModuleList([HarmonicNN(input_size, output_size)])
        else:
            self.layers = nn.ModuleList([HarmonicNN(input_size, layers[0])])
            for i, l in enumerate(layers[:-1]):
                self.layers.append(HarmonicNN(l, layers[i+1]))
            self.layers.append(HarmonicNN(layers[-1], output_size))
        
        self.linear = None if output_size == 1 else nn.Parameter(torch.rand(output_size))
    
    def forward(self, x):
        for l in self.layers:
            x = l(x.clone())
        if self.linear is None:
            return x
        else:
            return torch.mul(x, self.linear.clone())
