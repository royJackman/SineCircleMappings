import torch
import operator

import torch.nn as nn
import pandas as pd
import numpy as np

from HarmonicNN import MultilayerHarmonicNN
from tqdm import trange

torch.set_printoptions(linewidth=1000)

data = pd.read_csv('Data/MNIST/train.csv')
data = torch.tensor(data[data.columns].values, dtype=torch.float64)

labels = data[:, 0]
pixels = data[:, 1:].reshape(42000, 28, 28).double()
pixels = torch.div(pixels, -1.0 * pixels.max().item())
pixels = torch.add(pixels, 1.0)
del data

examples = 1000
tests = 100

torch.manual_seed(55)
HNN = torch.jit.script(MultilayerHarmonicNN(56, 10))
crit = torch.nn.MSELoss()
opti = torch.optim.Adam(HNN.parameters(), lr = 0.01)

onehot_transitions = torch.nn.functional.one_hot(torch.arange(0, 10)).double()
print('HNN Params:', sum([len(p.flatten()) for p in HNN.parameters()]))
for repetition in range(15):
    total_loss = 0.0
    correct = 0
    for step in trange(examples):
        image = pixels[int(step)]
        number = int(labels[int(step)])

        example_loss = 0.0
        # votes = {}
        guess = torch.zeros(10)
        for i in range(28):
            inp = torch.cat((image[i, :].flatten(), image[:, i].flatten()), 0)[np.newaxis, :]
            pred = HNN(inp)
            # guess = int(pred.item())
            # votes[guess] = votes.get(guess, 0) + 1
            # loss = crit(pred.double(), torch.tensor(number, dtype=torch.float64))
            guess += pred.flatten().double()
            loss = crit(pred.double(), onehot_transitions[number])
            example_loss += loss.item()
            opti.zero_grad()
            loss.backward(retain_graph=True)
            opti.step()
        total_loss += example_loss/28.0
        # guess = max(votes.items(), key=operator.itemgetter(1))[0]

        correct += 1 if torch.argmax(guess).item() == number else 0

    grade = 0
    for step in trange(tests):
        image = pixels[examples + int(step)]
        number = int(labels[examples + int(step)])
        # votes = {}
        guess = torch.zeros(10)
        for i in range(28):
            inp = torch.cat((image[i, :].flatten(), image[:, i].flatten()), 0)[np.newaxis, :]
            pred = HNN(inp)
            guess += pred.flatten().double()
            # guess = int(pred.item())
            # votes[guess] = votes.get(guess, 0) + 1
        # guess = max(votes.items(), key=operator.itemgetter(1))[0]
        grade += 1 if torch.argmax(guess).item() == number else 0

    print(f'Round {repetition + 1}:')
    print('Training correct:', correct)
    print('Training accuracy:', correct/float(examples))

    print('Testing correct:', grade)
    print('Testing accuracy:', grade/float(tests))
    print('-' * 80)