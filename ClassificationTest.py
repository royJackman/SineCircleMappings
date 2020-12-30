import csv
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from SCMNet import SCMNet
from tqdm import trange

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(0)
model = torch.jit.script(SCMNet(8, 1, 12, 12, 6))
reservoir0 = torch.rand(12).to(device)
crit = nn.MSELoss()
opti = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

data = list(csv.reader(open('Data/abalone.data', 'r')))
transformed = []

gender_cat = {'M': 1.0, 'F': 2.0, 'I': 0.0}

def transform(item):
    return [gender_cat[item[0]], float(item[1]), float(item[2]), float(item[3]), float(item[4]), float(item[5]), float(item[6]), float(item[7]), float(item[8])]

for d in data:
    transformed.append(transform(d))

data = torch.tensor(transformed)
data = data/data.max(0, keepdim=True)[0]
inputs = data[:, :-1]
outputs = data[:, -1:]
split = int(data.size()[0] * 0.8)
# split = 1000

train_x = inputs[:split, :].to(device)
train_y = outputs[:split, :].to(device)
test_x = inputs[split:, :].to(device)
test_y = outputs[split:, :].to(device)

# plt.figure(1)
# plt.ion()

total_loss = 0.0
for i in range(3):
    epoch_loss = 0.0
    for j in trange(split):
        pred, reservoir0 = model(train_x[j, :], reservoir0)
        loss = crit(pred.double(), train_y[j, :].flatten().double())
        # plt.plot(int(i*split+j), pred.item(), 'ro')
        # plt.plot(int(i*split+j), train_y[j, :].flatten()[0], 'go')
        plt.show(); plt.pause(0.03)
        epoch_loss += loss.item()
        opti.zero_grad()
        loss.backward(retain_graph=True)
        opti.step()
    total_loss += epoch_loss
    print(f'Epoch: {i}, Avg Loss: {epoch_loss/split}')

loss = 0.0
for i, v in enumerate(test_x):
    pred, _ = model(v, reservoir0)
    loss += crit(pred.double(), test_y[i, :].flatten().double()).item()
print("Test:", loss/len(test_x))

# plt.ioff()
# plt.show()

print(f'Avg Loss: {total_loss/(split * 10)}')
print('Alphas:   ', model.alphas.detach().numpy(), 
      '\nKs:       ', model.ks.detach().numpy(), 
      '\nOmegas:   ', model.omegas.detach().numpy(), 
      '\nReservoir:', reservoir0.detach().flatten().numpy())
