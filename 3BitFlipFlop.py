import torch
import matplotlib.pyplot as plt
from MixedNet import MixedReservoir
from tqdm import trange

data = torch.load('3bit_flipflop_single_theta_T20_L5_D5_p50_TRAIN.pt')
inputs, outputs = data

torch.manual_seed(0)
torch.autograd.set_detect_anomaly(True)

model = MixedReservoir(3, 3, [64], [[48, 16]]).double()
crit = torch.nn.MSELoss()
opti = torch.optim.Adam(model.parameters(), lr=0.01)

total_loss = 0.0

dataloader = torch.utils.data.DataLoader(inputs, batch_size=64, shuffle=True)
for i, d in enumerate(dataloader):
    opti.zero_grad()
    pred = model(d.double())
    print(pred.shape)
    loss = crit(pred, outputs[i * 64: (i+1) * 64])
    print(f'Batch {i+1} loss: {loss.item()}')
    total_loss += loss.item()
    print(total_loss)
    loss.backward(retain_graph=True)
    opti.step()

print(f'Total loss, {total_loss}')
