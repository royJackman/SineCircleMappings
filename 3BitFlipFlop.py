import torch
import matplotlib.pyplot as plt
from MixedNet import MixedReservoir
from tqdm import trange

data = torch.load('3bit_flipflop_single_theta_T20_L5_D5_p50_TRAIN.pt')
inputs, outputs = data

torch.manual_seed(0)
torch.autograd.set_detect_anomaly(True)

model = MixedReservoir(3, 3, [30, 30], [[30], [30]]).double()
crit = torch.nn.MSELoss()
opti = torch.optim.Adam(model.parameters(), lr=0.01)
# opti = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

total_loss = 0.0
batch_size = 64

dataloader = torch.utils.data.DataLoader(inputs, batch_size=batch_size, shuffle=True)
for i in range(int(len(inputs/batch_size))):
    opti.zero_grad()
    pred = model(inputs[i * batch_size: (i+1) * batch_size])
    # print(inputs[0], pred[0], outputs[i * 64])
    loss = crit(pred, outputs[i * batch_size: (i+1) * batch_size])
    print(f'Batch {i+1} loss: {loss.item()}')
    total_loss += loss.item()
    print(total_loss)
    loss.backward(retain_graph=True)
    opti.step()

print(f'Total loss, {total_loss}')
