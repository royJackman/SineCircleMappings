import random
import torch

from MixedNet import MixedNet
from tqdm import trange

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

data = torch.load('data/2cycle_P20_p20_examples65536_TRAIN.pt')
inputs, outputs = data

test_data = torch.load('data/2cycle_P20_p20_examples65536_TEST.pt')
test_inputs, test_outputs = test_data

inputs = inputs.to(device).double()
outputs = outputs.to(device).double()
test_inputs = test_inputs.to(device).double()
test_outputs = test_outputs.to(device).double()

distribution = 0.2 * torch.ones(5)
layer_size = 16
model_batch = 8

inner_epochs = 10
inner_batch = 100

input_size = 2
output_size = 2

for epoch in range(100):
    node_distribution = torch.mul(layer_size, distribution).type(torch.uint8).tolist()
    while sum(node_distribution) != layer_size:
        node_distribution[random.randint(0,4)] += 1 if sum(node_distribution) < layer_size else -1
    
    models = [MixedNet(input_size, output_size, [layer_size], [node_distribution]).double().to(device=device) for i in range(model_batch)]
    crits = [torch.nn.MSELoss() for i in range(model_batch)]
    optis = [torch.optim.Adam(models[i].parameters(), lr=0.01) for i in range(model_batch)]

    for inner in trange(inner_epochs):
        for i in range(int(len(inputs)/inner_batch)):
            for o in optis:
                o.zero_grad()
            
            preds = [m(inputs[i*inner_batch: (i+1)*inner_batch]) for m in models]
            losses = [c(preds[j], outputs[i*inner_batch: (i+1)*inner_batch]) for j, c in enumerate(crits)]

            for j, l in enumerate(losses):
                l.backward(retain_graph=True)
                optis[j].step()
    
    total_loss = 0.0
    for i in range(int(len(inputs)/inner_batch)):
        preds = [m(test_inputs[i*inner_batch: (i+1)*inner_batch]) for m in models]
        losses = [c(preds[j], test_outputs[i*inner_batch: (i+1)*inner_batch]) for j, c in enumerate(crits)]
    print(f'Epoch {epoch}: Distribution: {node_distribution} Average Loss: {sum(losses)/model_batch}')
            