import pandas as pd
import torch
from torch_geometric.nn import DeepGraphInfomax

from models.encoder_hetero import DoubleLayeredEncoder
from sintetic_utils import build_graph, build_hetero_graph, extract_hetero_dataset, \
    test_emb_quality

def corruption(x, edge_index, edge_weight, edge_type):
    return x[torch.randperm(x.size(0))], edge_index, edge_weight, edge_type

torch.manual_seed(9047)
dataset1_train, dataset1_test, dataset2_train, dataset2_test, train_labels, test_labels = extract_hetero_dataset()
train_graph1 = build_graph(dataset1_train, train_labels, h=1.2)
train_graph2 = build_graph(dataset2_train, train_labels, h=0.75)

test_graph1 = build_graph(dataset1_test, test_labels, h=1.2)
test_graph2 = build_graph(dataset2_test, test_labels, h=0.75)

train_hetero_graph = build_hetero_graph(train_graph1, train_graph2)
test_hetero_graph = build_hetero_graph(test_graph1, test_graph2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DeepGraphInfomax(
    hidden_channels=64, encoder=DoubleLayeredEncoder(train_hetero_graph.num_features, 64),
    summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
    corruption=corruption).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

cnt_wait = 0
patience = 20
best = 1e9
best_t = 0

#train_hetero_graph.edge_attr = torch.exp(-train_hetero_graph.edge_attr)
#test_hetero_graph.edge_attr = torch.exp(-test_hetero_graph.edge_attr)

def train(epoch):
    nr_edges = train_hetero_graph.edge_type.shape[0]
    model.train()
    optimizer.zero_grad()

    pos_z, neg_z, summary = model(train_hetero_graph.x, train_hetero_graph.edge_index, train_hetero_graph.edge_attr, train_hetero_graph.edge_type)
    loss = model.loss(pos_z, neg_z, summary)

    loss.backward()
    optimizer.step()
    return loss.item()


for epoch in range(1, 100):
    loss = train(epoch)
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

    if loss < best:
        best = loss
        best_t = epoch
        cnt_wait = 0
        torch.save(model.state_dict(), 'dgi_hetero_graph.pkl')
    else:
        cnt_wait += 1

    if cnt_wait == patience:
        print('Early stopping!')
        break

model.load_state_dict(torch.load('dgi_hetero_graph.pkl'))

model.eval()
z_train, _, _ = model(train_hetero_graph.x, train_hetero_graph.edge_index, train_hetero_graph.edge_attr, train_hetero_graph.edge_type)
z_test, _, _ = model(test_hetero_graph.x, test_hetero_graph.edge_index, test_hetero_graph.edge_attr, test_hetero_graph.edge_type)

train, test = test_emb_quality(z_train.detach(), train_labels, z_test.detach(), test_labels)

f = open('results.txt', 'a')
f.write('\ndgi_hetero_graph, ' + format(train[0]) + ', ' + format(test[0]) + ', ' + format(train[1]) + ', ' + format(
    test[1]) + ', ' + format(train[2]) + ', ' + format(test[2]))
f.close()
