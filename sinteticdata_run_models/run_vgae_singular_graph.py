import pandas as pd
import torch
from torch_geometric.nn import VGAE

from models.cat_va_encoder import VariationalGCNEncoder
from sintetic_utils import build_graph, extract_dataset, test_emb_quality, MMD

train_dataset, train_labels, test_dataset, test_labels = extract_dataset()
train_graph = build_graph(train_dataset, train_labels)
test_graph = build_graph(test_dataset, test_labels)

model = VGAE(encoder=VariationalGCNEncoder(train_graph.num_features, 64))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = train_graph
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# print(model.named_parameters())

# print(summary(model, [(1000, 350), (2, 1, 1000), (1,1000), (1000, 1350), (2, 1, 1200), (1,1200)], [torch.float, torch.long, torch.float, torch.float, torch.long, torch.float]))
# print(gnn_model_summary(model))
beta = 15
kl = False


def train():
    model.train()
    optimizer.zero_grad()
    z = model.encode(train_graph.x, train_graph.edge_index, train_graph.edge_attr)
    loss = model.recon_loss(z, train_graph.edge_index)
    if kl == True:
        loss = loss + (1 / train_graph.num_nodes) * model.kl_loss()
    else:
        true_samples = torch.normal(0, 1, size=(train_graph.num_nodes, 64))
        loss += loss + beta * MMD(true_samples, z)
    loss.backward()
    optimizer.step()
    return float(loss)


for epoch in range(1, 2000):
    loss = train()
    # auc, ap = test(test_data)
    print(f'Epoch: {epoch:03d}, LOSS: {loss:.4f}')


#tmodel.load_state_dict(torch.load('dgi_singular_graph.pkl'))

z_train = model.encode(train_graph.x, train_graph.edge_index, train_graph.edge_attr)
z_test = model.encode(test_graph.x, test_graph.edge_index, test_graph.edge_attr)

train, test = test_emb_quality(z_train.detach(), train_labels, z_test.detach(), test_labels)

f = open('results.txt', 'a')
f.write('\nvgae_singular_graph_kl, ' + format(train[0]) + ', ' + format(test[0]) + ', ' + format(train[1]) + ', ' + format(
    test[1]) + ', ' + format(train[2]) + ', ' + format(test[2]))
f.close()
