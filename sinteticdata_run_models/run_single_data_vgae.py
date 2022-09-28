import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.nn import DeepGraphInfomax, VGAE

from sintetic_utils import build_graph, test_emb_quality, extract_hetero_dataset, MMD
from sinteticdata_run_models.models.DGI_single_graph_encoder import Encoder, corruption
from sinteticdata_run_models.models.cat_va_encoder import VariationalGCNEncoder

dataset1_train, dataset1_test, dataset2_train, dataset2_test, train_labels, test_labels = extract_hetero_dataset()
train_graph1 = build_graph(dataset1_train, train_labels)
train_graph2 = build_graph(dataset2_train, train_labels)

test_graph1 = build_graph(dataset1_test, test_labels)
test_graph2 = build_graph(dataset2_test, test_labels)


model = DeepGraphInfomax(
    hidden_channels=64, encoder=Encoder(train_graph1.num_features, 64),
    summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
    corruption=corruption)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# plot_ax(torch.cat([dataset1.x,dataset2.x], dim=1).numpy(), data1.y, 0, 'label', 0, 'sintetic_data', 0, 0)
for train_graph, test_graph, i in zip([train_graph1, train_graph2], [test_graph1, test_graph2], [1,2]):
    model = VGAE(encoder=VariationalGCNEncoder(train_graph.num_features, 64))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = train_graph
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # print(model.named_parameters())

    # print(summary(model, [(1000, 350), (2, 1, 1000), (1,1000), (1000, 1350), (2, 1, 1200), (1,1200)], [torch.float, torch.long, torch.float, torch.float, torch.long, torch.float]))
    # print(gnn_model_summary(model))
    beta = 15
    kl = True


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

    for epoch in range(1, 1000):
        loss = train()
        # auc, ap = test(test_data)

        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')


    model.eval()
    z_train, _, _ = model(train_graph.x, train_graph.edge_index, train_graph.edge_attr)
    z_test, _, _ = model(test_graph.x, test_graph.edge_index, test_graph.edge_attr)

    train, test = test_emb_quality(z_train.detach(), train_labels, z_test.detach(), test_labels)

    f = open('results.txt', 'a')
    f.write(
        '\ndgi_1_graph_'+format(i)+', '+ format(train[0]) + ', ' + format(test[0]) + ', ' + format(train[1]) + ', ' + format(
            test[1]) + ', ' + format(train[2]) + ', ' + format(test[2]))
    f.close()
