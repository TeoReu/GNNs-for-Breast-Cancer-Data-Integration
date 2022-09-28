import os

import networkx as nx
import torch
import torch_geometric
import torch_geometric.transforms as T
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from torch_geometric.data import HeteroData


def build_graph_radius(x, r):
    #x = torch.from_numpy(x)

    edge_index1 = torch_geometric.nn.radius_graph(x, r=r, loop=True)
    edge_index2 = torch_geometric.nn.knn_graph(x, k=2, loop=True, flow='source_to_target')
    edge_index = torch.unique(torch.cat([edge_index1, edge_index2], dim=1), dim=1)

    data = torch_geometric.data.Data(x=x, edge_index=edge_index, pos=x, dtype=torch.float)

    return data


def radius_transformation(x, r, raw_dgi=False):
    if raw_dgi:
        x = torch.from_numpy(x)
    transorm = T.Compose([
        # T.NormalizeFeatures(),
        T.RadiusGraph(r),
        #T.KNNGraph(1),
        T.Distance(),
        T.AddSelfLoops()
    ])

    data = torch_geometric.data.Data(x=x, pos=x, dtype=torch.float)

    return transorm(data)

def combined_transformation(x, r, k, raw_dgi=False):
    if raw_dgi:
        x = torch.from_numpy(x)

    transform = T.Compose([
        T.NormalizeFeatures(),
        T.RadiusGraph(r),
        T.KNNGraph(k),
        #T.AddSelfLoops(),
        T.Distance(),
    ])

    data = torch_geometric.data.Data(x=x, pos=x, dtype=torch.float)

    return transform(data)


def euclidian_similarity(x, m):
    # x = torch.from_numpy(x)
    W = torch.cdist(x, x)
    # W = torch.nn.functional.normalize(W, p=2.0, dim=1)
    return torch.mean(W) * m


def RBF_kernel_similarity(x):
    return torch.exp(-torch.dist(x, x))


def build_radius_dataset(dt, s1, s2, m):
    s1_train = dt.train[s1]
    s1_test = dt.test[s1]
    s2_train = dt.train[s2]
    s2_test = dt.test[s2]

    e1_train = euclidian_similarity(s1_train, m)
    g1_train = build_graph_radius(s1_train, e1_train.item())

    e1_test = euclidian_similarity(s1_test, m)
    g1_test = build_graph_radius(s1_test, e1_test.item())

    e2_train = euclidian_similarity(s2_train, m)
    g2_train = build_graph_radius(s2_train, e2_train.item())

    e2_test = euclidian_similarity(s2_test, m)
    g2_test = build_graph_radius(s2_test, e2_test.item())

    return g1_train, g2_train, g1_test, g2_test


def build_neighbours_dataset(dt, s1, s2, k):
    s1_train = dt.train[s1]
    s1_test = dt.test[s1]
    s2_train = dt.train[s2]
    s2_test = dt.test[s2]

    g1_train = build_graph_neighbours(s1_train, k)

    g1_test = build_graph_radius(s1_test, k)

    g2_train = build_graph_radius(s2_train, k)

    g2_test = build_graph_radius(s2_test, k)

    return g1_train, g2_train, g1_test, g2_test


def draw_graph(data):
    g = torch_geometric.utils.to_networkx(data)
    # g.add_edges_from(data.edge_index)
    # pretty colours: #db9cc3, #d17c62
    g.remove_edges_from(nx.selfloop_edges(g))

    nx.draw_spring(g.to_undirected(reciprocal=False, as_view=False), node_size=10,
                   node_color=data.y)
    plt.show()


def draw_kawai(data):
    g = torch_geometric.utils.to_networkx(data).to_undirected(reciprocal=False, as_view=False)
    g.remove_edges_from(nx.selfloop_edges(g))

    k = nx.kamada_kawai_layout(g.to_undirected(reciprocal=False, as_view=False), weight=data.edge_attr)

    nx.draw(g, pos=k, node_size=10,
            node_color=data.y)
    plt.show()


def build_graph_neighbours(x, k):
    x = torch.from_numpy(x)
    data = torch_geometric.data.Data(pos=x, dtype=torch.float)

    edge_index = torch_geometric.nn.knn_graph(data.pos, k=k, loop=False, flow='source_to_target')

    data = torch_geometric.data.Data(x=x, edge_index=edge_index, pos=x, dtype=torch.float)

    return data


def plot_ax(x, y, e, gc, hp, intgr, ep, fold):
    tsneRAW = TSNE(random_state=42, perplexity=40)
    x = tsneRAW.fit_transform(x)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title("graph_construction " + format(gc) + " " + format(hp) + "_epoch_" + format(e))
    ax.scatter(x[:, 0], x[:, 1], c=y, s=2)
    #fig.savefig('SINTETIC_DATA_DGI_CAT_N' + '/' + format(intgr) + '_fold_' + format(fold) + "_" + format(gc) + '_' + format(
    #    hp) + '_' + format(ep) + '.jpg', dpi=400)
    plt.show()


def edge_cut(edge_index, pos):
    row, col = edge_index
    edge_attr = torch.exp(-torch.dist(pos[row] - pos[col]))

    return edge_attr


def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.show()


def gnn_model_summary(model):
    model_params_list = list(model.named_parameters())
    print("----------------------------------------------------------------")
    line_new = "{:>20}  {:>25} {:>15}".format("Layer.Parameter", "Param Tensor Shape", "Param #")
    print(line_new)
    print("----------------------------------------------------------------")
    for elem in model_params_list:
        p_name = elem[0]
        p_shape = list(elem[1].size())
        p_count = torch.tensor(elem[1].size()).prod().item()
        line_new = "{:>20}  {:>25} {:>15}".format(p_name, str(p_shape), str(p_count))
        print(line_new)
    print("----------------------------------------------------------------")
    total_params = sum([param.nelement() for param in model.parameters()])
    print("Total params:", total_params)
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Trainable params:", num_trainable_params)
    print("Non-trainable params:", total_params - num_trainable_params)


def RbfKernel(data1, data2, sigma):
    delta = abs(np.subtract(data1, data2))
    squaredEuclidean = (np.square(delta).sum(axis=1))
    result = np.exp(-(squaredEuclidean) / (2 * sigma ** 2))
    return result


def plot_dist_distribution(W, title, mean):
    np.random.seed(42)
    x = torch.reshape(W, (-1,)).numpy()

    lim = 2 * mean
    plt.hist(x, density=False, bins=1000)
    plt.xlim(0, lim)

    plt.title(title)  # density=False would make counts
    plt.ylabel('Quantity')
    plt.xlabel('Distances')
    plt.savefig('distance_distribution/' + title + '.jpg')
    plt.show()


def save_embedding(savedir, savefile, *args):
    save_path = os.path.join(savedir, savefile)
    if len(args) > 1:
        np.savez(save_path, emb_train=args[0].detach().numpy(), emb_test=args[1])
    else:
        np.savez(save_path, emb_train=args[0])


def MMD(x, y):
    """Emprical maximum mean discrepancy. The lower the result
       the more evidence that distributions are the same.

    Args:
        x: first sample, distribution P
        y: second sample, distribution Q
        kernel: kernel type such as "multiscale" or "rbf"
    """
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    dxx = rx.t() + rx - 2. * xx  # Used for A in (1)
    dyy = ry.t() + ry - 2. * yy  # Used for B in (1)
    dxy = rx.t() + ry - 2. * zz  # Used for C in (1)

    XX, YY, XY = (torch.zeros(xx.shape),
                  torch.zeros(xx.shape),
                  torch.zeros(xx.shape))



    bandwidth_range = [10, 15, 20, 50]
    for a in bandwidth_range:
        XX += torch.exp(-0.5 * dxx / a)
        YY += torch.exp(-0.5 * dyy / a)
        XY += torch.exp(-0.5 * dxy / a)

    return torch.mean(XX + YY - 2. * XY)

def get_label(s):
    if s == 'DR':
        return 'drnp'
    elif s == 'ER':
        return 'ernp'
    elif s == 'PAM':
        return 'pam50np'
    else:
        return 'icnp'

def test_emb_quality(emb_train, train_labels, emb_test, test_labels):
    accsTrain = []
    accsTest = []
    nb = GaussianNB()
    nb.fit(emb_train, train_labels)

    x_p_classes = nb.predict(emb_train)
    accTrain = accuracy_score(train_labels, x_p_classes)
    accsTrain.append(accTrain)

    y_p_classes = nb.predict(emb_test)
    accsTest.append(accuracy_score(test_labels, y_p_classes))

    svm = SVC(C=1.5, kernel='rbf', random_state=42, gamma='auto')
    svm.fit(emb_train, train_labels)

    x_p_classes = svm.predict(emb_train)
    accTrain = accuracy_score(train_labels, x_p_classes)
    accsTrain.append(accTrain)

    y_p_classes = svm.predict(emb_test)
    accsTest.append(accuracy_score(test_labels, y_p_classes))

    rf = RandomForestClassifier(n_estimators=50, random_state=42,
                                max_features=.5)
    rf.fit(emb_train, train_labels)

    x_p_classes = rf.predict(emb_train)
    accTrain = accuracy_score(train_labels, x_p_classes)
    accsTrain.append(accTrain)

    y_p_classes = rf.predict(emb_test)
    accsTest.append(accuracy_score(test_labels, y_p_classes))

    return accsTrain, accsTest

def build_hetero_graph(graph1, graph2):
    data = HeteroData()

    data['1'].x = torch.cat([graph1.x, torch.zeros(graph2.num_nodes, graph2.num_features)], dim=1)
    data['2'].x = torch.cat([torch.zeros(graph1.num_nodes, graph1.num_features), graph2.x], dim=1)

    data['1', 'close', '1'].edge_index = graph1.edge_index
    data['1', 'close', '1'].edge_attr = graph1.edge_attr

    data['2', 'close', '2'].edge_index = graph2.edge_index
    data['2', 'close', '2'].edge_attr = graph2.edge_attr


    node_to_node = torch.zeros((2, graph1.num_nodes), dtype=torch.int)
    for i in range(graph1.num_nodes):
        node_to_node[0][i] = int(i)
        node_to_node[1][i] = int(i)

    data['1', 'is', '2'].edge_index = node_to_node
    data['1','is','2'].edge_attr = torch.ones(graph1.num_nodes,1)
    data = T.ToUndirected()(data)
    #data = T.AddSelfLoops()(data)

    data = data.to_homogeneous()

    return data
