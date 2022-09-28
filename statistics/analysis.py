import argparse
import math
import os.path as osp

import torch
import torch.nn as nn
from torch.nn import Parameter, Linear
from torch.nn.init import uniform

from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
import torch_geometric.transforms as T
from torch_geometric.nn.inits import reset
from torchvision import models
from torchsummary import summary

from misc.dataset import Dataset, DatasetWhole
from misc.helpers import normalizeRNA
from models.infomax_weighted import DGI_CAT
from utils import build_graph_neighbours, euclidian_similarity, build_graph_radius, draw_graph, plot_ax, \
    radius_transformation, plot_grad_flow, gnn_model_summary, draw_kawai, plot_dist_distribution

'''
dataset = 'Cora'
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
dataset = Planetoid(path, dataset)
'''
EPS = 1e-13

parser = argparse.ArgumentParser()
parser.add_argument('--dtype', help='The type of data (Pam50, Pam50C, IC, ER)', type=str, default='W')

if __name__ == "__main__":
    args = parser.parse_args()


patience = 1000000


dataset = DatasetWhole(args.dtype)

dataset1 = dataset.train['clin']
dataset2 = normalizeRNA(dataset.train['rnanp'])
dataset3 = dataset.train['cnanp']

'''
W1 = torch.cdist(torch.from_numpy(dataset1), torch.from_numpy(dataset1))
W1 = W1[W1 != 0]
mean1 = torch.mean(W1)
W2 = torch.cdist(torch.from_numpy(dataset2), torch.from_numpy(dataset2))
W2 = W2[W2 != 0]
mean2 = torch.mean(W2)
W3 = torch.cdist(torch.from_numpy(dataset3), torch.from_numpy(dataset3))
W3 = W3[W3 != 0]
mean3 = torch.mean(W3)

plot_dist_distribution(W1, 'CLIN - average distance between two points'+format(mean1), mean1)
plot_dist_distribution(W2, 'RNA - average distance between two points'+format(mean2), mean2)
plot_dist_distribution(W3, 'CNA - average distance between two points'+format(mean3), mean3)

plot_dist_distribution(torch.exp(-W1), 'CLIN - similarity score exp(-dist)'+ format(torch.exp(-mean1)), torch.exp(-mean1))
plot_dist_distribution(torch.exp(-W2), 'RNA - similarity score exp(-dist)' + format(torch.exp(-mean2)), torch.exp(-mean2))
plot_dist_distribution(torch.exp(-W3), 'CNA - similarity score exp(-dist)' + format(torch.exp(-mean3)), torch.exp(-mean3))
'''

dataset12 = torch.cat([torch.from_numpy(dataset1), torch.from_numpy(dataset2)], dim=1)
dataset23 = torch.cat([torch.from_numpy(dataset2), torch.from_numpy(dataset3)], dim=1)
dataset31 = torch.cat([torch.from_numpy(dataset3), torch.from_numpy(dataset1)], dim=1)

W12 = torch.cdist(dataset12, dataset12)
W23 = torch.cdist(dataset23, dataset23)
W31 = torch.cdist(dataset31, dataset31)

mean12 = torch.mean(W12)
mean23 = torch.mean(W23)
mean31 = torch.mean(W31)

plot_dist_distribution(W12, 'CLIN + RNA - average distance between two points'+format(mean12), mean12)
plot_dist_distribution(W23, 'RNA + CNA - average distance between two points'+format(mean23), mean23)
plot_dist_distribution(W31, 'CNA + CLIN - average distance between two points'+format(mean31), mean31)
