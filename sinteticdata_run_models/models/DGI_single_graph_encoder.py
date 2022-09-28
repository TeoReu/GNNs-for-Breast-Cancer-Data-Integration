import torch
from torch_geometric import nn
from torch_geometric.nn import GCNConv


def corruption(x, edge_index, edge_weight):
    return x[torch.randperm(x.size(0))], edge_index, edge_weight


def unweighted_corruption(x, edge_index):
    return x[torch.randperm(x.size(0))], edge_index


class Encoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv = GCNConv(in_channels, hidden_channels, cached=True, normalize=False)
        self.prelu = torch.nn.PReLU(hidden_channels)

    def forward(self, x, edge_index, edge_weight):
        x = self.conv(x, edge_index, edge_weight)
        x = self.prelu(x)
        return x


class UnweightedEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv = GCNConv(in_channels, hidden_channels, cached=True, normalize=False)
        self.prelu = torch.nn.PReLU(hidden_channels)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = self.prelu(x)
        return x
