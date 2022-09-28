import torch
from torch import nn
from torch_geometric.nn import GCNConv

from models.cat import CAT


class DoubleLayeredEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv = GCNConv(in_channels, 2 * hidden_channels)
        self.prelu = nn.PReLU(2 * hidden_channels)
        self.conv2 = GCNConv(2 * hidden_channels, hidden_channels)
        self.prelu2 = nn.PReLU(hidden_channels)
        # self.cat = CAT3(hidden_channels)

    def forward(self, x, edge_index, edge_weight, edge_type):
        x = self.conv(x, edge_index, edge_weight)
        x = self.prelu(x)
        x = self.conv2(x, edge_index, edge_type.float())
        x = self.prelu2(x)

        n1, n2 = torch.split(x, int(x.size(0) / 2))

        return (n1 + n2) / 2


class DoubleLayeredEncoderCAT(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv = GCNConv(in_channels, hidden_channels, cached=True, normalize=False)
        self.prelu = nn.PReLU(hidden_channels)
        self.cat = CAT(hidden_channels)

    def forward(self, x, edge_index, edge_weight):
        x = self.conv(x, edge_index, edge_weight)
        x = self.prelu(x)

        n1, n2 = torch.split(x, int(x.size(0) / 2))

        return self.cat(n1,n2)