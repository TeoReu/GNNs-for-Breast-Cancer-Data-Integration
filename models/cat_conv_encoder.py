import torch
from torch_geometric.nn import GCNConv
from torch import nn

from models.cat import CAT


class Encoder(nn.Module):
    def __init__(self, in_channels_1, in_channels_2, hidden_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels_1, int(hidden_channels), cached=False, normalize=True)
        self.conv2 = GCNConv(in_channels_2, int(hidden_channels), cached=False, normalize=True)

        self.prelu1 = nn.PReLU(hidden_channels)
        self.prelu2 = nn.PReLU(hidden_channels)
        #self.cat = CAT(hidden_channels)
        # self.dense = CAT(hidden_channels)

    def forward(self, x1, edge_index1, edge_weight1, x2, edge_index2, edge_weight2):
        x1 = self.conv1(x1, edge_index1, edge_weight1)
        x1 = self.prelu1(x1)

        x2 = self.conv2(x2, edge_index2, edge_weight2)
        x2 = self.prelu2(x2)

        return (x1+x2)/2

