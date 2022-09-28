import torch
from torch import nn
from torch_geometric.nn import GCNConv


class CAT2(nn.Module):
    def __init__(self, n_h):
        super(CAT2, self).__init__()
        self.dense = nn.Sequential(nn.Linear(2 * n_h, n_h), nn.ReLU())

    def forward(self, h_1, h_2):
        output = torch.cat((h_1, h_2), dim=1)
        output = self.dense(output)
        return output


class CAT3(nn.Module):
    def __init__(self, n_h):
        super(CAT3, self).__init__()
        self.dense = nn.Sequential(nn.Linear(3 * n_h, n_h), nn.ReLU())

    def forward(self, h_1, h_2, h_3):
        output = torch.cat((h_1, h_2, h_3), dim=1)
        output = self.dense(output)
        return output


class Weighted2GraphEncoderSUM(nn.Module):
    def __init__(self, in_channels_1, in_channels_2, hidden_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels_1, int(hidden_channels), cached=False, normalize=True)
        self.conv2 = GCNConv(in_channels_2, int(hidden_channels), cached=False, normalize=True)

        self.prelu1 = nn.PReLU(hidden_channels)
        self.prelu2 = nn.PReLU(hidden_channels)
        # self.cat = CAT(hidden_channels)
        # self.dense = CAT(hidden_channels)

    def forward(self, x1, edge_index1, edge_weight1, x2, edge_index2, edge_weight2):
        x1 = self.conv1(x1, edge_index1, edge_weight1)
        x1 = self.prelu1(x1)

        x2 = self.conv2(x2, edge_index2, edge_weight2)
        x2 = self.prelu2(x2)

        return (x1 + x2) / 2


class Weighted2GraphEncoderCAT(nn.Module):
    def __init__(self, in_channels_1, in_channels_2, hidden_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels_1, int(hidden_channels), cached=False, normalize=True)
        self.conv2 = GCNConv(in_channels_2, int(hidden_channels), cached=False, normalize=True)

        self.prelu1 = nn.PReLU(hidden_channels)
        self.prelu2 = nn.PReLU(hidden_channels)
        self.cat = CAT2(hidden_channels)

    def forward(self, x1, edge_index1, edge_weight1, x2, edge_index2, edge_weight2):
        x1 = self.conv1(x1, edge_index1, edge_weight1)
        x1 = self.prelu1(x1)

        x2 = self.conv2(x2, edge_index2, edge_weight2)
        x2 = self.prelu2(x2)

        return self.cat(x1, x2)


class DoubleLayeredGraphEncoderSUM(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv = GCNConv(in_channels, hidden_channels, cached=True, normalize=False)
        self.prelu = nn.PReLU(hidden_channels)
        # self.cat = CAT3(hidden_channels)

    def forward(self, x, edge_index, edge_weight):
        x = self.conv(x, edge_index, edge_weight)
        x = self.prelu(x)

        n1, n2 = torch.split(x, int(x.size(0) / 2))

        return (n1 + n2) / 2


class DoubleLayeredGraphEncoderCAT(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv = GCNConv(in_channels, hidden_channels, cached=True, normalize=False)
        self.prelu = nn.PReLU(hidden_channels)
        self.cat = CAT2(hidden_channels)

    def forward(self, x, edge_index, edge_weight):
        x = self.conv(x, edge_index, edge_weight)
        x = self.prelu(x)

        n1, n2 = torch.split(x, int(x.size(0) / 2))

        return self.cat(n1, n2)


class OneLayeredGraphEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv = GCNConv(in_channels, hidden_channels, cached=True, normalize=False)
        self.prelu = torch.nn.PReLU(hidden_channels)

    def forward(self, x, edge_index, edge_weight):
        x = self.conv(x, edge_index, edge_weight)
        x = self.prelu(x)
        return x


class UnweightedOneLayeredGraphEncoderEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv = GCNConv(in_channels, hidden_channels, cached=True, normalize=False)
        self.prelu = torch.nn.PReLU(hidden_channels)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = self.prelu(x)
        return x
