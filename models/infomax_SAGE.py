import torch
import torch.nn as nn
from torch_geometric.nn import GCN, SAGEConv

from models.infomax import DGI, AvgReadout, Discriminator


class DGI_SAGE(nn.Module):
    def __init__(self, n_in, n_h):
        super(DGI, self).__init__()
        self.gcn = SAGEConv(n_in, n_h, num_layers=3, act=nn.PReLU())
        self.activation = nn.PReLU()
        self.read = AvgReadout()

        self.sigm = nn.Sigmoid()

        self.disc = Discriminator(n_h)

    def forward(self, seq1, seq2, edge_index, msk, samp_bias1, samp_bias2):
        h_1 = self.gcn(seq1, edge_index)

        c = self.read(h_1, msk)
        c = self.sigm(c)

        h_2 = self.gcn(seq2, edge_index)

        ret = self.disc(c, h_1, h_2, samp_bias1, samp_bias2)

        return ret

    # Detach the return variables
    def embed(self, seq, edge_index, msk):
        h_1 = self.gcn(seq, edge_index)
        c = self.read(h_1, msk)

        return h_1.detach(), c.detach()
