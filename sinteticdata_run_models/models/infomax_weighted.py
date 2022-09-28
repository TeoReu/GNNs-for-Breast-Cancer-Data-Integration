import torch
import torch.nn as nn
from torch_geometric.nn import GCN

from models.infomax import AvgReadout, Discriminator

def corruption(x1, edge_index1, edge_weight1, x2, edge_index2, edge_weight2):
    rand = torch.randperm(x1.size(0))
    return x1[rand], edge_index1, edge_weight1, x2[rand], edge_index2, edge_weight2

class DGI_CAT(nn.Module):
    def __init__(self, n_in1, n_in2, n_h, encoder):
        super(DGI_CAT, self).__init__()
        self.encoder = encoder(n_in1, n_in2, n_h)
        self.activation = nn.PReLU()
        self.read = AvgReadout()

        self.sigm = nn.Sigmoid()

        self.disc = Discriminator(n_h)

    def forward(self, seq1a, edge_index1a, edge_weight1a, seq1b, edge_index1b, edge_weight1b, msk, samp_bias1, samp_bias2):
        h_1 = self.encoder(seq1a, edge_index1a, edge_weight1a, seq1b, edge_index1b, edge_weight1b)

        c = self.read(h_1, msk)
        c = self.sigm(c)
        seq2a, edge_index2a, edge_weight2a, seq2b, edge_index2b, edge_weight2b = corruption(seq1a, edge_index1a, edge_weight1a, seq1b, edge_index1b, edge_weight1b)
        h_2 = self.encoder( seq2a, edge_index2a, edge_weight2a, seq2b, edge_index2b, edge_weight2b)

        ret = self.disc(c, h_1, h_2, samp_bias1, samp_bias2)

        return ret

    # Detach the return variables
    def embed(self, seq1a, edge_index1a, edge_weight1a, seq1b, edge_index1b, edge_weight1b, msk):
        h_1 = self.encoder(seq1a, edge_index1a, edge_weight1a, seq1b, edge_index1b, edge_weight1b)
        c = self.read(h_1, msk)

        return h_1.detach(), c.detach()

