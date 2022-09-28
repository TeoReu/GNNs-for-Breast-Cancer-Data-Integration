import torch
from torch import nn


class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        c_x = c
        c_x = c_x.expand_as(h_pl)

        sc_1 = torch.squeeze(self.f_k(h_pl, c_x), 1)
        sc_2 = torch.squeeze(self.f_k(h_mi, c_x), 1)

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1, sc_2), 0)

        return logits


class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq, msk):
        if msk is None:
            return torch.mean(seq, 0)
        else:
            msk = torch.unsqueeze(msk, -1)
            return torch.sum(seq * msk, 1) / torch.sum(msk)


def corruption_2_graphs(x1, edge_index1, edge_weight1, x2, edge_index2, edge_weight2):
    rand = torch.randperm(x1.size(0))
    return x1[rand], edge_index1, edge_weight1, x2[rand], edge_index2, edge_weight2


class DGI2Graphs(nn.Module):
    def __init__(self, n_in1, n_in2, n_h, encoder):
        super(DGI2Graphs, self).__init__()
        self.encoder = encoder(n_in1, n_in2, n_h)
        self.activation = nn.PReLU()
        self.read = AvgReadout()

        self.sigm = nn.Sigmoid()

        self.disc = Discriminator(n_h)

    def forward(self, seq1a, edge_index1a, edge_weight1a, seq1b, edge_index1b, edge_weight1b, msk, samp_bias1,
                samp_bias2):
        h_1 = self.encoder(seq1a, edge_index1a, edge_weight1a, seq1b, edge_index1b, edge_weight1b)

        c = self.read(h_1, msk)
        c = self.sigm(c)
        seq2a, edge_index2a, edge_weight2a, seq2b, edge_index2b, edge_weight2b = corruption_2_graphs(seq1a,
                                                                                                     edge_index1a,
                                                                                                     edge_weight1a,
                                                                                                     seq1b,
                                                                                                     edge_index1b,
                                                                                                     edge_weight1b)
        h_2 = self.encoder(seq2a, edge_index2a, edge_weight2a, seq2b, edge_index2b, edge_weight2b)

        ret = self.disc(c, h_1, h_2, samp_bias1, samp_bias2)

        return ret

    # Detach the return variables
    def embed(self, seq1a, edge_index1a, edge_weight1a, seq1b, edge_index1b, edge_weight1b, msk):
        h_1 = self.encoder(seq1a, edge_index1a, edge_weight1a, seq1b, edge_index1b, edge_weight1b)
        c = self.read(h_1, msk)

        return h_1.detach(), c.detach()

class DGI(nn.Module):
    def __init__(self, n_in, n_h):
        super(DGI, self).__init__()
        self.gcn = GCN(n_in, n_h, num_layers=3, act=nn.PReLU())
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