import torch
from torch import nn


class CAT(nn.Module):
    def __init__(self, n_h):
        super(CAT, self).__init__()
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
