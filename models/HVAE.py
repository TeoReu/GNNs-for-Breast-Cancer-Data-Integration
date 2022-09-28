import torch
from torch import nn
import torch.nn.functional as F


class catVAE(nn.Module):
    def __init__(self, feature_dim, ds, ls):
        super(catVAE, self).__init__()
        self.enc = nn.Linear(feature_dim, ds)
        self.BN = nn.BatchNorm1d(ds)
        self.mean = nn.Linear(ds, ds // 2)
        self.std = nn.Linear(ds, ds // 2)
        nn.init.zeros_(self.std.weight)

        self.dec1 = nn.Linear(ds // 2, ds)  # sigmoid activation
        self.BN2 = nn.BatchNorm1d(ds)
        self.D = nn.Dropout(0.2)

        self.dec2 = nn.Linear(ds, feature_dim)
        self.sigm = nn.Sigmoid()

    def encoder(self, x):
        x = nn.ELU()(self.enc(x))
        x = self.BN(x)
        z_mean = self.mean(x)
        z_std = self.std(x)

        return z_mean, z_std

    def reparametrize(self, mean, std):
        s = torch.exp(std / 2)
        eps = torch.randn_like(s)
        return mean + std * eps

    def decoder(self, z):
        x = self.dec1(z)
        x = self.BN2(x)
        x = self.D(x)
        x = self.sigm(self.dec2(x))

        return x

    def forward(self, x):
        mean, std = self.encoder(x)
        z = self.reparametrize(mean, std)
        out = self.decoder(z)
        return out, z, mean, std

    @torch.no_grad()
    def encode(self, x):
        x = nn.ELU()(self.enc(x))
        x = self.BN(x)

        z_mean = self.mean(x)
        z_std = self.std(x)
        z = self.reparametrize(z_mean, z_std)
        return z

    def reconstruction_loss(self, inp, out):
        loss = nn.BCELoss()
        return loss(inp, out)

    def kl_divergence(self, mu, log_var):
        kl = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return kl


class numVAE(nn.Module):
    def __init__(self, feature_dim, ds, ls):
        super(numVAE, self).__init__()
        self.enc = nn.Linear(feature_dim, ds)
        self.BN = nn.BatchNorm1d(ds)
        self.mean = nn.Linear(ds, ds // 2)
        self.std = nn.Linear(ds, ds // 2)
        nn.init.zeros_(self.std.weight)

        self.dec1 = nn.Linear(ds // 2, ds)  # sigmoid activation
        self.BN2 = nn.BatchNorm1d(ds)
        self.D = nn.Dropout(0.2)

        self.dec2 = nn.Linear(ds, feature_dim)
        self.sigm = nn.Sigmoid()

    def encoder(self, x):
        x = nn.ELU()(self.enc(x))
        x = self.BN(x)
        z_mean = self.mean(x)
        z_std = self.std(x)

        return z_mean, z_std

    def reparametrize(self, mean, std):
        s = torch.exp(std / 2)
        eps = torch.randn_like(s)
        return mean + std * eps

    def decoder(self, z):
        x = self.dec1(z)
        x = self.BN2(x)
        x = self.D(x)
        x = self.dec2(x)

        return x

    def forward(self, x):
        mean, std = self.encoder(x)
        z = self.reparametrize(mean, std)
        out = self.decoder(z)
        return out, z, mean, std

    @torch.no_grad()
    def encode(self, x):
        x = nn.ELU()(self.enc(x))
        x = self.BN(x)
        z_mean = self.mean(x)
        z_std = self.std(x)
        z = self.reparametrize(z_mean, z_std)
        return z

    def reconstruction_loss(self, inp, out):
        loss = nn.MSELoss(size_average=inp.shape[0], reduction='mean')
        return loss(inp, out)

    def kl_divergence(self, mu, log_var):
        kl = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return kl


class HVAE(nn.Module):
    def __init__(self, ds, ls):
        super(HVAE, self).__init__()

        self.enc = nn.Linear(ds, ds)
        self.BN = nn.BatchNorm1d(ds)
        self.mean = nn.Linear(ds, ls)
        self.std = nn.Linear(ds, ls)
        nn.init.zeros_(self.std.weight)

        self.dec1 = nn.Linear(ls, ds)
        self.BN2 = nn.BatchNorm1d(ds)
        self.D = nn.Dropout(0.2)
        self.sigm = nn.Sigmoid()

        self.BN3 = nn.BatchNorm1d(ds)
        self.dec2 = nn.Linear(ds, ds)

    def encoder(self, x):
        x = nn.ELU()(self.enc(x))
        x = self.BN(x)
        z_mean = self.mean(x)
        z_std = self.std(x)

        return z_mean, z_std

    def reparametrize(self, mean, std):
        s = torch.exp(std / 2)
        eps = torch.randn_like(s)
        return mean + std * eps

    def decoder(self, z):
        x = nn.ELU()(self.dec1(z))
        x = self.BN2(x)
        x = self.D(x)
        x = self.dec2(x)

        return x

    def forward(self, x):
        mean, std = self.encoder(x)
        z = self.reparametrize(mean, std)
        out = self.decoder(z)
        return out, z, mean, std

    @torch.no_grad()
    def encode(self, x):
        x = nn.ELU()(self.enc(x))
        x = self.BN(x)
        z_mean = self.mean(x)
        z_std = self.std(x)
        z = self.reparametrize(z_mean, z_std)
        return z

    def reconstruction_loss(self, inp, out):
        loss = nn.MSELoss(size_average=inp.shape[0], reduction='mean')
        return loss(inp, out)

    def kl_divergence(self, mu, log_var):
        kl = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return kl
