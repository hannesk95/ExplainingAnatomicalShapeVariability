import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add

from conv import SpiralConv


def Pool(x, trans, dim=1):
    row, col = trans._indices()
    value = trans._values().unsqueeze(-1)
    out = torch.index_select(x, dim, col) * value
    out = scatter_add(out, row, dim, dim_size=trans.size(0))
    return out


class SpiralEnblock(nn.Module):
    def __init__(self, in_channels, out_channels, indices):
        super(SpiralEnblock, self).__init__()
        self.conv = SpiralConv(in_channels, out_channels, indices)
        self.reset_parameters()

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, down_transform):
        out = F.elu(self.conv(x))
        out = Pool(out, down_transform)
        return out


class SpiralDeblock(nn.Module):
    def __init__(self, in_channels, out_channels, indices):
        super(SpiralDeblock, self).__init__()
        self.conv = SpiralConv(in_channels, out_channels, indices)
        self.reset_parameters()

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, up_transform):
        out = Pool(x, up_transform)
        out = F.elu(self.conv(out))
        return out


class AE(nn.Module):
    def __init__(self, in_channels, out_channels, latent_channels,
                 spiral_indices, down_transform, up_transform, training=True):
        super(AE, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.latent_channels = latent_channels
        self.latent_channels = latent_channels
        self.spiral_indices = spiral_indices
        self.down_transform = down_transform
        self.up_transform = up_transform
        self.num_vert = self.down_transform[-1].size(0)
        self.training = training

        # encoder
        self.en_layers = nn.ModuleList()
        for idx in range(len(out_channels)):
            if idx == 0:
                self.en_layers.append(
                    SpiralEnblock(in_channels, out_channels[idx],
                                  self.spiral_indices[idx]))
            else:
                self.en_layers.append(
                    SpiralEnblock(out_channels[idx - 1], out_channels[idx],
                                  self.spiral_indices[idx]))
        self.en_layers.append(nn.Linear(self.num_vert * out_channels[-1], 2*latent_channels))
        #self.en_layers.append(nn.Linear(8*latent_channels, 2*latent_channels))

        # decoder
        self.de_layers = nn.ModuleList()
        self.de_layers.append(
            nn.Linear(latent_channels, self.num_vert * out_channels[-1]))
        for idx in range(len(out_channels)):
            if idx == 0:
                self.de_layers.append(
                    SpiralDeblock(out_channels[-idx - 1],
                                  out_channels[-idx - 1],
                                  self.spiral_indices[-idx - 1]))
            else:
                self.de_layers.append(
                    SpiralDeblock(out_channels[-idx], out_channels[-idx - 1],
                                  self.spiral_indices[-idx - 1]))
        self.de_layers.append(
            SpiralConv(out_channels[0], in_channels, self.spiral_indices[0]))

        self.reset_parameters()

	    # Excitation
        self.reg_sq = nn.Sequential(
            nn.Linear(1, 8),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(8, 8),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(8, 1))

    def reset_parameters(self):
        for name, param in self.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            else:
                nn.init.xavier_uniform_(param)

    def encoder(self, x):
        for i, layer in enumerate(self.en_layers):
            if i != len(self.en_layers) - 1:
                x = layer(x, self.down_transform[i])
            else:
                x = x.view(-1, layer.weight.size(1))
                x = layer(x)
        
        mu = x[:, :self.latent_channels]
        log_var = x[:, self.latent_channels:]

        return mu, log_var

    def decoder(self, x):
        num_layers = len(self.de_layers)
        num_features = num_layers - 2
        for i, layer in enumerate(self.de_layers):
            if i == 0:
                x = layer(x)
                x = x.view(-1, self.num_vert, self.out_channels[-1])
            elif i != num_layers - 1:
                x = layer(x, self.up_transform[num_features - i])
            else:
                x = layer(x)
        return x

    def reparameterize(self, mu, log_var):
        if self.training:
            #log_var = log_var.clamp(max=10)
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            sample = mu + eps * std
        else:
            sample = mu

        return sample

    def reg(self, z):
        z = torch.split(z, 1, 1)[0]
        return self.reg_sq(z)

    def forward(self, x, *indices):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        out = self.decoder(z)
        return out, mu, log_var, self.reg(z)

# Inhibition
class Regressor(nn.Module):
    def __init__(self, n_vae_dis=16):
        super(Regressor, self).__init__()

        self.reg_sq = nn.Sequential(
            nn.Linear(n_vae_dis - 1, 8),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(8, 8),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        return self.reg_sq(x)
