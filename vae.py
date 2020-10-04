# NOTE: this code is currently copypasta'd from pytorch official examples repo at
# https://github.com/pytorch/examples/blob/master/vae/main.py
# In the future, this could probably be added as a submodule
import pdb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(
        self,
        input_ndim,
        output_ndim,
        encode_layer_sizes=(128, 64, 32), 
        decode_layer_sizes=(32, 64, 128),
    ):
        super(VAE, self).__init__()

        # create encoding layers
        self.encode_layers = nn.ModuleList()
        for li in range(len(encode_layer_sizes)):
            if li == 0:
                self.encode_layers.append(nn.Linear(input_ndim, encode_layer_sizes[li]))
            else:
                self.encode_layers.append(nn.Linear(encode_layer_sizes[li-1], encode_layer_sizes[li]))
        self.mu_encode_layer = nn.Linear(encode_layer_sizes[-1], output_ndim)
        self.logvar_encode_layer = nn.Linear(encode_layer_sizes[-1], output_ndim)

        # create decoding layers
        self.decode_layers = nn.ModuleList()
        for li in range(len(decode_layer_sizes)):
            if li == 0:
                self.decode_layers.append(nn.Linear(output_ndim, decode_layer_sizes[li]))
            else:
                self.decode_layers.append(nn.Linear(decode_layer_sizes[li-1], decode_layer_sizes[li]))
        # self.decode_layers.append(nn.Linear(decode_layer_sizes[-1], input_ndim))
        self.mu_decode_layer = nn.Linear(decode_layer_sizes[-1], input_ndim)
        self.logvar_decode_layer = nn.Linear(decode_layer_sizes[-1], input_ndim)

        # self.fc1 = nn.Linear(input_ndim, 64)
        # self.fc21 = nn.Linear(64, output_ndim)
        # self.fc22 = nn.Linear(64, output_ndim)
        # self.fc3 = nn.Linear(output_ndim, 64)
        # self.fc4 = nn.Linear(64, input_ndim)

    def encode(self, x):
        # h1 = F.relu(self.fc1(x))
        # return self.fc21(h1), self.fc22(h1)
        for li,layer in enumerate(self.encode_layers):
            if li == 0:
                h = F.relu(layer(x))
            else:
                h = F.relu(layer(h))
        return self.mu_encode_layer(h), self.logvar_encode_layer(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        # h3 = F.relu(self.fc3(z))
        # return torch.sigmoid(self.fc4(h3))
        for li,layer in enumerate(self.decode_layers):
            if li == 0:
                h = F.relu(layer(z))
            else:
                h = F.relu(layer(h))
        return self.mu_decode_layer(h), self.logvar_decode_layer(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        mu_x, logvar_x = self.decode(z)
        return mu_x, logvar_x, mu, logvar


def loss_function(x, mu_x, logvar_x, mu, logvar):
    # see Appendix C.2 from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    nll_gaussian = torch.sum(
            0.5 * np.log(2 * np.pi)
            + 0.5 * logvar_x
            + 0.5 * (x - mu_x)**2 / logvar_x.exp() 
        )

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return nll_gaussian + kld
