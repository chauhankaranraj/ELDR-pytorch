# NOTE: this code is currently copypasta'd from pytorch official examples repo at
# https://github.com/pytorch/examples/blob/master/vae/main.py
# In the future, this could probably be added as a submodule
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self, input_ndim, output_ndim):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(input_ndim, 64)
        self.fc21 = nn.Linear(64, output_ndim)
        self.fc22 = nn.Linear(64, output_ndim)
        self.fc3 = nn.Linear(output_ndim, 64)
        self.fc4 = nn.Linear(64, input_ndim)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


# Reconstruction + KL divergence losses summed over all elements and batch

def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # return BCE + KLD

    # FIXME: bce + kld doesnt seem to converge to a meaningful representation
    # for some reason.
    # Using smooth l1 loss seems to fix this but we would be completely omitting
    # the kld loss which doesnt sound like a good idea. Nevertheless, for we'll
    # use this purely for convergence reasons for iris dataset
    return F.smooth_l1_loss(recon_x, x, reduction='sum')
