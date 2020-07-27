import pdb
import numpy as np
from itertools import permutations

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F


class TGT(nn.Module):
    """
    This class is a simple wrapper over the deltas in TGT.

    Paper: https://arxiv.org/pdf/2003.01640.pdf
    Authors' code: https://github.com/GDPlumb/ELDR

    The reason for making deltas into a nn.Module instead of just storing them as a
    matrix is to autodiff and pytorch optimizers to optimize them in a fully
    "pytorch-oriented" way. As opposed to getting gradients and manually updating the
    deltas.
    TODO: add mean init support
    """
    def __init__(
        self,
        input_ndim: int,
        output_ndim: int,
        num_clusters: int,
        init_mode: str = "xavier_uniform",
    ):
        super(TGT, self).__init__()

        # set deltas as weights/parameters of a module
        self.deltas = nn.ParameterList(
            torch.nn.Parameter(torch.empty(size=(1, input_ndim), requires_grad=True))
            for _ in range(num_clusters-1)
        )

        # initialize
        if init_mode == "xavier_uniform":
            for i in range(len(self.deltas)):
                nn.init.xavier_uniform_(self.deltas[i])
        elif init_mode == "zero":
            for i in range(len(self.deltas)):
                nn.init.constant_(self.deltas[i], 0)
        elif init_mode == "mean":
            raise NotImplementedError("This is on the todo list. Not implemented yet")
        else:
            raise ValueError(f"Unknown value for parameter `init_mode`. \
                Got {init_mode}, expected one of `zero`, `mean`, or `xavier_unform`")

    def forward(self, initial_ci, target_ci):
        if initial_ci == 0:
            delta = self.deltas[target_ci - 1]
        elif target_ci == 0:
            delta = -1.0 * self.deltas[initial_ci - 1]
        else:
            delta = self.deltas[target_ci - 1] + self.deltas[initial_ci - 1] * -1.0
        return delta


def transitive_global_translations_experimental(
    dim_reducer: nn.Module, 
    low_dim_group_means: torch.Tensor,
    high_dim_group_means: torch.Tensor,
    consecutive_steps : int = 10,
    learning_rate: float = 0.001,
    l1_lambda: float = 0.5,
    tol: float = 0.0001,
    min_epochs: int = 2000,
    stopping_epochs: int = 2000,
    verbose: bool = False,
) -> torch.Tensor:
    """Implementation of tgt using deltas as a pytorch feed forward module
    and using adam optimizer instead of manually updating gradients according
    to the paper 

    Args:
        dim_reducer (nn.Module): VAE that maps high dimensional data to low dim
            space. This must be a differentiable object for tgt to be applied
        low_dim_group_means (torch.Tensor): (c, m) shaped tensor of mean of each group
            in low dimensional space
        high_dim_group_means (torch.Tensor): (c, d) shaped tensor of mean of each group
            in high dimensional space
        consecutive_steps (int, optional): number of consecutive updates to deltas of
            a give pair of initial and target means. Defaults to 10.
        learning_rate (float, optional): Defaults to 0.001.
        l1_lambda (float, optional): Defaults to 0.5.
        tol (float, optional): [description]. Defaults to 0.0001.
        min_epochs (int, optional): Defaults to 2000.
        stopping_epochs (int, optional): Defaults to 2000.
        verbose (bool, optional): Defaults to False.

    Returns:
        torch.Tensor: (c-1, d) shaped tensor. Each column is basis explanation (delta)
            corresponding wrt reference group
    """
    # how many groups are there
    num_clusters = high_dim_group_means.shape[0]

    # dimensions
    input_ndim = high_dim_group_means.shape[1]
    output_ndim = low_dim_group_means.shape[1]

    # init deltas module
    tgt_deltas = TGT(
        input_ndim=input_ndim, 
        output_ndim=output_ndim, 
        num_clusters=num_clusters,
    )

    # cluster id permutations
    cluster_id_perms = list(permutations(range(num_clusters), r=2))

    epoch = 0
    perm_idx = 0
    best_epoch = 0
    best_loss = np.inf
    best_deltas = None
    optimizer = optim.Adam(tgt_deltas.parameters(), lr=learning_rate)
    while True:
        optimizer.zero_grad()

        # update initial and target cluster indices every `consecutive_steps`
        if epoch % consecutive_steps == 0:
            # FIXME: remove this if doesnt work
            # initial_ci, target_ci = np.random.choice(num_clusters, 2, replace=False)
            initial_ci, target_ci = cluster_id_perms[perm_idx]
            perm_idx += 1
            if perm_idx == len(cluster_id_perms):
                perm_idx = 0

        # initial point in high dim and target in low dim
        initial_pt = high_dim_group_means[initial_ci].unsqueeze(dim=0)
        target_pt = low_dim_group_means[target_ci].unsqueeze(dim=0)

        # get the current delta
        delta = tgt_deltas(initial_ci=initial_ci, target_ci=target_ci)

        # embed initial + delta
        mu, logvar = dim_reducer.encode(initial_pt + delta)
        recon_pt = dim_reducer.reparameterize(mu, logvar)

        # compute loss
        loss_target = F.mse_loss(recon_pt, target_pt)
        loss_global = l1_lambda * torch.norm(delta, p=1)
        loss = loss_target + loss_global
        loss.backward()

        # save best loss and deltas
        if loss.data < (best_loss - tol):
            best_epoch = epoch
            best_loss = loss.data
            # get parameter values; gradients not needed
            with torch.no_grad():
                best_state_dict = tgt_deltas.state_dict()

        # update corresponding deltas
        optimizer.step()

        # stopping condition
        if epoch - best_epoch > stopping_epochs and epoch > min_epochs:
            break

        if verbose:
            print('Epoch: {:3d} Loss: {:5f}'.format(epoch, loss.data), end="\r")

        epoch += 1

    tgt_deltas.load_state_dict(best_state_dict)
    return tgt_deltas


def transitive_global_translations(
    model: nn.Module,
    low_dim_group_means: torch.Tensor,
    high_dim_group_means: torch.Tensor,
    init_mode: str = "zero",
    consecutive_steps : int = 10,
    learning_rate: float = 0.0005,
    discount: float = 0.99,
    l1_lambda: float = 0.5,
    clip_val: float = 5,
    tol: float = 0.0001,
    min_epochs: int = 2000,
    stopping_epochs: int = 2000,
    verbose: bool = False,
) -> torch.Tensor:
    """Implementation of tgt as done by authors here:
        https://github.com/GDPlumb/ELDR/blob/master/Code/explain_cs.py
    But using pytorch instead of tensorflow.

    Args:
        model (nn.Module): VAE that maps high dimensional data to low dim
            space. This must be a differentiable object for tgt to be applied
        low_dim_group_means (torch.Tensor): (c, m) shaped tensor of mean of each group
            in low dimensional space
        high_dim_group_means (torch.Tensor): (c. d) shaped tensor of mean of each group
            in high dimensional space
        init_mode (str, optional): method for initializing deltas. Must be one of
        "zero", "mean". Defaults to "zero".
        consecutive_steps (int, optional): number of consecutive updates to deltas of
            a give pair of initial and target means. Defaults to 10.
        learning_rate (float, optional): Defaults to 0.001.
        discount (float, optional): Defaults to 0.99.
        l1_lambda (float, optional): Defaults to 0.5.
        tol (float, optional): [description]. Defaults to 0.0001.
        min_epochs (int, optional): Defaults to 2000.
        stopping_epochs (int, optional): Defaults to 2000.
        verbose (bool, optional): Defaults to False.

    Returns:
        torch.Tensor: (c-1, d) shaped tensor. Each column is basis explanation (delta)
            corresponding wrt reference group
    """
    # how many groups are there
    num_clusters = high_dim_group_means.shape[0]

    # dimensions
    input_ndim = high_dim_group_means.shape[1]
    output_ndim = low_dim_group_means.shape[1]

    # init deltas with respect to reference group
    # row i is the explanation for "Cluster 0 to Cluster i + 1"
    # FIXME: try using a list of delta vectors instead of matrix. maybe that will give non null grad
    deltas = [
        torch.zeros(size=(1, input_ndim), requires_grad=True)
        for _ in range(num_clusters-1)
    ]

    if init_mode == "mean":
        # TODO: rm for loop
        for i in range(1, num_clusters):
            deltas[i - 1] = high_dim_group_means[i, :] - high_dim_group_means[0, :]
    elif init_mode == "zero":
        pass
    else:
        raise ValueError(f"Unexpected value for `init_mode` parameter: {init_mode}")

    converged = False
    ema = None
    epoch = 0
    best_epoch = 0
    best_loss = np.inf
    best_deltas = None
    while not converged:
        # update initial and target cluster indices every `consecutive_steps`
        if epoch % consecutive_steps == 0:
            initial_ci, target_ci = np.random.choice(num_clusters, 2, replace=False)

        # initial point in high dim and target in low dim
        initial_pt = high_dim_group_means[initial_ci].unsqueeze(dim=0)
        target_pt = low_dim_group_means[target_ci].unsqueeze(dim=0)

        # extract delta vector according to `initial_ci` and `target_ci` indices
        if initial_ci == 0:
            curr_delta = deltas[target_ci - 1]
        elif target_ci == 0:
            curr_delta = -1.0 * deltas[initial_ci - 1]
        else:
            curr_delta = deltas[target_ci - 1] + deltas[initial_ci - 1] * -1.0 

        # calculate losses
        mu, logvar = model.encode(initial_pt + curr_delta)
        recon_pt = model.reparameterize(mu, logvar)
        loss_target = F.mse_loss(recon_pt, target_pt)
        loss_global = torch.norm(curr_delta, p=1)
        loss = loss_target + loss_global
        loss.backward()

        # apply discount
        if epoch == 0:
            ema = loss
        else:
            ema = discount * ema + (1 - discount) * loss

        # save best
        if ema < best_loss - tol:
            best_epoch = epoch
            best_loss = ema
            best_deltas = deltas

        # update the corresponding delta
        if initial_ci == 0:
            deltas[target_ci - 1].data -= learning_rate * deltas[target_ci - 1].grad.clamp(min=-1*clip_val, max=clip_val)
        elif target_ci == 0:
            deltas[initial_ci - 1].data += learning_rate * deltas[initial_ci - 1].grad.clamp(min=-1*clip_val, max=clip_val)
        else:
            deltas[initial_ci - 1].data += learning_rate * 0.5 * deltas[initial_ci - 1].grad.clamp(min=-1*clip_val, max=clip_val)
            deltas[target_ci - 1].data -= learning_rate * 0.5 * deltas[target_ci - 1].grad.clamp(min=-1*clip_val, max=clip_val)

        # stopping condition
        if epoch - best_epoch > stopping_epochs and epoch > min_epochs:
            break
        
        if verbose:
            print('Epoch: {:3d} Loss: {:5f}'.format(epoch, ema), end="\r")

        epoch += 1
        
    return best_deltas