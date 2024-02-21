import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as nnf
import numpy as np

class FullGatherLayer(torch.autograd.Function):
    """
    Gather tensors from all process and support backward propagation
    for the gradients across processes.

    from https://github.com/facebookresearch/vicreg/
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]

def batch_all_gather(x):
    """
    from https://github.com/facebookresearch/vicreg/
    """
    x_list = FullGatherLayer.apply(x)
    return torch.cat(x_list, dim=0)

def off_diagonal(x):
    """
    look at this thing for getting the off diagonal
    elements of a square matrix
    from https://github.com/facebookresearch/vicreg/
    """
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def build_projector(in_features: int, layer_sizes: list[int], norm=nn.BatchNorm1d):
    # prepend the input size (i.e. the pooled fcn output)
    layer_sizes = [in_features] + layer_sizes
    nlayers = len(layer_sizes)
    layers = []
    for idx in range(1, nlayers-1):
        layers.append(nn.Linear(layer_sizes[idx-1], layer_sizes[idx]))
        layers.append(norm(layer_sizes[idx]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(layer_sizes[-2], layer_sizes[-1], bias=False))
    #layers.append(norm(layer_sizes[-1]))
    return nn.Sequential(*layers)

def _gen_test_embeddings(seed=708):
    """
    expected  NTXentLoss for seed 708
    temperature eps loss
    0.05 0.0 0.0001091679
    0.05 0.1 0.005941073
    0.15 0.0 0.7775759
    0.15 0.1 1.6968876    
    """
    rs = np.random.RandomState(seed)
    e1 = torch.from_numpy(rs.normal(0,1, size=(64,2048)).astype(np.float32))
    e2 = e1 + rs.normal(0,1, size=(64,2048)).astype(np.float32)
    return nnf.normalize(e1), nnf.normalize(e2)
    #return e1, e2


def _test_ntxent():
    """
    compare to precalculated patchwork on the test embeddings
    """
    e1, e2 = _gen_test_embeddings()
    #z1, z2 = nnf.normalize(e1), nnf.normalize(e2)
    assert abs(NTXentLoss(64, 0.15, 0.1)(e1,e2) - 1.6968876) < 1e-6

def _test_vicreg():
    """
    compare to precalculated facebookresearch on test embeddings
    """
    e1, e2 = _gen_test_embeddings()
    assert abs(VicRegLoss(25,25,1)(e1,e2) - 24.402542) < 1e-6

class NTXentLoss(nn.Module):
    def __init__(self, gbs: int, temperature: float, eps: float, device:torch.device|str='cpu'):
        super(NTXentLoss, self).__init__()
        self.gbs = gbs
        self.temperature = temperature
        self.eps = eps
        self.device = device

        # negative mask
        self.neg_mask = torch.ones((2*gbs,2*gbs))
        self.neg_mask = self.neg_mask - torch.diag(torch.ones(2*gbs), 0)
        self.neg_mask = self.neg_mask - torch.diag(torch.ones(gbs), gbs)
        self.neg_mask = self.neg_mask - torch.diag(torch.ones(gbs), -gbs)
        self.neg_mask = self.neg_mask.to(device)

        # constant positive target for cross_entropy
        self.target = torch.zeros(2*self.gbs, dtype=torch.long).to(device)

    def forward(self, e1: torch.Tensor, e2: torch.Tensor):
        """
        e1: unnormalized embedding aug 1
        e2: unnormalized embedding aug 2
        a little different than patchwork: i was nervous about
        numerical stability issues with log(exp(x)) type calculations
        ... similar to from_logits in keras cross entropy
        the IFM paper does it in a similar way (don't know why)
        """
        z1 = nn.functional.normalize(e1, dim=1)
        z2 = nn.functional.normalize(e2, dim=1)

        if dist.is_initialized() and dist.get_world_size() > 1:
            # we're running distributed, gather the global batch
            z1, z2 = batch_all_gather(z1), batch_all_gather(z2)

        # s_ij and s_ji are the same (symmetrical)
        # calculate s_ij (gbs) and duplicate (2*gbs)
        pos_sim = (z1*z2).sum(dim=1, keepdim=True) # gbs
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0) # 2*gbs

        # calculate all pairs of similarities and mask out negatives
        z = torch.cat((z1,z2), dim=0) # 2*gbs
        S = torch.matmul(z, z.T) # 2*gbs x 2*gbs
        neg_sim = S * self.neg_mask # 2*gbs x 2*gbs

        # we masked out the positives from the full matrix
        # and handle them differently for IFM (+- eps)
        # concate them back on, positives will be the first column
        logits = torch.cat([pos_sim - self.eps,  neg_sim + self.eps], dim=1) / self.temperature
        # calculate cross entropy with target set to the positive similarity (0th/first column, hence torch.zeros(2*gbs))
        loss = nnf.cross_entropy(logits, self.target, reduction='mean')
        return loss


class VicRegLoss(nn.Module):
    def __init__(self, sim_coeff: float, std_coeff: float, cov_coeff: float):
        super(VicRegLoss, self).__init__()
        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff
    
    def forward(self, e1: torch.Tensor, e2: torch.Tensor):
        """
        e1: unnormalized embedding aug 1
        e2: unnormalized embedding aug 2
        taken almost directly from https://github.com/facebookresearch/vicreg/ 
        """
        assert e1.shape == e2.shape
        N,d = e1.shape
        repr_loss = nnf.mse_loss(e1, e2)

        if dist.is_initialized() and dist.get_world_size() > 1:
            # we're running distributed, gather the global batch
            e1, e2 = batch_all_gather(e1), batch_all_gather(e2)
        e1 = e1 - e1.mean(dim=0)
        e2 = e2 - e2.mean(dim=0)

        std_e1 = torch.sqrt(e1.var(dim=0) + 0.0001)
        std_e2 = torch.sqrt(e2.var(dim=0) + 0.0001)
        std_loss = torch.mean(nnf.relu(1 - std_e1)) / 2 + torch.mean(nnf.relu(1 - std_e2)) / 2

        cov_e1 = (e1.T @ e1) / (N - 1)
        cov_e2 = (e2.T @ e2) / (N - 1)
        cov_loss = off_diagonal(cov_e1).pow_(2).sum().div(d) + off_diagonal(cov_e2).pow_(2).sum().div(d)

        loss = self.sim_coeff * repr_loss + self.std_coeff * std_loss + self.cov_coeff * cov_loss
        return loss
