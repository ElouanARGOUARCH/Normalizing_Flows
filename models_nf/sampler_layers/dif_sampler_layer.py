import torch
from torch import nn
from torch.distributions import Categorical

class SoftmaxWeight(nn.Module):
    def __init__(self, K, p, hidden_dimensions =[]):
        super().__init__()
        self.K = K
        self.p = p
        self.network_dimensions = [self.p] + hidden_dimensions + [self.K]
        network = []
        for h0, h1 in zip(self.network_dimensions, self.network_dimensions[1:]):
            network.extend([nn.Linear(h0, h1),nn.Tanh(),])
        network.pop()
        self.f = nn.Sequential(*network)

    def log_prob(self, z):
        unormalized_log_w = self.f.forward(z)
        return unormalized_log_w - torch.logsumexp(unormalized_log_w, dim=-1, keepdim=True)

class LocationScaleFlow(nn.Module):
    def __init__(self, K, p):
        super().__init__()
        self.K = K
        self.p = p

        self.m = nn.Parameter(torch.randn(self.K, self.p))
        self.log_s = nn.Parameter(torch.zeros(self.K, self.p))

    def backward(self, z):
        desired_size = list(z.shape)
        desired_size.insert(-1, self.K)
        Z = z.unsqueeze(-2).expand(desired_size)
        return Z * torch.exp(self.log_s).expand_as(Z) + self.m.expand_as(Z)

    def forward(self, x):
        desired_size = list(x.shape)
        desired_size.insert(-1, self.K)
        X = x.unsqueeze(-2).expand(desired_size)
        return (X-self.m.expand_as(X))/torch.exp(self.log_s).expand_as(X)

    def log_det_J(self,x):
        return -self.log_s.sum(-1)

class DIFSamplerLayer(nn.Module):
    def __init__(self,p, K, p_log_density):

        super().__init__()

        self.p = p
        self.K = K

        self.w = SoftmaxWeight(self.K, self.p)
        self.T = LocationScaleFlow(self.K, self.p)

        self.q_log_density = None
        self.p_log_density = p_log_density

        self.lr = 5e-3

    def log_v(self, x):
        z = self.T.forward(x)
        log_v = self.p_log_density(z) + torch.diagonal(self.w.log_prob(z), 0, -2, -1) + self.T.log_det_J(x)
        return log_v - torch.logsumexp(log_v, dim=-1, keepdim=True)

    def sample_backward(self, z):
        x = self.T.backward(z)
        pick = Categorical(torch.exp(self.w.log_prob(z))).sample()
        return torch.stack([x[i, pick[i], :] for i in range(z.shape[0])])

    def log_phi(self, z):
        x = self.T.backward(z)
        return torch.logsumexp(torch.diagonal(self.log_v(x), 0, -2, -1) + self.p_log_density(x) - self.T.log_det_J(z), dim=-1)

    def log_psi(self, x):
        z = self.T.forward(x)
        return torch.logsumexp(torch.diagonal(self.w.log_prob(z), 0, -2, -1) + self.q_log_density(z) + self.T.log_det_J(x),dim=-1)
