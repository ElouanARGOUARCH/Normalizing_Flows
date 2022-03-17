import torch
from torch import nn

class RealNVPSamplerLayer(nn.Module):
    def __init__(self,p,hidden_dim, p_log_density):
        super().__init__()
        self.p = p
        net = []
        hs = [self.p] + hidden_dim + [2*self.p]
        for h0, h1 in zip(hs, hs[1:]):
            net.extend([
                nn.Linear(h0, h1),
                nn.Tanh(),
            ])
        net.pop()
        self.net = nn.Sequential(*net)

        self.mask = [torch.cat([torch.zeros(int(self.p/2)), torch.ones(self.p - int(self.p/2))], dim = 0),torch.cat([torch.ones(int(self.p/2)), torch.zeros(self.p - int(self.p/2))], dim = 0)]
        self.q_log_density = None
        self.p_log_density = p_log_density
        self.lr = 5e-3

    def log_phi(self,z):
        x = z
        log_det = torch.zeros(z.shape[:-1])
        for mask in self.mask:
            out = self.net(x*mask)
            m, log_s = out[...,:self.p]*(1 - mask),out[...,self.p:]* (1 - mask)
            x = (x*(1-mask) -m)/torch.exp(log_s) + x*mask
            log_det -= torch.sum(log_s, dim=-1)
        return self.p_log_density(x) + log_det

    def sample_backward(self, z):
        with torch.no_grad():
            x = z
            for mask in self.mask:
                out = self.net(x*mask)
                m, log_s = out[...,:self.p]* (1 - mask),out[...,self.p:]* (1 - mask)
                x = (x*(1-mask) -m)/torch.exp(log_s) + x*mask
            return x

    def log_psi(self, x):
        with torch.no_grad():
            z = x
            log_det = torch.zeros(x.shape[:-1])
            for mask in reversed(self.mask):
                out = self.net(mask * z)
                m, log_s = out[...,:self.p]* (1 - mask),out[...,self.p:]* (1 - mask)
                z = z*(1 - mask) * torch.exp(log_s) + m + mask * z
                log_det += torch.sum(log_s, dim = -1)
            return self.q_log_density(z) + log_det
