import torch
from torch import nn
import torch.nn.functional as F

class MaskedLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mask = torch.ones(out_features, in_features).to(device)

    def set_mask(self, mask):
        self.mask = mask.T

    def forward(self, input):
        return F.linear(input, self.mask * self.weight, self.bias)


class MAFLayer(nn.Module):
    def __init__(self, p,hidden_sizes, q_log_density):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.p = p
        self.net = []
        self.hidden_sizes=hidden_sizes
        self.q_log_density = q_log_density
        self.lr = 5e-5

        hs = [self.p] + self.hidden_sizes + [2*self.p]
        for h0, h1 in zip(hs, hs[1:]):
            self.net.extend([
                MaskedLinear(h0, h1),
                nn.Tanh(),
            ])
        self.net.pop()
        self.net = nn.Sequential(*self.net)

        self.m = {}
        self.update_masks()

    def update_masks(self):
        L = len(self.hidden_sizes)
        self.m[-1] = torch.randperm(self.p)
        print(self.m[-1])
        for l in range(L):
            self.m[l] = torch.randint(torch.min(self.m[l - 1]), self.p - 1, [self.hidden_sizes[l]])

        masks = [self.m[l - 1][:, None] <= self.m[l][None, :] for l in range(L)]
        masks.append(self.m[L - 1][:, None] < self.m[-1][None, :])
        masks[-1] = masks[-1].repeat(1, 2)

        # set the masks in all MaskedLinear layers
        layers = [l for l in self.net.modules() if isinstance(l, MaskedLinear)]
        for l, m in zip(layers, masks):
            l.set_mask(m.to(self.device))

    def sample_forward(self,x):
        out = self.net(x)
        m, log_s = out[...,self.p:],out[...,:self.p]
        return m + torch.exp(log_s)*x

    def log_psi(self,x):
        out = self.net(x)
        m, log_s = out[...,self.p:],out[...,:self.p]
        z = m + torch.exp(log_s)*x
        return self.q_log_density(z) + torch.sum(log_s, dim = -1)

    def sample_backward(self,z):
        x = torch.zeros_like(z)
        for i in self.m[-1]:
            out = self.net(x)
            m, log_s = out[..., self.p:], out[..., :self.p]
            temp = (z - m)/torch.exp(log_s)
            x[:,i] = temp[:,i]
        return x
