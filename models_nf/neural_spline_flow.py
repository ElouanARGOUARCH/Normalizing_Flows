import torch
from torch import nn
import numpy as np
import normflow as nf

from sklearn.datasets import make_moons

from matplotlib import pyplot as plt

from tqdm import tqdm

class NeuralSplineFlow(nn.Module):
    def __init__(self, target_samples,K, hidden_units, hidden_layers):
        self.target_samples = target_samples
        self.K = K
        self.hidden_units = hidden_units
        self.hidden_layers = hidden_layers
        self.p = target_samples.shape[-1]
        flows = []
        for i in range(K):
            flows += [nf.flows.AutoregressiveRationalQuadraticSpline(self.p, hidden_layers,hidden_units)]
            flows += [nf.flows.LULinearPermute(self.p)]

        # Set prior and q0
        self.q0 = nf.distributions.DiagGaussian(2, trainable=False)

        # Construct flow model
        self.model = nf.NormalizingFlow(q0=q0, flows=flows)

    def train(self, epochs, batch_size = None):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        self.model.to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = 1e-3, weight_decay=1e-5)

        if batch_size is None:
            batch_size = self.target_samples.shape[0]
        dataset = torch.utils.data.TensorDataset(self.target_samples)

        pbar = tqdm(range(epochs))
        for t in pbar:
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
            for i, batch in enumerate(dataloader):
                x = batch[0].to(device)
                self.optimizer.zero_grad()
                batch_loss = self.model.forward_kld(x)
                batch_loss.backward()
                self.optimizer.step()
            with torch.no_grad():
                iteration_loss = torch.tensor([self.model.forward_kld(batch[0].to(device)) for i, batch in enumerate(dataloader)]).mean().item()
            self.loss_values.append(iteration_loss)
            pbar.set_postfix_str('loss = ' + str(round(iteration_loss,6)) + ' ; device: ' + str(self.model.device))

        self.to('cpu')
        for model in self.model:
            model.to(torch.device('cpu'))
