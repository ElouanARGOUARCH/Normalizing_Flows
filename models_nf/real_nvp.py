import torch
from torch import nn
import normflow as nf

from tqdm import tqdm

class RealNVP(nn.Module):
    def __init__(self, target_samples,K, hidden_units, hidden_layers):
        super().__init__()
        self.target_samples = target_samples
        self.K = K
        self.hidden_units = hidden_units
        self.hidden_layers = hidden_layers
        self.p = target_samples.shape[-1]
        flows = []
        for i in range(self.K):
            param_map = nf.nets.MLP([1,64,64,64,2], init_zeros=True)
            # Add flow layer
            flows.append(nf.flows.AffineCouplingBlock(param_map))
            # Swap dimensions
            flows.append(nf.flows.Permute(2, mode='swap'))

        # Set prior and q0
        self.q0 = nf.distributions.DiagGaussian(2, trainable=False)

        # Construct flow model
        self.model = nf.NormalizingFlow(q0=self.q0, flows=flows)
        self.loss_values = []

    def compute_number_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

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
            pbar.set_postfix_str('loss = ' + str(round(iteration_loss,6)) + ' ; device: ' + str(device))
        self.to('cpu')
