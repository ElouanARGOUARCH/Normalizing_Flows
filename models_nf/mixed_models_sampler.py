import torch
from torch import nn
from tqdm import tqdm
from models_nf.multivariate_normal_reference import MultivariateNormalReference

class MixedModelSampler(nn.Module):
    def __init__(self, target_log_density, p,structure):

        super().__init__()
        self.target_log_density = target_log_density
        self.p = p
        self.structure = structure
        self.N = len(self.structure)
        self.reference = MultivariateNormalReference(self.p)

        self.model = [structure[0][0](self.p,self.structure[0][1], p_log_density=self.target_log_density)]
        for i in range(1,self.N):
            self.model.append(structure[i][0](self.p, structure[i][1], p_log_density=self.model[i-1].log_phi))
        for i in range(self.N-1):
            self.model[i].q_log_density = self.model[i+1].log_psi
        self.model[-1].q_log_density = self.reference.log_density
        self.loss_values = []

    def sample(self, num_samples):
        z = self.reference.sample(num_samples)
        for i in range(self.N - 1, -1, -1):
            z = self.model[i].sample_backward(z)
        return z

    def log_density(self, x):
        return self.model[0].log_psi(x)

    def proxy_log_density(self, z):
        return self.model[-1].log_phi(z)

    def loss(self, batch):
        return - self.proxy_log_density(batch).mean()

    def DKL_latent(self,batch_z):
        return (self.reference.log_density(batch_z) - self.proxy_log_density(batch_z)).mean()

    def DKL_observed(self,batch_x):
        return (self.log_density(batch_x) - self.target_log_density(batch_x)).mean()

    def train(self, num_samples, epochs, batch_size):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        self.para_dict = []
        for model in self.model:
            self.para_dict.insert(-1, {'params': model.parameters(), 'lr': model.lr})
            model.to(device)
        self.optimizer = torch.optim.Adam(self.para_dict)

        if batch_size is None:
            batch_size = self.target_samples.shape[0]

        reference_samples = self.reference.sample(num_samples)
        dataset = torch.utils.data.TensorDataset(reference_samples)

        pbar = tqdm(range(epochs))
        for t in pbar:
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
            for i, batch in enumerate(dataloader):
                z = batch[0].to(device)
                self.optimizer.zero_grad()
                batch_loss = self.loss(z)
                batch_loss.backward()
                self.optimizer.step()
            with torch.no_grad():
                iteration_loss = torch.tensor(
                    [self.loss(batch[0].to(device)) for i, batch in enumerate(dataloader)]).mean().item()
            self.loss_values.append(iteration_loss)
            pbar.set_postfix_str('loss = ' + str(round(iteration_loss, 6)))

        for model in self.model:
            model.to(torch.device('cpu'))
        self.to(torch.device('cpu'))
