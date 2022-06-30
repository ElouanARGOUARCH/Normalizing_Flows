import torch
from torch import nn
from tqdm import tqdm
from models_nf.multivariate_normal_reference import MultivariateNormalReference

class MixedModelDensityEstimator(nn.Module):
    def __init__(self, target_samples,structure):
        super().__init__()
        self.target_samples = target_samples
        self.p = self.target_samples.shape[-1]
        self.structure = structure
        self.N = len(self.structure)

        self.reference = MultivariateNormalReference(self.p)

        self.model = [structure[-1][0](self.p,self.structure[-1][1], q_log_density=self.reference.log_density)]
        for i in range(self.N - 2, -1, -1):
            self.model.insert(0, structure[i][0](self.p, structure[i][1], q_log_density=self.model[0].log_psi))
        self.loss_values= []

    def compute_number_params(self):
        number_params = 0
        for model in self.model:
            number_params += sum(p.numel() for p in model.parameters() if p.requires_grad)
        return number_params

    def sample_model(self, num_samples):
        with torch.no_grad():
            z = self.reference.sample(num_samples)
            for i in range(self.N - 1, -1, -1):
                z = self.model[i].sample_backward(z)
            return z

    def sample_latent(self, x):
        with torch.no_grad():
            for i in range(self.N):
                x = self.model[i].sample_forward(x)
            return x

    def log_density(self, x):
        return self.model[0].log_psi(x)

    def loss(self, batch):
        return - self.log_density(batch).mean()

    def train(self, epochs, batch_size = None):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        self.para_dict = []
        for model in self.model:
            self.para_dict.insert(-1,{'params':model.parameters(), 'lr': model.lr})
            model.to(device)
        self.optimizer = torch.optim.Adam(self.para_dict)

        if batch_size is None:
            batch_size = self.target_samples.shape[0]
        dataset = torch.utils.data.TensorDataset(self.target_samples)

        pbar = tqdm(range(epochs))
        for t in pbar:
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
            for i, batch in enumerate(dataloader):
                x = batch[0].to(device)
                self.optimizer.zero_grad()
                batch_loss = self.loss(x)
                batch_loss.backward()
                self.optimizer.step()
            with torch.no_grad():
                iteration_loss = torch.tensor([self.loss(batch[0].to(device)) for i, batch in enumerate(dataloader)]).mean().item()
            self.loss_values.append(iteration_loss)
            pbar.set_postfix_str('loss = ' + str(round(iteration_loss,6)))
        self.to('cpu')
        for model in self.model:
            model.to(torch.device('cpu'))