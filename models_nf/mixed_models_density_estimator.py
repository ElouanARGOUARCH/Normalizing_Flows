import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from utils.color_visual import *

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

    def train(self, epochs, batch_size = None, visual = False):

        self.para_dict = []
        for i in range(self.N):
            self.para_dict.insert(-1,{'params':self.model[i].parameters(), 'lr': self.model[i].lr})
        self.optimizer = torch.optim.Adam(self.para_dict)

        if batch_size is None:
            batch_size = self.target_samples.shape[0]
        dataset = torch.utils.data.TensorDataset(self.target_samples)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)

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
        self.to(torch.device('cpu'))

    def train_visual(self, best_loss, best_iteration, loss_values):
        fig = plt.figure(figsize=(12, 4))
        ax = plt.subplot(111)
        Y1, Y2 = best_loss - (max(loss_values) - best_loss) / 2, max(loss_values) + (max(loss_values) - best_loss) / 4
        ax.set_ylim(Y1, Y2)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.plot(loss_values, label='Loss values during training', color='black')
        ax.scatter([best_iteration], [best_loss], color='black', marker='d')
        ax.axvline(x=best_iteration, ymax=(best_loss - best_loss + (max(loss_values) - best_loss) / 2) / (
                max(loss_values) + (max(loss_values) - best_loss) / 4 - best_loss + (
                max(loss_values) - best_loss) / 2), color='black', linestyle='--')
        ax.text(0, best_loss - (max(loss_values) - best_loss) / 8,
                'best iteration = ' + str(best_iteration) + '\nbest loss = ' + str(np.round(best_loss, 5)),
                verticalalignment='top', horizontalalignment='left', fontsize=12)
        if len(loss_values) > 30:
            x1, x2 = best_iteration - int(len(loss_values) / 15), min(best_iteration + int(len(loss_values) / 15),
                                                                      len(loss_values) - 1)
            k = len(loss_values) / (2.5 * (x2 - x1 + 1))
            offset = (Y2 - Y1) / (6 * k)
            y1, y2 = best_loss - offset, best_loss + offset
            axins = zoomed_inset_axes(ax, k, loc='upper right')
            axins.axvline(x=best_iteration, ymax=(best_loss - y1) / (y2 - y1), color='black', linestyle='--')
            axins.scatter([best_iteration], [best_loss], color='black', marker='d')
            axins.xaxis.set_major_locator(MaxNLocator(integer=True))
            axins.plot(loss_values, color='black')
            axins.set_xlim(x1 - .5, x2 + .5)
            axins.set_ylim(y1, y2)
            mark_inset(ax, axins, loc1=3, loc2=4)

    def model_visual(self, num_samples=5000, flow=False):
        if self.p == 1:
            linspace = 500
            with torch.no_grad():
                backward_samples = [self.reference.sample(num_samples)]
                tt = torch.linspace(torch.min(backward_samples[0]), torch.max(backward_samples[0]), linspace).unsqueeze(
                    1)
                backward_density = [torch.exp(self.reference.log_density(tt))]
                backward_linspace = [tt]
                for i in range(self.N - 1, -1, -1):
                    samples = self.model[i].sample_backward(backward_samples[0])
                    tt = torch.linspace(torch.min(samples), torch.max(samples), linspace).unsqueeze(1)
                    density = torch.exp(self.model[i].log_psi(tt))
                    backward_samples.insert(0, samples)
                    backward_linspace.insert(0, tt)
                    backward_density.insert(0, density)

            with torch.no_grad():
                forward_samples = [self.target_samples[:num_samples]]
                for i in range(self.N):
                    forward_samples.append(self.model[i].sample_forward(forward_samples[-1]))

            fig = plt.figure(figsize=((self.N + 1) * 8, 2 * 7))
            ax = fig.add_subplot(2, self.N + 1, 1)
            sns.histplot(forward_samples[0][:, 0].cpu(), stat="density", alpha=0.5, bins=125, color='red',
                         label="Input Target Samples")
            ax.legend(loc=1)
            for i in range(1, self.N):
                ax = fig.add_subplot(2, self.N + 1, i + 1)
                sns.histplot(forward_samples[i][:, 0].cpu(), stat="density", alpha=0.5, bins=125, color='magenta',
                             label="Intermediate Samples")
                ax.legend(loc=1)
            ax = fig.add_subplot(2, self.N + 1, self.N + 1)
            sns.histplot(forward_samples[-1][:, 0].cpu(), stat="density", alpha=0.5, bins=125, color='orange',
                         label="Proxy Samples")
            ax.legend(loc=1)

            ax = fig.add_subplot(2, self.N + 1, self.N + 1 + 1)
            sns.histplot(backward_samples[0][:, 0].cpu(), stat="density", alpha=0.5, bins=125, color='blue',
                         label="Model Samples")
            ax.plot(backward_linspace[0].cpu(), backward_density[0].cpu(), color='blue',
                    label="Output Model density")
            ax.legend(loc=1)
            for i in range(1, self.N):
                ax = fig.add_subplot(2, self.N + 1, self.N + 1 + i + 1)
                sns.histplot(backward_samples[i][:, 0].cpu(), stat="density", alpha=0.5, bins=125, color='purple',
                             label="Intermediate Samples")
                ax.plot(backward_linspace[i].cpu(), backward_density[i].cpu(), color='purple',
                        label="Intermediate density")
                ax.legend(loc=1)
            ax = fig.add_subplot(2, self.N + 1, self.N + 1 + self.N + 1)
            ax.plot(backward_linspace[-1].cpu(), backward_density[-1].cpu(), color='green', label="Reference density")
            sns.histplot(backward_samples[-1][:, 0].cpu(), stat="density", alpha=0.5, bins=125, color='green',
                         label="Reference Samples")
            if hasattr(self.reference, 'log_r'):
                ax.text(0.01, 0.99, 'r = ' + str(round(torch.exp(self.reference.log_r).item(), 4)),
                        verticalalignment='top', horizontalalignment='left',
                        transform=ax.transAxes, fontsize=12)
            ax.legend(loc=1)

        elif self.p > 1 and self.p <= 5:
            if flow == True:
                delta = 200
                with torch.no_grad():
                    samples = self.reference.sample(num_samples)
                    backward_samples = [samples]
                    grid = torch.cartesian_prod(torch.linspace(torch.min(samples[:,0]).item(), torch.max(samples[:,0]).item(),delta),torch.linspace(torch.min(samples[:,1]).item(), torch.max(samples[:,1]).item(),delta))
                    grid = torch.cat((grid, torch.mean(samples[:,2:], dim = 0)*torch.ones(grid.shape[0], self.p-2)), dim = -1)
                    density = torch.exp(self.reference.log_density(grid)).reshape(delta,delta).T.cpu().detach()
                    backward_density = [density]
                    x_range = [[torch.min(samples[:,0]).item(), torch.max(samples[:,0]).item()]]
                    y_range = [[torch.min(samples[:, 1]).item(), torch.max(samples[:, 1]).item()]]
                    for i in range(self.N - 1, -1, -1):
                        samples = self.model[i].sample_backward(backward_samples[0])
                        backward_samples.insert(0, samples)
                        grid = torch.cartesian_prod(torch.linspace(torch.min(samples[:, 0]).item(), torch.max(samples[:, 0]).item(), delta),torch.linspace(torch.min(samples[:, 1]).item(), torch.max(samples[:, 1]).item(), delta))
                        grid = torch.cat((grid, torch.zeros(grid.shape[0], self.p - 2)), dim=-1)
                        density = torch.exp(self.model[i].log_psi(grid)).reshape(delta, delta).T.cpu().detach()
                        backward_density.insert(0, density)
                        x_range.insert(0,[torch.min(samples[:,0]).item(), torch.max(samples[:,0]).item()])
                        y_range.insert(0,[torch.min(samples[:, 1]).item(), torch.max(samples[:, 1]).item()])

                with torch.no_grad():
                    forward_samples = [self.target_samples[:num_samples]]
                    for i in range(self.N):
                        forward_samples.append(self.model[i].sample_forward(forward_samples[-1]))

                fig = plt.figure(figsize=((self.N + 1) * 8, 3 * 7))
                ax = fig.add_subplot(3, self.N + 1, 1)
                ax.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
                ax.scatter(forward_samples[0][:, 0].cpu(), forward_samples[0][:, 1].cpu(), alpha=0.5, color=red_color)
                ax.set_title(r'$P$ samples')
                ax.set_xlim((x_range[0][0], x_range[0][1]))
                ax.set_ylim((y_range[0][0], y_range[0][1]))
                for i in range(1, self.N):
                    ax = fig.add_subplot(3, self.N + 1, i + 1)
                    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
                    ax.scatter(forward_samples[i][:, 0].cpu(), forward_samples[i][:, 1].cpu(), alpha=0.5,
                                color=pink_color)
                    ax.set_xlim((x_range[i][0], x_range[i][1]))
                    ax.set_ylim((y_range[i][0], y_range[i][1]))
                ax = fig.add_subplot(3, self.N + 1, self.N + 1)
                ax.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
                ax.scatter(forward_samples[-1][:, 0].cpu(), forward_samples[-1][:, 1].cpu(), alpha=0.5,
                            color=orange_color)
                ax.set_xlim((x_range[-1][0], x_range[-1][1]))
                ax.set_ylim((y_range[-1][0], y_range[-1][1]))
                ax.set_title(r'$\Phi$ Samples')

                ax = fig.add_subplot(3, self.N + 1, self.N + 1 + 1)
                ax.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
                ax.scatter(backward_samples[0][:, 0].cpu(), backward_samples[0][:, 1].cpu(), alpha=0.5, color=blue_color)
                ax.set_xlim((x_range[0][0], x_range[0][1]))
                ax.set_ylim((y_range[0][0], y_range[0][1]))
                ax.set_title(r'$\Psi$ Samples')
                ax = fig.add_subplot(3, self.N + 1, 2*(self.N + 1) + 1)
                ax.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
                ax.pcolormesh(torch.linspace(torch.min(backward_samples[0][:, 0]).item(), torch.max(backward_samples[0][:, 0]).item(), 200),torch.linspace(torch.min(backward_samples[0][:, 1]).item(), torch.max(backward_samples[0][:, 1]).item(), 200), backward_density[0],cmap = blue_cmap,shading='auto' )
                ax.set_xlim((x_range[0][0], x_range[0][1]))
                ax.set_ylim((y_range[0][0], y_range[0][1]))
                ax.set_title(r'$\Psi$ Density')
                for i in range(1, self.N):
                    ax = fig.add_subplot(3, self.N + 1, self.N + 1 + i + 1)
                    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
                    ax.scatter(backward_samples[i][:, 0].cpu(), backward_samples[i][:, 1].cpu(), alpha=0.5,
                                color=purple_color)
                    ax.set_xlim((x_range[i][0], x_range[i][1]))
                    ax.set_ylim((y_range[i][0], y_range[i][1]))
                    ax = fig.add_subplot(3, self.N + 1, 2 * (self.N + 1) + i + 1)
                    ax.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
                    ax.set_xlim((x_range[i][0], x_range[i][1]))
                    ax.set_ylim((y_range[i][0], y_range[i][1]))
                    ax.pcolormesh(torch.linspace(torch.min(backward_samples[i][:, 0]).item(), torch.max(backward_samples[i][:, 0]).item(), 200),torch.linspace(torch.min(backward_samples[i][:, 1]).item(), torch.max(backward_samples[i][:, 1]).item(), 200), backward_density[i], cmap=purple_cmap,shading='auto')

                ax = fig.add_subplot(3, self.N + 1, self.N + 1 + self.N + 1)
                ax.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
                ax.scatter(backward_samples[-1][:, 0].cpu(), backward_samples[-1][:, 1].cpu(), alpha=0.5,
                            color=green_color)
                if hasattr(self.reference, 'r') and self.reference.log_r.requires_grad == True:
                    ax.text(0.01, 0.99, 'r = ' + str(list(np.round(torch.exp(self.reference.log_r)[:2].detach().cpu().numpy(),3))),
                            verticalalignment='top', horizontalalignment='left',
                            transform=ax.transAxes, fontsize=12)
                ax.set_xlim((x_range[-1][0], x_range[-1][1]))
                ax.set_ylim((y_range[-1][0], y_range[-1][1]))
                ax.set_title(r'$Q$ samples')
                ax = fig.add_subplot(3, self.N + 1, 2*(self.N + 1) + self.N + 1)
                ax.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
                ax.pcolormesh(torch.linspace(torch.min(backward_samples[-1][:, 0]).item(), torch.max(backward_samples[-1][:, 0]).item(), 200),torch.linspace(torch.min(backward_samples[-1][:, 1]).item(), torch.max(backward_samples[-1][:, 1]).item(), 200), backward_density[-1], cmap=green_cmap,shading='auto')
                if hasattr(self.reference, 'r') and self.reference.log_r.requires_grad == True:
                    ax.text(0.01, 0.99, 'r = ' + str(list(np.round(torch.exp(self.reference.log_r)[:2].detach().cpu().numpy(),3))),
                            verticalalignment='top', horizontalalignment='left',
                            transform=ax.transAxes, fontsize=12)
                ax.set_xlim((x_range[-1][0], x_range[-1][1]))
                ax.set_ylim((y_range[-1][0], y_range[-1][1]))
                ax.set_title(r"$Q$ Density")

            else:
                num_samples = min(num_samples, self.target_samples.shape[0])
                with torch.no_grad():
                    backward_samples = [self.reference.sample(num_samples)]
                    for i in range(self.N - 1, -1, -1):
                        samples = self.model[i].sample_backward(backward_samples[0])
                        backward_samples.insert(0, samples)

                with torch.no_grad():
                    forward_samples = [self.target_samples[:num_samples]]
                    for i in range(self.N):
                        forward_samples.append(self.model[i].sample_forward(forward_samples[-1]))

                dfs = []
                for i in range(self.N + 1):
                    df_forward = pd.DataFrame(forward_samples[i].cpu().numpy())
                    df_forward['label'] = 'Forward'
                    df_backward = pd.DataFrame(backward_samples[i].cpu().numpy())
                    df_backward['label'] = 'Backward'
                    dfs.append(pd.concat([df_forward, df_backward]))

                g = sns.PairGrid(dfs[0], hue="label", height=12 / self.p,
                                 palette={'Forward': 'red', 'Backward': 'blue'})
                g.map_upper(sns.scatterplot, alpha=.4)
                g.map_diag(sns.histplot, bins=60, kde=False, alpha=.4, stat='density')

                for i in range(1, self.N):
                    g = sns.PairGrid(dfs[i], hue="label", height=12 / self.p,
                                     palette={'Forward': 'magenta', 'Backward': 'purple'})
                    g.map_upper(sns.scatterplot, alpha=.4)
                    g.map_diag(sns.histplot, bins=60, kde=True, alpha=.4, stat='density')

                g = sns.PairGrid(dfs[-1], hue="label", height=12 / self.p,
                                 palette={'Backward': 'green', 'Forward': 'orange'})
                g.map_upper(sns.scatterplot, alpha=.4)
                g.map_diag(sns.histplot, bins=60, kde=False, alpha=.4, stat='density')
                if hasattr(self.reference, 'log_r'):
                    def write_power_factor(*args, **kwargs):
                        ax = plt.gca()
                        id_dim = ax.get_subplotspec().rowspan.start
                        _pf = torch.exp(self.reference.log_r)[id_dim]
                        label = f"r={_pf:.2f}"
                        ax.annotate(label, xy=(0.1, 0.95), size=8, xycoords=ax.transAxes)

                    g.map_diag(write_power_factor)

        else:
            dim_displayed = 5
            perm = torch.randperm(self.p)
            num_samples = min(num_samples, self.target_samples.shape[0])
            with torch.no_grad():
                backward_samples = [self.reference.sample(num_samples)]
                for i in range(self.N - 1, -1, -1):
                    samples = self.model[i].sample_backward(backward_samples[0])
                    backward_samples.insert(0, samples)

            with torch.no_grad():
                forward_samples = [self.target_samples[:num_samples]]
                for i in range(self.N):
                    forward_samples.append(self.model[i].sample_forward(forward_samples[-1]))

            dfs = []
            for i in range(self.N + 1):
                df_forward = pd.DataFrame(forward_samples[i][:, perm][:, :dim_displayed].cpu().numpy())
                df_forward['label'] = 'Forward'
                df_backward = pd.DataFrame(backward_samples[i][:, perm][:, :dim_displayed].cpu().numpy())
                df_backward['label'] = 'Backward'
                dfs.append(pd.concat([df_forward, df_backward]))

            g = sns.PairGrid(dfs[0], hue="label", height=12 / dim_displayed,
                             palette={'Forward': 'red', 'Backward': 'blue'})
            g.map_upper(sns.scatterplot, alpha=.4)
            g.map_diag(sns.histplot, bins=60, kde=False, alpha=.4, stat='density')

            for i in range(1, self.N):
                g = sns.PairGrid(dfs[i], hue="label", height=12 / dim_displayed,
                                 palette={'Forward': 'magenta', 'Backward': 'purple'})
                g.map_upper(sns.scatterplot, alpha=.4)
                g.map_diag(sns.histplot, bins=60, kde=True, alpha=.4, stat='density')

            g = sns.PairGrid(dfs[-1], hue="label", height=12 / dim_displayed,
                             palette={'Backward': 'green', 'Forward': 'orange'})
            g.map_upper(sns.scatterplot, alpha=.4)
            g.map_diag(sns.histplot, bins=60, kde=False, alpha=.4, stat='density')
            if hasattr(self.reference, 'log_r'):
                def write_power_factor(*args, **kwargs):
                    ax = plt.gca()
                    id_dim = ax.get_subplotspec().rowspan.start
                    _pf = torch.exp(self.reference.log_r)[id_dim]
                    label = f"r={_pf:.2f}"
                    ax.annotate(label, xy=(0.1, 0.95), size=8, xycoords=ax.transAxes)

                g.map_diag(write_power_factor)