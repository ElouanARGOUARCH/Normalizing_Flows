import numpy as np
import torch
from matplotlib import image
from torch import nn
from models_nf import MixedModelDensityEstimator, RealNVPDensityEstimatorLayer

torch.manual_seed(0)
number_runs = 20

rgb = image.imread("euler.jpg")
lines, columns = rgb.shape[:-1]
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
grey = torch.tensor(rgb2gray(rgb))

for i in range(number_runs):
    #Sample data according to image
    vector_density = grey.flatten()
    vector_density = vector_density/torch.sum(vector_density)
    lignes, colonnes = grey.shape

    num_samples = 300000
    cat = torch.distributions.Categorical(probs = vector_density)
    categorical_samples = cat.sample([num_samples])
    target_samples = torch.cat([((categorical_samples // columns + torch.rand(num_samples)) / lines).unsqueeze(-1),((categorical_samples % columns + torch.rand(num_samples)) / columns).unsqueeze(-1)],dim=-1)

    num_samples = target_samples.shape[0]
    epochs = 1000
    batch_size = 30000
    structure = [[RealNVPDensityEstimatorLayer, [100, 100, 100]], [RealNVPDensityEstimatorLayer, [100, 100, 100]],[RealNVPDensityEstimatorLayer, [100, 100, 100]]]
    realnvp = MixedModelDensityEstimator(target_samples, structure)

    realnvp.train(epochs, batch_size)

    filename = 'runs_euler_rnvp/euler_rnvp' + str(i) + '.sav'
    torch.save(realnvp,filename)
