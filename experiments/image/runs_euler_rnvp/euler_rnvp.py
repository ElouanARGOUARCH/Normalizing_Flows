import numpy as np
import torch
from matplotlib import image
torch.manual_seed(0)
number_runs = 20

from models_nf import RealNVP
rgb = image.imread("euler.jpg")
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
grey = torch.tensor(rgb2gray(rgb))
loss_values = []

for i in range(number_runs):
    #Sample data according to image
    vector_density = grey.flatten()
    vector_density = vector_density/torch.sum(vector_density)
    lines, columns = grey.shape

    num_samples = 300000
    cat = torch.distributions.Categorical(probs = vector_density)
    categorical_samples = cat.sample([num_samples])
    target_samples = torch.cat([((categorical_samples // columns + torch.rand(num_samples)) / lines).unsqueeze(-1),((categorical_samples % columns + torch.rand(num_samples)) / columns).unsqueeze(-1)],dim=-1)

    real_nvp = RealNVP(target_samples, 3, None, None)

    epochs = 1000
    batch_size = 30000
    real_nvp.train(epochs, batch_size)

    filename = 'runs_euler_rnvp' + str(i) + '.sav'
    torch.save(real_nvp, filename)