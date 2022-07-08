import numpy as np
import torch
from matplotlib import image
import matplotlib.pyplot as plt

torch.manual_seed(0)
number_runs = 10

from models_nf import NeuralSplineFlow
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

    nsf = NeuralSplineFlow(target_samples, 10,32,3)

    epochs = 1000
    batch_size = 30000
    nsf.train(epochs, batch_size)

    filename = 'runs_euler_nsf' + str(i) + '.sav'
    #torch.save(nsf, filename)
    with torch.no_grad():
        grid = torch.cartesian_prod(torch.linspace(0,1,lines),torch.linspace(0,1, columns))
        density = torch.exp(nsf.model.log_prob(grid)).reshape(lines,columns).T
        figure = plt.figure(figsize = (12,8))
        ax = figure.add_subplot(121)
        ax.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
        ax.imshow(torch.flip(torch.flip(density.T,[0,1]),[0,1]),extent = [0,columns,0, lines])
        ax = figure.add_subplot(122)
        ax.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
        ax.imshow(torch.flip(torch.flip(density.T,[0,1]),[0,1]),extent = [0,columns,0, lines])
    figure.savefig('runs_euler_nsf' + str(i) + '.png')
    loss_values.append(torch.tensor([nsf.loss_values[-1]]))
with open('readme.txt', 'w') as f:
    f.write('mean =' + str(torch.mean(torch.cat(loss_values))))
    f.write('\n')
    f.write('std =' + str(torch.std(torch.cat(loss_values))))