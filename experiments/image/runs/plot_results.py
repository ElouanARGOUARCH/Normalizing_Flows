import torch
import numpy
import matplotlib.pyplot as plt
lines = 256
columns = 197
number_runs = 10
for i in range(number_runs):
    filename = 'euler_rnvp' + str(i) + '.sav'
    dif = torch.load(filename)
    with torch.no_grad():
        grid = torch.cartesian_prod(torch.linspace(0, lines, lines), torch.linspace(0, columns, columns))
        density = torch.exp(dif.log_density(grid)).reshape(lines, columns).T
        figure = plt.figure(figsize=(12, 8))
        ax = figure.add_subplot(121)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        ax.imshow(torch.flip(torch.flip(density.T, [0, 1]), [0, 1]), extent=[0, columns, 0, lines])
        dif_samples = dif.sample_model(300000)
        hist_dif_samples, x_edges, y_edges = numpy.histogram2d(dif_samples[:, 1].numpy(), dif_samples[:, 0].numpy(),
                                                               bins=(lines, columns), range=[[0, columns], [0, lines]])
        ax = figure.add_subplot(122)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        ax.imshow(torch.flip(torch.flip(torch.tensor(hist_dif_samples).T, [0, 1]), [0, 1]),
                   extent=[0, columns, 0, lines])
        filename_png = 'euler_dif' + str(i) + '.png'
        figure.savefig(filename_png)