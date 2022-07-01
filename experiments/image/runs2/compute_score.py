import torch
list_score_nll = []
list_score_dkl = []
number_runs = 10
lines = 256
columns = 197
import numpy
for i in range(number_runs):
    filename = 'euler_rnvp' + str(i) + '.sav'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dif = torch.load(filename,map_location=torch.device(device))
    nll = torch.tensor([dif.loss_values[-1]])
    list_score_nll.append(nll)
    hist_target_samples, x_edges, y_edges = numpy.histogram2d(dif.target_samples[:, 1].numpy(),
                                                              dif.target_samples[:, 0].numpy(), bins=(lines, columns),
                                                              range=[[0, columns], [0, lines]], normed = True)
    hist_target_samples = torch.tensor(hist_target_samples)

    DKL_hist = nll + torch.sum(torch.nan_to_num(hist_target_samples*torch.log(hist_target_samples)))
    list_score_dkl.append(torch.tensor([DKL_hist]))

print('mean score NLL = ' + str(torch.mean(torch.cat(list_score_nll), dim = 0).item()))
print('std NLL= ' + str(torch.std(torch.cat(list_score_nll), dim = 0).item()))
print('mean score DKL= ' + str(torch.mean(torch.cat(list_score_dkl), dim = 0).item()))
print('std DKL = ' + str(torch.std(torch.cat(list_score_dkl), dim = 0).item()))
