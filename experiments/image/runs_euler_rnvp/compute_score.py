import torch
list_score_nll = []
list_score_dkl = []
number_runs = 20
lines = 256
columns = 197
for i in range(number_runs):
    filename = 'runs_euler_rnvp' + str(i) + '.sav'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dif = torch.load(filename,map_location=torch.device(device))
    list_score_nll.append(torch.tensor([dif.loss_values[-1]]))

print('mean score NLL = ' + str(torch.mean(torch.cat(list_score_nll), dim = 0).item()))
print('std NLL= ' + str(torch.std(torch.cat(list_score_nll), dim = 0).item()))
