import torch
from models_nf import NeuralSplineFlow

###MNIST###

import torchvision.datasets as datasets
import matplotlib.pyplot as plt
mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=None)
images = mnist_trainset.data.flatten(start_dim=1)
targets = mnist_trainset.targets

digit = 'all'
if digit != 'all':
    extracted = images[targets == digit].float()
else:
    extracted = images.float()
target_samples = extracted

num_samples = target_samples.shape[0]
print('number of samples = ' + str(num_samples))
p = target_samples.shape[-1]
plt.imshow(target_samples[torch.randint(low = 0, high = num_samples, size = [1])].reshape(28,28))

train_set, test_set = target_samples[:4000], target_samples[4000:]

nsf = NeuralSplineFlow(target_samples, 20, 64, 4)
nsf.train(1, 60000)

filename = 'nsf_mnist.sav'
torch.save(nsf, filename)