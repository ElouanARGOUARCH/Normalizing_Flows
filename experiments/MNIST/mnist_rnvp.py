import torch
from models_nf import MixedModelDensityEstimator, RealNVPDensityEstimatorLayer

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
plt.imshow(target_samples[torch.randint(low = 0, high = num_samples, size = [1])].reshape(28,28))

train_set, test_set = target_samples[:4000], target_samples[4000:]

structure = [[RealNVPDensityEstimatorLayer,[256,256,256]] for i in range(10)]
rnvp = MixedModelDensityEstimator(target_samples, structure)
rnvp.train(1, 10000)

filename = 'rnvp_mnist.sav'
torch.save(rnvp, filename)