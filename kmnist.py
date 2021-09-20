import os
from torch.utils.data import DataLoader
from torch import randint
from torchvision.datasets import KMNIST
from torchvision.transforms import ToTensor, Normalize, Compose
from numpy.random import permutation
from numpy import arange


def get(data_root="datasets/", batch_size_train=256, batch_size_test=512, label_noise_pct=0.0, **kwargs):
    # Create the folder where the data will be downloaded
    # exist_ok avoid exception if the folder already exists
    os.makedirs(data_root, exist_ok=True)

    # Next, we prepare a preprocessing pipeline which will be applied before feeding our data into the model
    # namely, ToTensor() transforms an image in a tensor and squishes its values between 0 and 1
    # Normalize(), instead, normalizes it w.r.t. the given mean and std. Since MNIST is grayscale,
    # we have only 1 color channel, hence, mean and std are considered as singleton tuples. If we had RGB
    # images, we should've written someting like Normalize((mean channel 1, mean channel 2, mean channel 3), ...)
    transforms = Compose([
        ToTensor(),
        Normalize((0.1918,), (0.3483,))
    ])

    # We download the train and the test dataset in the given root and applying the given transforms
    trainset = KMNIST(data_root, train=True, transform=transforms, download=True)
    testset = KMNIST(data_root, train=False, transform=transforms, download=True)

    if label_noise_pct is not None and label_noise_pct > 0.0 and label_noise_pct < 1.0:
        inject_label_noise(trainset, label_noise_pct, trainset.targets.max().item() + 1)

    # We feed our datasets into DataLoaders, which automatically manage the split into batches for SGD
    # shuffle indicates whether the data needs to be shuffled before the creation of batches
    # it's an overhead, but is necessary for a clean training, so we don't use it for the test set
    trainloader = DataLoader(trainset, batch_size=batch_size_train, shuffle=True, **kwargs)
    testloader = DataLoader(trainset, batch_size=batch_size_test, shuffle=False, **kwargs)

    return trainloader, testloader, trainset, testset

def inject_label_noise(trainset, pct, num_classes=10):
    perm = permutation(arange(len(trainset.targets)))
    index_pct = int(len(trainset) * pct)
    perm = perm[:index_pct]

    trainset.targets[perm] = randint(num_classes, perm.shape)

