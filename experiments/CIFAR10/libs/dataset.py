
import torch
from fedtorchPRO.imitate.cifar import CIFAR10
from fedtorchPRO.imitate import transforms 

def PerImageStandardization(image):

    # compute image mean
    image_mean = image.mean(dim=(-1, -2, -3),keepdim=True)

    # compute image standard deviation
    stddev = image.std(axis=(-1, -2, -3),keepdim=True,correction = 0)

    # compute minimum standard deviation
    min_stddev = torch.rsqrt(torch.tensor(torch.numel(image)))

    # compute adjusted standard deviation
    adjusted_stddev = torch.max(stddev, min_stddev)

    # normalize image
    image = torch.div(image - image_mean, adjusted_stddev)

    return image

def get_source():
    return CIFAR10(
        root='./datasets/CIFAR10/',
        train=True,
        transform = transforms.Compose([
            # transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.RandomCrop(24),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            PerImageStandardization
        ]),
        download = True,
    )

def get_test():
    return CIFAR10(
        root='./datasets/CIFAR10/',
        train=False,
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.CenterCrop(24),
            PerImageStandardization,
        ]),
        download = True,
    )
