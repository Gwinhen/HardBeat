import numpy as np
import sys
import torch

from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms as T
from torchvision.datasets import CIFAR10, STL10, SVHN, GTSRB

from models.densenet import densenet121
from models.googlenet import googlenet
from models.inception import inception_v3
from models.mobilenetv2 import mobilenet_v2
from models.resnet_cifar import resnet18, resnet34
from models.resnet import resnet50
from models.vgg import vgg11_bn


_cifar_networks = {
    'vgg11_bn':   vgg11_bn(),
    'resnet18':     resnet18(),
    'resnet34':     resnet34(),
    'resnet50':     resnet50(),
    'densenet121':  densenet121(),
    'mobilenet_v2': mobilenet_v2(),
    'googlenet':    googlenet(),
    'inception_v3': inception_v3(),
}

_gtsrb_networks = {
    'mobilenet_v2': mobilenet_v2(num_classes=43),
    'resnet50':     resnet50(num_classes=43),
}

_mean = {
    'default':      [0.5   , 0.5   , 0.5   ],
    'cifar10':      [0.4914, 0.4822, 0.4465],
    'stl10':        [0.4914, 0.4822, 0.4465],
    'svhn':         [0.0   , 0.0   , 0.0   ],
    'gtsrb':        [0.3337, 0.3064, 0.3171],
}

_std = {
    'default':      [0.5   , 0.5   , 0.5   ],
    'cifar10':      [0.2471, 0.2435, 0.2616], # huyvnphan/PyTorch_CIFAR10
    'stl10':        [0.2471, 0.2435, 0.2616],
    'svhn':         [1.0   , 1.0   , 1.0   ],
    'gtsrb':        [0.2672, 0.2564, 0.2629],
}

_size = {
    'cifar10':      ( 32,  32, 3),
    'stl10':        ( 32,  32, 3),
    'svhn':         ( 32,  32, 3),
    'gtsrb':        ( 32,  32, 3),
}

_num = {
    'cifar10':      10,
    'stl10':        10,
    'svhn':         10,
    'gtsrb':        43,
}


def get_classes(dataset):
    return _num[dataset]


def get_size(dataset):
    return _size[dataset]


def get_norm(dataset):
    mean = torch.FloatTensor(_mean[dataset])
    std  = torch.FloatTensor(_std[dataset])
    normalize   = T.Normalize(mean, std)
    unnormalize = T.Normalize(- mean / std, 1 / std)
    return normalize, unnormalize


def get_data(loader, label, num):
    for i, (x_batch, y_batch) in enumerate(loader):
        indices = np.where(y_batch == label)[0]

        if i == 0:
            x_data, y_data = x_batch[indices], y_batch[indices]
        else:
            x_data = torch.cat((x_data, x_batch[indices]))
            y_data = torch.cat((y_data, y_batch[indices]))

        if i > 8 and x_data.size(0) >= num:
            break

        sys.stdout.write(f'\r{i}: {x_data.size(0)}')
    print()

    return x_data[:num], y_data[:num]


def get_loader(dataset, train, batch_size, ratio=1.0):
    if dataset == 'cifar10':
        if train:
            transform = T.Compose([T.RandomCrop(32, padding=4),
                                   T.RandomHorizontalFlip(),
                                   T.ToTensor(),
                                   T.Normalize(_mean[dataset], _std[dataset])])
        else:
            transform = T.Compose([T.ToTensor(),
                                   T.Normalize(_mean[dataset], _std[dataset])])
        dataset = CIFAR10(root='./data', train=train, transform=transform)
    elif dataset == 'stl10':
        split = 'train' if train else 'test'
        transform = T.Compose([T.Resize(32),
                               T.ToTensor(),
                               T.Normalize(_mean[dataset], _std[dataset])])
        dataset = STL10(root='./data', split=split, transform=transform)
    elif dataset == 'svhn':
        split = 'train' if train else 'test'
        transform = T.Compose([T.ToTensor(),
                               T.Normalize(_mean[dataset], _std[dataset])])
        dataset = SVHN(root='./data/svhn', split=split, transform=transform)
    elif dataset == 'gtsrb':
        split = 'train' if train else 'test'
        size = get_size(dataset)[0]
        transform = T.Compose([T.Resize(size),
                               T.CenterCrop(size),
                               T.ToTensor(),
                               T.Normalize(_mean[dataset], _std[dataset])])
        dataset = GTSRB(root='./data', split=split, transform=transform)

    if ratio < 1:
        indices = np.arange(int(len(dataset) * ratio))
        dataset = Subset(dataset, indices)

    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=8,
                            shuffle=train,
                            pin_memory=True,
                            drop_last=train)
    return dataloader


def get_model(dataset, network):
    if dataset in ['cifar10', 'stl10', 'svhn']:
        # huyvnphan/PyTorch_CIFAR10
        model = _cifar_networks[network]
        model.load_state_dict(torch.load(f'ckpt/{network}.pt'))
    elif dataset == 'gtsrb':
        model = _gtsrb_networks[network]

    return model


def decision_function(model, inputs, normalize, target):
    inputs = torch.clamp(inputs, 0, 1)
    pred = model(normalize(inputs).cuda()).max(dim=1)[1]
    compare = torch.eq(pred, target)
    return compare
