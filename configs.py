import os
import random

import numpy as np
import torch
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from save_mat import Save_mat
from utils import datasets

default_workers = os.cpu_count()


def nclass(config):
    r = {
        'imagenet100': 100,
        'cifar10': 10,
        'nuswide': 21,
        'AID' : 17,
        'DFC15' : 8,
        'UCMD' : 17,
        'MLRS' : 60
    }[config['dataset']]

    return r


def R(config):
    r = {
        'imagenet100': 1000,
        'cifar10': 59000,  # mAP@all
        'cifar10_2': 50000,
        'nuswide': 5000,
        'AID' : 3000,
        'DFC15' : 3342,
        'UCMD' : 2100,
        'MLRS' : 1000
    }[config['dataset'] + {2: '_2'}.get(config['dataset_kwargs']['evaluation_protocol'], '')]

    return r

def scheduler(config, optimizer):
    s_type = config['scheduler']
    kwargs = config['scheduler_kwargs']

    if s_type == 'step':
        return lr_scheduler.StepLR(optimizer,
                                   kwargs['step_size'],
                                   kwargs['gamma'])
    elif s_type == 'mstep':
        return lr_scheduler.MultiStepLR(optimizer,
                                        [int(float(m) * int(config['epochs'])) for m in
                                         kwargs['milestones'].split(',')],
                                        kwargs['gamma'])
    else:
        raise Exception('Scheduler not supported yet: ' + s_type)


def compose_transform(mode='train', resize=0, crop=0, norm=0,
                      augmentations=None):
    """

    :param mode:
    :param resize:
    :param crop:
    :param norm:
    :param augmentations:
    :return:
    if train:
      Resize (optional, usually done in Augmentations)
      Augmentations
      ToTensor
      Normalize

    if test:
      Resize
      CenterCrop
      ToTensor
      Normalize
    """
    # norm = 0, 0 to 1
    # norm = 1, -1 to 1
    # norm = 2, standardization
    mean, std = {
        0: [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
        1: [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]],
        2: [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]
    }[norm]

    compose = []

    if resize != 0:
        compose.append(transforms.Resize(resize))

    if mode == 'train' and augmentations is not None:
        compose += augmentations

    if mode == 'test' and crop != 0 and resize != crop:
        compose.append(transforms.CenterCrop(crop))

    compose.append(transforms.ToTensor())

    if norm != 0:
        compose.append(transforms.Normalize(mean, std))

    return transforms.Compose(compose)


def dataset(config, filename, transform_mode):
    dataset_name = config['dataset']
    nclass = config['arch_kwargs']['nclass']

    resize = config['dataset_kwargs'].get('resize', 0)
    crop = config['dataset_kwargs'].get('crop', 0)
    norm = config['dataset_kwargs'].get('norm', 2)
    reset = config['dataset_kwargs'].get('reset', False)

    if dataset_name in ['imagenet100', 'nuswide','AID' , 'DFC15','UCMD','MLRS']:

        # resizec = 0 if resize == 256 else resize
        # cropc = 224 if crop == 0 else crop
        if transform_mode == 'train':
            transform = compose_transform('train', 0, crop, 2, {
                'imagenet100': [
                    transforms.RandomResizedCrop(crop),
                    # transforms.Resize(resize),
                    # transforms.RandomCrop(crop),
                    transforms.RandomHorizontalFlip()
                ],
                'nuswide': [
                    transforms.Resize(resize),
                    transforms.RandomCrop(crop),
                    transforms.RandomHorizontalFlip()
                ],
                'AID' : [
                    transforms.Resize(resize),
                    transforms.RandomCrop(crop),
                    transforms.RandomHorizontalFlip()
                ],
                'DFC15' : [
                    transforms.Resize(resize),
                    transforms.RandomCrop(crop),
                    transforms.RandomHorizontalFlip()
                ],
                'UCMD' : [
                    transforms.Resize(resize),
                    transforms.RandomCrop(crop),
                    transforms.RandomHorizontalFlip()
                ],
                'MLRS': [
                    transforms.Resize(resize),
                    transforms.RandomCrop(crop),
                    transforms.RandomHorizontalFlip()
                ]
            }[dataset_name])
        else:
            transform = compose_transform('test', resize, crop, 2)

        datafunc = {
            'imagenet100': datasets.imagenet100,
            'nuswide': datasets.nuswide,
            'AID':datasets.AID,
            'DFC15':datasets.DFC15,
            'UCMD':datasets.UCMD,
            'MLRS':datasets.MLRS
        }[dataset_name]
        d = datafunc(transform=transform, filename=filename)

    else:  # cifar10/ cifar100
        resizec = 0 if resize == 32 else resize
        cropc = 0 if crop == 32 else crop

        if transform_mode == 'train':
            transform = compose_transform('train', resizec, 0, norm, [
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.05, contrast=0.05),
            ])
        else:
            transform = compose_transform('test', resizec, cropc, norm)
        ep = config['dataset_kwargs'].get('evaluation_protocol', 1)
        d = datasets.cifar(nclass, transform=transform, filename=filename, evaluation_protocol=ep, reset=reset)

    return d


def dataloader(d, bs=256, shuffle=True, workers=-1, drop_last=True):
    if workers < 0:
        workers = default_workers
    l = DataLoader(d,
                   bs,
                   shuffle,
                   drop_last=drop_last,
                   num_workers=workers)
    return l


def seeding(seed):
    seed = int(seed)
    if seed != -1:
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True



def tensor_to_dataset(tensor, transform=None):
    class TransformTensorDataset(Dataset):
        def __init__(self, tensor, ts=None):
            super(TransformTensorDataset, self).__init__()
            self.tensor = tensor
            self.ts = ts

        def __getitem__(self, index):
            if self.ts is not None:
                return self.ts(self.tensor[index])
            return self.tensor[index]

        def __len__(self):
            return len(self.tensor)

    ttd = TransformTensorDataset(tensor, transform)
    return ttd
