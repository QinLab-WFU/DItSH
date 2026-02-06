import os

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.datasets.folder import pil_loader
from PIL import Image
from tqdm import tqdm
DATA_FOLDER = {
    'nuswide': 'data/nuswide_v2_256',  # resize to 256x256
    'imagenet': 'data/imagenet_resize',  # resize to 224x224
    'cifar': 'data/cifar',  # auto generate
    'AID' : 'data/AID',
    'DFC15' : 'data/DFC15',
    'UCMD' : 'data/UCMD',
    'MLRS' : './data'
}

class MLRSs(Dataset):
    """
    Flicker 25k dataset.

    Args
        root(str): Path of dataset.
        mode(str, 'train', 'query', 'retrieval'): Mode of dataset.
        transform(callable, optional): Transform images.
    """

    def __init__(self, root, mode, transform=None):
        self.root = root
        self.transform = transform
        # self.diff = None
        if mode == 'train':
            self.data = [Image.open(os.path.join(root, 'MLRS', i)).convert('RGB') for i in MLRS.TRAIN_DATA]
            self.targets = MLRS.TRAIN_TARGETS
            # self.targets.dot(self.targets.T) == 0
        elif mode == 'query':
            self.data = [Image.open(os.path.join(root, 'MLRS', i)).convert('RGB') for i in MLRS.QUERY_DATA]
            self.targets = MLRS.QUERY_TARGETS
        elif mode == 'retrieval':
            self.data = [Image.open(os.path.join(root, 'MLRS', i)).convert('RGB') for i in
                         MLRS.RETRIEVAL_DATA]
            self.targets = MLRS.RETRIEVAL_TARGETS
        else:
            raise ValueError(r'Invalid arguments: mode, can\'t load dataset!')

    def __getitem__(self, index):
        img = self.data[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, self.targets[index]

    def __len__(self):
        return len(self.data)

    def get_targets(self):
        return torch.FloatTensor(self.targets)

    @staticmethod
    def init(root, num_query, num_train):
        # Load dataset
        img_txt_path = os.path.join(root, 'img.txt')
        targets_txt_path = os.path.join(root, 'targets.txt')

        # Read files
        with open(img_txt_path, 'r') as f:
            data = np.array([i.strip() for i in f])
        targets = np.loadtxt(targets_txt_path, dtype=int)

        # Split dataset
        with open(img_txt_path, 'r') as f:
            data = np.array([i.strip() for i in f])
        targets = np.loadtxt(targets_txt_path, dtype=int)

        # Split dataset
        perm_index = np.random.permutation(data.shape[0])
        query_index = perm_index[:num_query]
        train_index = perm_index[num_query: num_query + num_train]
        retrieval_index = perm_index[num_query:]

        MLRS.QUERY_DATA = data[query_index]
        MLRS.QUERY_TARGETS = targets[query_index, :]

        MLRS.TRAIN_DATA = data[train_index]
        MLRS.TRAIN_TARGETS = targets[train_index, :]

        MLRS.RETRIEVAL_DATA = data[retrieval_index]
        MLRS.RETRIEVAL_TARGETS = targets[retrieval_index, :]



class HashingDataset(Dataset):
    def __init__(self, root,
                 transform=None,
                 target_transform=None,
                 filename='train',
                 separate_multiclass=False):
        self.loader = pil_loader
        self.separate_multiclass = separate_multiclass
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.filename = filename
        self.train_data = []
        self.train_labels = []

        filename = os.path.join(self.root, self.filename)

        with open(filename, 'r') as f:
            while True:
                lines = f.readline()
                if not lines:
                    break

                path_tmp = lines.split()[0]
                label_tmp = lines.split()[1:]
                self.is_onehot = len(label_tmp) != 1
                if not self.is_onehot:
                    label_tmp = lines.split()[1]
                if self.separate_multiclass:
                    assert self.is_onehot, 'if multiclass, please use onehot'
                    nonzero_index = np.nonzero(np.array(label_tmp, dtype=int))[0]
                    for c in nonzero_index:
                        self.train_data.append(path_tmp)
                        label_tmp = ['1' if i == c else '0' for i in range(len(label_tmp))]
                        self.train_labels.append(label_tmp)
                else:
                    self.train_data.append(path_tmp)
                    self.train_labels.append(label_tmp)

        self.train_data = np.array(self.train_data)
        self.train_labels = np.array(self.train_labels, dtype=float)

        print(f'Number of data: {self.train_data.shape[0]}')

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.train_data[index], self.train_labels[index]
        target = torch.tensor(target)

        img = self.loader(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.train_data)


def one_hot(nclass):
    def f(index):
        index = torch.tensor(int(index)).long()
        return torch.nn.functional.one_hot(index, nclass)

    return f


def cifar(nclass, **kwargs):
    transform = kwargs['transform']
    ep = kwargs['evaluation_protocol']
    fn = kwargs['filename']
    reset = kwargs['reset']

    prefix = DATA_FOLDER['cifar']

    CIFAR = CIFAR10 if int(nclass) == 10 else CIFAR100
    traind = CIFAR(f'{prefix}{nclass}',
                   transform=transform, target_transform=one_hot(int(nclass)),
                   train=True, download=True)
    testd = CIFAR(f'{prefix}{nclass}', train=False, download=True)

    combine_data = np.concatenate([traind.data, testd.data], axis=0)
    combine_targets = np.concatenate([traind.targets, testd.targets], axis=0)

    path = f'{prefix}{nclass}/0_{ep}_{fn}'

    load_data = fn == 'train.txt'
    load_data = load_data and (reset or not os.path.exists(path))

    if not load_data:
        print(f'Loading {path}')
        data_index = torch.load(path)
    else:
        train_data_index = []
        query_data_index = []
        db_data_index = []

        data_id = np.arange(combine_data.shape[0])  # [0, 1, ...]

        for i in range(nclass):
            class_mask = combine_targets == i
            index_of_class = data_id[class_mask].copy()  # index of the class [2, 10, 656,...]
            np.random.shuffle(index_of_class)

            if ep == 1:
                query_n = 100  # // (nclass // 10)
                train_n = 500  # // (nclass // 10)

                index_for_query = index_of_class[:query_n].tolist()

                index_for_db = index_of_class[query_n:].tolist()
                index_for_train = index_for_db[:train_n]
            else:  # ep2 = take all data
                query_n = 1000  # // (nclass // 10)

                index_for_query = index_of_class[:query_n].tolist()
                index_for_db = index_of_class[query_n:].tolist()
                index_for_train = index_for_db

            train_data_index.extend(index_for_train)
            query_data_index.extend(index_for_query)
            db_data_index.extend(index_for_db)

        train_data_index = np.array(train_data_index)
        query_data_index = np.array(query_data_index)
        db_data_index = np.array(db_data_index)

        torch.save(train_data_index, f'data/cifar{nclass}/0_{ep}_train.txt')
        torch.save(query_data_index, f'data/cifar{nclass}/0_{ep}_test.txt')
        torch.save(db_data_index, f'data/cifar{nclass}/0_{ep}_database.txt')

        data_index = {
            'train.txt': train_data_index,
            'test.txt': query_data_index,
            'database.txt': db_data_index
        }[fn]

    traind.data = combine_data[data_index]
    traind.targets = combine_targets[data_index]

    return traind

def imagenet100(**kwargs):
    transform = kwargs['transform']
    filename = kwargs['filename']

    d = HashingDataset(DATA_FOLDER['imagenet'], transform=transform, filename=filename)
    return d


def AID(**kwargs):
    transform = kwargs['transform']
    filename = kwargs['filename']
    separate_multiclass = kwargs.get('separate_multiclass', False)

    d = HashingDataset(DATA_FOLDER['AID'],
                       transform=transform,
                       filename=filename,
                       separate_multiclass=separate_multiclass)
    return d


def nuswide(**kwargs):
    transform = kwargs['transform']
    filename = kwargs['filename']
    separate_multiclass = kwargs.get('separate_multiclass', False)

    d = HashingDataset(DATA_FOLDER['nuswide'],
                       transform=transform,
                       filename=filename,
                       separate_multiclass=separate_multiclass)
    return d

def DFC15(**kwargs):
    transform = kwargs['transform']
    filename = kwargs['filename']
    separate_multiclass = kwargs.get('separate_multiclass', False)

    d = HashingDataset(DATA_FOLDER['DFC15'],
                       transform=transform,
                       filename=filename,
                       separate_multiclass=separate_multiclass)
    return d
def UCMD(**kwargs):
    transform = kwargs['transform']
    filename = kwargs['filename']
    separate_multiclass = kwargs.get('separate_multiclass', False)

    d = HashingDataset(DATA_FOLDER['UCMD'],
                       transform=transform,
                       filename=filename,
                       separate_multiclass=separate_multiclass)
    return d

def MLRS(**kwargs):
    transform = kwargs['transform']
    # filename = kwargs['filename']
    # separate_multiclass = kwargs.get('separate_multiclass', False)
    # mlrs = MLRS()
    MLRSs.init('./data/MLRS/', 1000, 5000)
    d = MLRSs(root=DATA_FOLDER['MLRS'],mode='train',transform=transform)
    # query = MLRSs(root=DATA_FOLDER['MLRS'],mode='query',transform=transform)
    # retrieval = MLRSs(root=DATA_FOLDER['MLRS'],mode='retrieval',transform=transform)
    # d = (train,query,retrieval)
    # print(d)?
    return d