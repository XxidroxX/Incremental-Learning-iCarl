from PIL import Image
import os
import os.path
import numpy as np
import pickle

from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import check_integrity, download_and_extract_archive
from sklearn.model_selection import StratifiedShuffleSplit


class CIFAR100(VisionDataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }
    # DEVO RITORNARE UNA LISTA DI INDICI PER OGNI CLASSE.
    def split_percentage(self, p):
        train_list = dict()
        val_list = dict()
        for k, v in self.dataset.items():
            num = len(v)
            sss = StratifiedShuffleSplit(n_splits=1, train_size=p, random_state=42)
            for ti, tti in sss.split(v, [0] * num):
                train_index = ti
                test_index = tti
            
            train_list[k] = [v[i] for i in train_index]
            val_list[k]   = [v[i] for i in test_index]
        
        return train_list, val_list
    
    # DEVO RITORNARE UNA LISTA DI INDICI PER OGNI CLASSE.
    def get_test_indexes(self):
        return self.dataset
    
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):

        super(CIFAR100, self).__init__(root, transform=transform, target_transform=target_transform)

        self.train = train  # training set or test set
        
        self.dataset = dict()
        
        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' + ' You can use download=True to download it')

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data    = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])  #<-- ID DELLA CLASSE
        
        for index, (data, target) in enumerate(zip(self.data[0], self.targets)):
            if target in self.dataset.keys():
                self.dataset.get(target).append(index)
            else:
                self.dataset[target] = [index]
        
        if train == True: 
            train_list = list()
            val_list = list()
            for k, v in self.dataset.items():
                num = len(v)
                sss = StratifiedShuffleSplit(n_splits=1, train_size=0.5)
                for ti, tti in sss.split(v, [0] * num):
                    train_index = ti
                    test_index = tti
                for i in train_index:
                    train_list.append(v[i])
                for i in test_index:
                    val_list.append(v[i])
            
        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self._load_meta()

    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        if not check_integrity(path, self.meta['md5']):
            raise RuntimeError('Dataset metadata file not found or corrupted.' + ' You can use download=True to download it')
        with open(path, 'rb') as infile:
            data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]

        # dictionary: {class: class_id}
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")
    
