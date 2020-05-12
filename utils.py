import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset, DataLoader
import numpy as np
from CIFAR100 import CIFAR100
from operator import itemgetter

# COSTANTI
ROOT_FOLDER   = "./data"    # DOVE SCARICAR IL DATASET
DEVICE        = 'cuda'       # 'cuda' or 'cpu'
NUM_CLASSES   = 100         # 
BATCH_SIZE    = 256         # Higher batch sizes allows for larger learning rates. 
                            # An empirical heuristic suggests that, when changing the batch size, learning rate should change by the same factor to have comparable results
LR            = 1e-3        # The initial Learning Rate                                              
MOMENTUM      = 0.9         # Hyperparameter for SGD, keep this at 0.9 when using SGD
WEIGHT_DECAY  = 5e-5        # Regularization, you can keep this at the default
NUM_EPOCHS    = 30          # Total number of training epochs (iterations over dataset)              
STEP_SIZE     = 20          # How many epochs before decreasing learning rate (if using a step-down policy)
GAMMA         = 0.1         # Multiplicative factor for learning rate step-down
LOG_FREQUENCY = 10

## LE HO TROVATE SU INTERNET E USANDO LA FUNZIONE CHE LE CALCOLA IN AUTOMATICO
CIFAR100_MEAN = (0.507, 0.487, 0.441)
CIFAR100_STD  = (0.267, 0.256, 0.276)

def gen_mean_std(dataset):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=50000, shuffle=False, num_workers=2)
    train = iter(dataloader).next()[0]
    mean = np.mean(train.numpy(), axis=(0, 2, 3))
    std = np.std(train.numpy(), axis=(0, 2, 3))
    return mean, std

#------------------------------
# Carico il train_dataloader e val_dataloader.
#------------------------------
def get_train_val_dataloader(classes):
    train_dataset = CIFAR100(
        root=ROOT_FOLDER,
        train=True,
        download=True,
        transform=transforms.Compose([
            # VEDERE QUALE RETE USARE E CROPPARE DI CONSEGUENZA
            #transforms.RandomCrop(32, padding=4),
            #transforms.RandomHorizontalFlip(), data augmentation
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ]))
    # train_indexes[0]: indici prima classe ecc...
    train_indexes, val_indexes = train_dataset.split_percentage(0.5)
    
    if isinstance(classes, int):    
        train_dataloader = DataLoader(Subset(train_dataset, train_indexes.get(classes)), batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)
        val_dataloader   = DataLoader(Subset(train_dataset, val_indexes.get(classes)), batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    else:
        # E' UN INTERVALLO
        train_dataloader = DataLoader(Subset(train_dataset, [i for sublist in itemgetter(*classes)(train_indexes) for i in sublist]), batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)
        val_dataloader   = DataLoader(Subset(train_dataset, [i for sublist in itemgetter(*classes)(val_indexes)   for i in sublist]), batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Check dataset sizes
    print('Train Dataset: {}'.format(len(train_dataloader.dataset)))
    print('Valid Dataset: {}'.format(len(val_dataloader.dataset)))
    
    return train_dataloader, val_dataloader

#------------------------------
# Carico il test_dataloader.
#------------------------------
def get_test_dataloader(classes):
    test_dataset = CIFAR100(
        root=ROOT_FOLDER,
        train=False,
        download=True,
        transform=transforms.Compose([
            # VEDERE QUALE RETE USARE E CROPPARE DI CONSEGUENZA
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ]))
    test_indexes = test_dataset.get_test_indexes()
    if isinstance(classes, int):
        loader = DataLoader(Subset(test_dataset, test_indexes.get(classes)), batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    else:
        loader = DataLoader(Subset(test_dataset, [i for sublist in itemgetter(*classes)(test_indexes) for i in sublist]), batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    print("Test Dataset: {}".format(len(loader.dataset)))
    return loader