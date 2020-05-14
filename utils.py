import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset, DataLoader
import numpy as np
from CIFAR100 import CIFAR100
from operator import itemgetter
import time
import copy


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

TITLE         = "BATCH_SIZE="+str(BATCH_SIZE)+", LR="+str(LR)+", NUM_EPOCHS="+str(NUM_EPOCHS)+", STEP_SIZE="+str(STEP_SIZE) 

## LE HO TROVATE SU INTERNET E USANDO LA FUNZIONE CHE LE CALCOLA IN AUTOMATICO
CIFAR100_MEAN = (0.507, 0.487, 0.441)
CIFAR100_STD  = (0.267, 0.256, 0.276)

""" 
    CALCULATE MEAN AND STD OF CIFAR100 DATASET
"""
def gen_mean_std(dataset):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=50000, shuffle=False, num_workers=2)
    train = iter(dataloader).next()[0]
    mean = np.mean(train.numpy(), axis=(0, 2, 3))
    std = np.std(train.numpy(), axis=(0, 2, 3))
    return mean, std

""" 
    LOAD TRAIN AND VALIDATION DATALOADER
"""
def get_train_val_dataloader(classes):
    train_dataset = CIFAR100(
        root=ROOT_FOLDER,
        train=True,
        download=True,
        transform=transforms.Compose([
            # VEDERE QUALE RETE USARE E CROPPARE DI CONSEGUENZA
            transforms.Resize(224),
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

""" 
    LOAD TEST DATALOADER 
"""
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

""" 
    SAVE RESULTS IN A FILE
"""
def save_result(title, accuracy_val, accuracy_train, loss_val, loss_train):
    with open(title+'.txt', 'w') as f:
        f.write("%s\n"   % str(title))
        f.write("%s\n"   % str(accuracy_val).strip('[]'))
        f.write("%s\n"   % str(accuracy_train).strip('[]'))
        f.write("%s\n"   % str(loss_val).strip('[]'))
        f.write("%s\n\n" % str(loss_train).strip('[]'))

""" 
    TRAIN/VALIDATION FUNCTION
    return the best model (the model with the highest accuracy on the validation set.)
"""
def train_model(model, train_dataloader, val_dataloader, criterion, optimizer, scheduler):
    since = time.time()

    val_acc_history = []
    val_loss_history = []
    train_acc_history = []
    train_loss_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(NUM_EPOCHS):
        print('Epoch {}/{}/LR={}'.format(epoch, NUM_EPOCHS - 1, scheduler.get_lr()))
        print('-' * 100)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                dl = train_dataloader
            else:
                model.train(False)   # Set model to evaluate mode
                dl = val_dataloader

            running_loss = 0.0
            running_corrects = 0
            
            # Iterate over data.
            for inputs, labels in dl:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss     += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()
            epoch_loss = running_loss / len(dl.dataset)
            epoch_acc = running_corrects.double() / len(dl.dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val':
                val_acc_history.append(epoch_acc)
                val_loss_history.append(epoch_loss)
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'train':
                train_acc_history.append(epoch_acc)
                train_loss_history.append(epoch_loss)
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    model.load_state_dict(best_model_wts)
    #save_result(TITLE, val_acc_history, train_acc_history, val_loss_history, train_loss_history)
    return model, train_acc_history, train_loss_history, val_acc_history, val_loss_history