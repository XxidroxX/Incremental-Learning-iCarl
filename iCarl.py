import torch
import torch.nn as nn
from iCIFAR100 import iCIFAR100
from sklearn.svm import SVC
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Subset
from ResNet import resnet32
from ResNetWithNorm import resnet32withNorm, ResNetWithNorm
import torchvision.transforms as transforms
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis
from torch.nn.utils import weight_norm
import matplotlib.pyplot as plt

# Constants:
num_epochs = 70
batch_size = 128
DEVICE = 'cuda'
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)
ROOT_FOLDER = "./data"


class Network(nn.Module):
    def __init__(self, classifier, features_extractor, neurons):
        super(Network, self).__init__()
        self.classifier = classifier
        self.features_extractor = features_extractor
        if isinstance(features_extractor, ResNetWithNorm):
            self.fc = weight_norm(nn.Linear(features_extractor.fc.in_features, 100, bias=False), name="weight")
            self.clf = nn.Sequential(
                weight_norm(nn.Linear(64, neurons), name="weight"),
                nn.ReLU(inplace=True),
                weight_norm(nn.Linear(neurons, neurons), name="weight"),
                nn.ReLU(inplace=True),
                weight_norm(nn.Linear(neurons, 100), name="weight"),
            )
        else:
            self.fc = nn.Linear(features_extractor.fc.in_features, 100, bias=False)
            self.clf = nn.Sequential(
                nn.Linear(64, neurons),
                nn.ReLU(inplace=True),
                nn.Linear(neurons, neurons),
                nn.ReLU(inplace=True),
                nn.Linear(neurons, 100)
            )
        torch.nn.init.kaiming_normal_(self.fc.weight, nonlinearity="relu")

    def forward(self, input):
        x = self.features_extractor(input)
        if self.classifier == "MLP":
            x = self.clf(x)
        else:
            x = self.fc(x)
        return x


class iCarl:
    def __init__(self, n_classes, memory=2000, classifier="NCM", loss="iCaRL", task_size=10, K=7, C=1, neurons=256, nca_dim=64,
                 examplar=True, random_exemplar=False, resnet_with_norm=False):
        super(iCarl, self).__init__()

        """ Salvo l'istanza del modello attuale """
        self.model = Network(classifier, resnet32withNorm(), neurons) if resnet_with_norm else Network(classifier, resnet32(), neurons)

        """ Salvo il vecchio modello """
        self.old_model = None

        """ Numero di classi imparate ad un certo step """
        self.n_classes = n_classes

        """ Memoria assegnata (K) per salvare gli exemplar. """
        self.memory = memory

        """ Numero di classi che processo alla volta """
        self.task_size = task_size

        """ Classificatore che uso. [NCM, KNN, SVM, MLP] """
        self.classifier = classifier

        """ True/False se fare uso o no di exemplar. """
        self.exemplar_usage = examplar

        """ Salvo i parametri dei classificatori. """
        self.K                = K
        self.neurons          = neurons
        self.resnet_with_norm = resnet_with_norm
        self.C                = C
        self.NCA_dim          = nca_dim

        """ lista delle medie per classe. """
        self.class_mean_set = []

        """ Scelgo gli exemplar casualmente o usando il criterio di icarl. """
        self.random_exemplar = random_exemplar

        """ Set degli exemplar, salvo solo gli indici """
        self.exemplar_sets = []

        """ Trasformazione per il training set"""
        self.train_transforms = transforms.Compose([transforms.RandomCrop((32, 32), padding=4),
                                                    transforms.RandomHorizontalFlip(p=0.5),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)])

        """ Trasformazioni per il test set """
        self.test_transforms = transforms.Compose([transforms.ToTensor(),
                                                   transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)])

        """ Trasformazioni per il classify """
        self.classify_transforms = transforms.Compose([transforms.RandomHorizontalFlip(p=1.),
                                                       transforms.ToTensor(),
                                                       transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)])

        """ Dataset di train e test """
        self.train_dataset = iCIFAR100(ROOT_FOLDER,
                                       train=True,
                                       transform=self.train_transforms,
                                       t1=self.test_transforms,
                                       t2=self.classify_transforms,
                                       download=True)
        self.test_dataset = iCIFAR100(ROOT_FOLDER,
                                      train=False,
                                      transform=self.test_transforms,
                                      download=True)

        """ Dataloader """
        self.train_loader = None
        self.test_loader = None

        """ Loss functions """
        self.loss_type = loss
        self.BCE = nn.BCEWithLogitsLoss()
        self.KLDIV = nn.KLDivLoss(reduction="batchmean")
        self.MSE = nn.MSELoss()
        self.L1 = nn.L1Loss()
        self.NLLoss = nn.NLLLoss()
        self.CE = nn.CrossEntropyLoss()

        """ Scelgo l'ordine delle 100 classi. """
        np.random.seed(1993)  # Fix the random seed
        self.classes = np.arange(100)
        np.random.shuffle(self.classes)

    def beforeTrain(self):
        """
        Procedure da eseguire prima del train come ad esempio incrementare il layer FC.
        :return: Void
        """
        self._update_dataloaders()
        self.model.to(DEVICE)

    def train(self):
        # Definisco l'optimizer e lo scheduler
        optimizer = optim.SGD(self.model.parameters(), lr=2., momentum=0.9, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[48, 62], gamma=0.2)

        for epoch in range(num_epochs):
            running_loss = 0.0
            self.model.train()
            for images, labels in self.train_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()

                if self.loss_type == "iCaRL":
                    loss_value = self._compute_loss(images, labels)
                elif self.loss_type == "KLDiv, KLDiv":
                    loss_value = self._kldiv_kldiv_loss(images, labels)
                elif self.loss_type == "KLDiv, BCEWL":
                    loss_value = self._kldiv_bce_loss(images, labels)
                elif self.loss_type == "MSE, MSE":
                    loss_value = self._mse_mse_loss(images, labels)
                elif self.loss_type == "MSE, BCEWL":
                    loss_value = self._mse_bce_loss(images, labels)
                elif self.loss_type == "CE, BCEWL":
                    loss_value = self._ce_bce_loss(images, labels)
                elif self.loss_type == "MSE, MSE":
                    loss_value = self._mse_mse_loss(images, labels)
                elif self.loss_type == "CE_W, BCEWL":
                    loss_value = self._ce_weights_bce_loss(images, labels)

                loss_value.backward()
                optimizer.step()
                running_loss += loss_value.item() * images.size(0)

            epoch_loss = running_loss / len(self.train_loader.dataset)
            accuracy = self._test()
            self.model.train()
            print('epoch:%d/%d,loss:%f,accuracy (FC layer):%.3f,LR=%s' % (
            epoch + 1, num_epochs, epoch_loss, accuracy, scheduler.get_last_lr()))
            scheduler.step()
        self.model.eval()

    def afterTrain(self):
        """
        Eseguo le operazioni sugli exemplar ed eseguo il test finale.
        :return:
        """
        self.model.eval()
        m = self.memory // self.n_classes
        if self.exemplar_usage:
            self._compute_exemplar_class_mean()
            self._reduce_exemplar_sets(m)
            for i in self.classes[self.n_classes - self.task_size: self.n_classes]:
                print('construct class %s examplar:' % i, end='')
                images, indexes, _ = self.train_dataset.get_images_by_class(i)
                self._construct_exemplar_set(images, indexes, m)

        # self.model.train()
        accuracy = self._test(True)

        self.model.eval()
        self.old_model = Network(self.classifier, resnet32withNorm(), self.neurons) if self.resnet_with_norm else Network(self.classifier, resnet32(), self.neurons)
        self.old_model.load_state_dict(self.model.state_dict())
        self.old_model = self.old_model.to(DEVICE)
        self.old_model.eval()

        self.n_classes += self.task_size
        print(self.classifier + " accuracy：" + str(accuracy))

    def _test(self, final_step=False):

        self.model.eval()  # Set Network to evaluation mode

        running_corrects = 0
        if final_step and self.classifier == "KNN":
            self._update_KNN()
        elif final_step and self.classifier == "SVN":
            self._update_SVN()
        elif final_step and self.classifier == "KNN+NCA":
            self._update_KNN_NCA()
        for images, labels in self.test_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            with torch.no_grad():
                if self.classifier == "NCM" and final_step:
                    preds = self.classify(images).to(DEVICE)
                elif self.classifier == "KNN" and final_step:
                    preds = self.knn_classify(images).to(DEVICE)
                elif self.classifier == "KNN+NCA" and final_step:
                    preds = self.knn_nca_classify(images).to(DEVICE)
                elif self.classifier == "SVM" and final_step:
                    preds = self.SVN_classify(images).to(DEVICE)
                else:
                    # Forward Pass
                    outputs = self.model(images)

                    # Get predictions
                    _, preds = torch.max(outputs.data, 1)

            # Update Corrects
            running_corrects += torch.sum(preds == labels.data).data.item()

        # Calculate Accuracy
        return 100 * running_corrects / float(len(self.test_loader.dataset))

    def _compute_exemplar_class_mean(self):
        """
        Compute the mean of all the exemplars.
        :return: None
        """
        self.class_mean_set = []
        self.model.eval()
        if self.n_classes > self.task_size:
            for label, P_y in enumerate(self.exemplar_sets):
                images_1, images_2 = self.train_dataset.get_images_by_indexes(P_y)
                class_mean_1, _ = self.compute_class_mean(images_1)
                class_mean_2, _ = self.compute_class_mean(images_2)
                class_mean = (class_mean_1 + class_mean_2) / 2
                class_mean = class_mean.data / class_mean.norm()
                self.class_mean_set.append(class_mean)
        for i in self.classes[self.n_classes - self.task_size: self.n_classes]:
            images_1, _, images_2 = self.train_dataset.get_images_by_class(i)
            class_mean_1, _ = self.compute_class_mean(images_1)
            class_mean_2, _ = self.compute_class_mean(images_2)
            class_mean = (class_mean_1 + class_mean_2) / 2
            class_mean = class_mean.data / class_mean.norm()
            self.class_mean_set.append(class_mean)

    def _construct_exemplar_set(self, images, ind, m):
        """
        Costruisco il set degli exemplar basato sugli indici.
        :param images: tutte le immagini di quella classe
        :param m: numero di immagini da salvare
        :return:
        """
        self.model.eval()
        if self.random_exemplar:
            Py = list()
            np.random.seed()
            indexes = np.arange(len(images))
            np.random.shuffle(indexes)
            for i in range(m):
                Py.append(ind[indexes[i]])
        else:
            images = torch.stack(images).to(DEVICE)
            with torch.no_grad():
                phi_X = torch.nn.functional.normalize(self.model.features_extractor(images)).cpu()

            mu_y = phi_X.mean(dim=0)  # vettore di 64 colonne
            mu_y.data = mu_y.data / mu_y.data.norm()

            Py = []
            # Accumulates sum of exemplars
            sum_taken_exemplars = torch.zeros(1, 64)

            indexes = list()
            for k in range(1, int(m + 1)):
                asd = nn.functional.normalize((1 / k) * (phi_X + sum_taken_exemplars))
                mean_distances = (mu_y - asd).norm(dim=1)
                used = -1
                a, indici = torch.sort(mean_distances)
                for item in a:
                    mins = (mean_distances == item).nonzero()
                    for j in mins:
                        if j not in indexes:
                            indexes.append(j)
                            Py.append(ind[j])
                            used = j
                            sum_taken_exemplars += phi_X[j]
                            break
                    if used != -1:
                        break

        print(len(Py))
        self.exemplar_sets.append(Py)

    def compute_class_mean(self, images):
        """
        Passo tutte le immagini di una determinata classe e faccio la media.
        :param special_transform:
        :param images: tutte le immagini della classe x
        :return: media della classe e features extractor.
        """
        self.model.eval()
        images = torch.stack(images).to(DEVICE)  # 500x3x32x32  #stack vs cat. Il primo le attacca in una nuova dim. 3x4 diventa 1x3x4.
        # cat invece le fa diventare 6x4
        with torch.no_grad():
            phi_X = torch.nn.functional.normalize(self.model.features_extractor(images))

        # phi_X.shape = 500x64
        mean = phi_X.mean(dim=0)
        mean.data = mean.data / mean.data.norm()
        return mean, phi_X

    def _reduce_exemplar_sets(self, images_per_class):
        for index in range(len(self.exemplar_sets)):
            self.exemplar_sets[index] = self.exemplar_sets[index][:images_per_class]
            print('Reduce size of class %d to %s examplar' % (self.classes[index], str(len(self.exemplar_sets[index]))))

    def _update_dataloaders(self):
        """
        Aggiorno i dataloader con le nuove immagini/labels delle nuove classi.
        :rtype: object
        """
        train_indexes = []
        if self.exemplar_usage:
            for i in self.exemplar_sets:
                train_indexes.extend(i)
        train_indexes.extend(self.train_dataset.get_indexes_by_classes(self.classes[self.n_classes - self.task_size: self.n_classes]))

        self.train_loader = DataLoader(dataset=Subset(self.train_dataset, train_indexes),
                                       shuffle=True,
                                       num_workers=4,
                                       batch_size=128)
        print(len(self.train_loader.dataset))
        test_indexes = self.test_dataset.get_indexes_by_classes(self.classes[:self.n_classes])
        self.test_loader = DataLoader(dataset=Subset(self.test_dataset, test_indexes),
                                      shuffle=False,
                                      num_workers=4,
                                      batch_size=128)
        print(len(self.test_loader.dataset))

    def _compute_loss(self, images, target):
        """
        Calcolo la loss usando la BCEWithLogits singola (senza usarne 2 separate)
        :param images: 128 immagini da processare
        :param target: 128 true labels
        :return: la loss
        """
        self.model.train()
        output = self.model(images)
        target = self.to_onehot(target, 100)
        output, target = output.to(DEVICE), target.to(DEVICE)
        if self.old_model is None:
            return self.BCE(output, target)
        else:
            with torch.no_grad():
                old_target = torch.sigmoid(self.old_model(images))

            n_c = self.classes[:self.n_classes - self.task_size]
            target[:, n_c] = old_target[:, n_c]
            return self.BCE(output, target)

    def _kldiv_kldiv_loss(self, images, target):
        self.model.train()
        output = self.model(images)
        target = self.to_onehot(target, 100)
        log_max = nn.LogSoftmax(dim=1)
        output, target = output.to(DEVICE), target.to(DEVICE)
        loss = self.KLDIV(log_max(output), target)
        if self.old_model is not None:
            with torch.no_grad():
                old_target = nn.functional.softmax(self.old_model(images), dim=1)

            n_c = self.classes[:self.n_classes - self.task_size]
            loss += self.KLDIV(log_max(output[:, n_c]), old_target[:, n_c])
        return loss

    def _kldiv_bce_loss(self, images, target):
        self.model.train()
        output = self.model(images)
        target = self.to_onehot(target, 100)
        log_max = nn.LogSoftmax(dim=1)
        output, target = output.to(DEVICE), target.to(DEVICE)
        loss = self.KLDIV(log_max(output), target)
        if self.old_model is not None:
            with torch.no_grad():
                old_target = torch.sigmoid(self.old_model(images))

            n_c = self.classes[:self.n_classes - self.task_size]
            loss += self.BCE(output[:, n_c], old_target[:, n_c])

        return loss

    def _ce_bce_loss(self, images, target):
        self.model.train()
        output = self.model(images)
        output, target = output.to(DEVICE), target.to(DEVICE)
        loss = self.CE(output, target)
        if self.old_model is not None:
            with torch.no_grad():
                old_target = torch.sigmoid(self.old_model(images))

            n_c = self.classes[:self.n_classes - self.task_size]
            loss += self.BCE(output[:, n_c], old_target[:, n_c])

        return loss

    def _mse_bce_loss(self, images, target):
        self.model.train()
        output = self.model(images)
        target = self.to_onehot(target, 100)
        output, target = output.to(DEVICE), target.to(DEVICE)
        loss = self.MSE(nn.functional.softmax(output, dim=1), target)
        if self.old_model is not None:
            with torch.no_grad():
                old_target = torch.sigmoid(self.old_model(images))

            n_c = self.classes[:self.n_classes - self.task_size]
            loss += self.BCE(output[:, n_c], old_target[:, n_c])

        return loss

    # ce pesaate +bce
    def _mse_mse_loss(self, images, target):
        self.model.train()
        output = self.model(images)
        target = self.to_onehot(target, 100)
        output, target = output.to(DEVICE), target.to(DEVICE)
        loss = self.MSE(nn.functional.softmax(output, dim=1), target)
        if self.old_model is not None:
            with torch.no_grad():
                old_target = nn.functional.softmax(self.old_model(images), dim=1)

            n_c = self.classes[:self.n_classes - self.task_size]
            loss += self.MSE(nn.functional.softmax(output[:, n_c], dim=1), old_target[:, n_c])

        return loss

    def _get_weights(self):
        weights = torch.zeros(100)
        old_weights = 500 / (self.memory // self.n_classes)
        for i in self.classes[:self.n_classes - self.task_size]:
            weights[i] = old_weights
        for i in self.classes[self.n_classes - self.task_size:]:
            weights[i] = 1
        return weights

    def _ce_weights_bce_loss(self, images, target):
        self.model.train()
        output = self.model(images)

        if self.n_classes != self.task_size:
            self.CE = nn.CrossEntropyLoss(weight=self._get_weights().to(DEVICE))
        output, target = output.to(DEVICE), target.to(DEVICE)
        loss = self.CE(output, target)
        if self.old_model is not None:
            with torch.no_grad():
                old_target = torch.sigmoid(self.old_model(images))

            n_c = self.classes[:self.n_classes - self.task_size]
            loss += self.BCE(output[:, n_c], old_target[:, n_c])
        return loss

    def _l1_bce_loss(self, images, target):
        self.model.train()
        output = self.model(images)
        target = self.to_onehot(target, 100)
        output, target = output.to(DEVICE), target.to(DEVICE)
        loss = self.L1(nn.functional.softmax(output, dim=1), target)
        if self.old_model is not None:
            with torch.no_grad():
                old_target = torch.sigmoid(self.old_model(images))

            n_c = self.classes[:self.n_classes - self.task_size]
            loss += self.BCE(output[:, n_c], old_target[:, n_c])

        return loss

    @staticmethod
    def to_onehot(targets, n_classes):
        return torch.eye(n_classes)[targets]

    def classify(self, images):
        # batch_sizex3x32x32
        result = []
        self.model.eval()
        with torch.no_grad():
            phi_X = nn.functional.normalize(self.model.features_extractor(images))

        ex_means = torch.stack(self.class_mean_set)
        # 10x64 (di ogni classe mi salvo la media di ogni features)
        for x in phi_X:
            # x: 64. media delle features di quella immagine
            distances_from_class = (ex_means - x).norm(dim=1)  # giusto
            y = distances_from_class.argmin()
            result.append(self.classes[y])
        return torch.tensor(result)

    def _update_KNN(self):
        self.knn = KNeighborsClassifier(n_neighbors=self.K)
        self.model.eval()

        labels = list()
        images = None
        for label in range(0, len(self.exemplar_sets)):
            image, _ = self.train_dataset.get_images_by_indexes(self.exemplar_sets[label])
            image = torch.stack(image).to(DEVICE)
            with torch.no_grad():
                image = torch.nn.functional.normalize(self.model.features_extractor(image)).cpu()
            if label == 0:
                images = image
            else:
                images = torch.cat((images, image), 0)
            labels.extend([self.classes[label]] * len(image))

        self.knn.fit(images, labels)

    def _update_KNN_NCA(self):
        self.knn = KNeighborsClassifier(n_neighbors=self.K)
        self.nca = NeighborhoodComponentsAnalysis(n_components=self.NCA_dim)
        self.model.eval()

        labels = list()
        images = None
        for label in range(0, len(self.exemplar_sets)):
            image, _ = self.train_dataset.get_images_by_indexes(self.exemplar_sets[label])
            image = torch.stack(image).to(DEVICE)
            with torch.no_grad():
                image = torch.nn.functional.normalize(self.model.features_extractor(image)).cpu()
            if label == 0:
                images = image
            else:
                images = torch.cat((images, image), 0)
            labels.extend([self.classes[label]] * len(image))
        # plt.figure()
        self.nca.fit(images, labels)
        self.knn.fit(self.nca.transform(images), labels)
        # X_embedded = self.nca.transform(images)
        # plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=labels, s=30, cmap='Set1')
        # plt.show()

    def knn_classify(self, test_images):
        self.model.eval()
        with torch.no_grad():
            phi_X = torch.nn.functional.normalize(self.model.features_extractor(test_images)).cpu()  # nx64
        y_pred = self.knn.predict(phi_X)
        return torch.tensor(y_pred)

    def knn_nca_classify(self, test_images):
        self.model.eval()
        with torch.no_grad():
            phi_X = torch.nn.functional.normalize(self.model.features_extractor(test_images)).cpu()  # nx64
        y_pred = self.knn.predict(self.nca.transform(phi_X))
        return torch.tensor(y_pred)

    def _update_SVN(self):
        self.svn = SVC(C=self.C)
        self.model.eval()

        labels = list()
        images = None
        for label in range(0, len(self.exemplar_sets)):
            image, _ = self.train_dataset.get_images_by_indexes(self.exemplar_sets[label])
            image = torch.stack(image).to(DEVICE)
            with torch.no_grad():
                image = torch.nn.functional.normalize(self.model.features_extractor(image)).cpu()
            if label == 0:
                images = image
            else:
                images = torch.cat((images, image), 0)
            labels.extend([self.classes[label]] * len(image))
        self.svn.fit(images, labels)

    def SVN_classify(self, test_images):
        self.model.eval()
        with torch.no_grad():
            phi_X = torch.nn.functional.normalize(self.model.features_extractor(test_images)).cpu()  # nx64

        y_pred = self.svn.predict(phi_X)
        return torch.tensor(y_pred)
