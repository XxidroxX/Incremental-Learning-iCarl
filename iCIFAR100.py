from PIL import Image
from torchvision.datasets import CIFAR100


class iCIFAR100(CIFAR100):

    def __init__(self, root, train=True, t1=None, t2=None, transform=None, target_transform=None, download=False):

        super(iCIFAR100, self).__init__(root, train=train, download=download, transform=transform, target_transform=target_transform)

        self.t1 = t1
        self.t2 = t2
        self.train = train  # training set or test set
        self.class_to_idx = dict()

        for index, (img, label) in enumerate(zip(self.data, self.targets)):
            if label in self.class_to_idx.keys():
                self.class_to_idx[label].append(index)
            else:
                self.class_to_idx[label] = list()
                self.class_to_idx[label].append(index)


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

    def get_images_by_indexes(self, indexes):
        """
        Retrieve all the images from a given list of indexes.
        :param indexes: list of all indexes.
        :return: Two list with the same images but with different transformations.
        """
        images1, images2 = list(), list()
        for i in indexes:
            images1.append(self.t1(Image.fromarray(self.data[i])))
            images2.append(self.t2(Image.fromarray(self.data[i])))
        return images1, images2

    def get_indexes_by_classes(self, classes):
        """
        Retrieve all the indexes of the images from a given range of classes.
        :param classes: range of classes.
        :return: a list of indexes.
        """
        a = list()
        for i in classes:
            a.extend(self.class_to_idx[i])
        return a

    def get_images_by_class(self, label):
        """
        Retrieve all images of that class "label"
        :param label: class
        :return: three list with all images with trans1, with trans2 and all the indexes.
        """
        a, b, c = list(), list(), list()
        for index, i in enumerate(self.data):
            if self.targets[index] == label:
                a.append(self.t1(Image.fromarray(i)))
                b.append(index)
                c.append(self.t2(Image.fromarray(i)))
        return a, b, c
