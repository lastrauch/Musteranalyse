import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset


class img_dataset(Dataset):
    def __init__(self, vocab):
        train, test, classes, inputs, class_names, img_vocab, word_to_img = self.load_data(vocab)

        self.train = train
        self.classes = classes
        self.inputs = inputs
        self.class_names = class_names
        self.img_vocab = img_vocab
        self.word_to_img = word_to_img

    def __len__(self):
        return self.inputs.shape[1]

    def load_data(self, vocab):
        train, test, classes = self.transforming()
        inputs, class_names = next(iter(train))
        img_vocab = self. img_vocab(vocab, class_names, inputs, classes)
        word_to_img = self.map_word_to_image(class_names, vocab, classes)

        return train, test, classes, inputs, class_names, img_vocab, word_to_img

    def transforming(self):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=10, shuffle=True, num_workers=2)
        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=10, shuffle=False, num_workers=2)
        class_names = trainset.classes
        #print(class_names)

        return trainloader, testloader, class_names

    def imshow(self, img, title= None):
        img = img.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std*img+mean
        img = np.clip(img, 0, 1)
        plt.imshow(img)
        if title is not None:
            plt.title(title)
        plt.show()

    def img_vocab(self, vocabulary, class_names, inputs, classes):
        img_vocabulary = {}
        img_labels = []
        for label, img in zip(class_names, inputs):
            if classes[label] in vocabulary:
                img_vocabulary[img] = vocabulary[classes[label]]
                img_labels.append(label)
        imagenet
        return img_vocabulary

    def map_word_to_image(self, class_names, word_vocab, classes):
        vocab_positions = {}
        temp = []
        for i, label in enumerate(class_names):
            if classes[label] in word_vocab:
                print("map word: ", classes[label])
                if classes[label] not in temp:
                    temp.append(classes[label])
                    vocab_positions[classes[label]] = i
        for word in word_vocab:
            if word not in temp:
                vocab_positions[word] = None

        return vocab_positions


