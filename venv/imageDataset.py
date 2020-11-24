import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms, datasets, models
import torch
from torch.autograd import Variable
from os import listdir, makedirs, getcwd, remove
from os.path import isfile, join, abspath, exists, isdir, expanduser
from PIL import Image
import torch
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, datasets, models
import torch.nn as nn
import torch.optim as optim

class img_dataset(Dataset):
    def __init__(self, vocab):
        train, test, classes, inputs, class_names, img_vocab, word_to_img = self.load_data(vocab)

        self.train = train
        self.test = test
        self.classes = classes
        self.inputs = inputs
        self.class_names = class_names
        self.img_vocab = img_vocab
        self.word_to_img = word_to_img

    def __len__(self):
        return self.inputs.shape[1]

    def load_data(self, vocab):
        train, test, classes = self.transforming()
        outputs = self.train_model(train)
        inputs, class_names = next(iter(train))
        #img_vocab = self. img_vocab(vocab, class_names, inputs, classes)
        #img_vocab = self.img_vocab(vocab, class_names, outputs, classes)
        word_to_img = self.map_word_to_image(class_names, vocab, classes)

        return train, test, classes, outputs, class_names, img_vocab, word_to_img

    def transforming(self):
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=1700, shuffle=True, num_workers=0)
        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(trainset, batch_size=1700, shuffle=True, num_workers=0)
        class_names = trainset.classes

        return trainloader, testloader, class_names


    def img_vocab(self, vocabulary, class_names, inputs, classes):
        print("ig vocab")
        img_vocabulary = {}
        img_labels = []
        for label, img in zip(class_names, inputs):
            if classes[label] in vocabulary:
                img_vocabulary[img] = vocabulary[classes[label]]
                img_labels.append(label)
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

    def train_model(self, trainloader):
        model = models.resnet34(pretrained=True)

        # Device configuration
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        for param in model.parameters():
            param.requires_grad = False

        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 128)
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

        for epoch in range(1):
            print("epoch: ", epoch)
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                print(i," out of ", len(trainloader))
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
            model.train()
            print(running_loss)
        print('Finished Training')
        return model(self.inputs)




