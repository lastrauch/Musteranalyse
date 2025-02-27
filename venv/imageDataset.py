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
import os
import pickle

class img_dataset(Dataset):
    def __init__(self, vocab):
        train, classes, inputs, class_names, word_to_img, model = self.load_data(vocab)

        self.train = train
        self.classes = classes
        self.inputs = inputs
        self.class_names = class_names
        self.word_to_img = word_to_img
        self.model = model

    def __len__(self):
        return self.inputs.shape[1]

    def load_data(self, vocab):
        train, classes = self.transforming()
        if os.path.exists("/content/Musteranalyse/venv/model_15.pt"):
            print('\nloading pickle...')
            infile = open("/content/Musteranalyse/venv/model_15.pt",'rb')
            model = pickle.load(infile)
            infile.close()
            print('pickle loaded\n')
        else:
            model = self.train_model(train)
        inputs, class_names = next(iter(train))
        word_to_img = self.map_word_to_image(class_names, vocab, classes)

        return train, classes, inputs, class_names, word_to_img, model

    def transforming(self):
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=1000, shuffle=True, num_workers=0)
        class_names = trainset.classes

        return trainloader, class_names


    def img_vocab(self, vocabulary, class_names, inputs, classes):
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

        for epoch in range(15):
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
                print(running_loss)
        if not os.path.exists("/content/Musteranalyse/venv/model_15.pt"):
            PATH = "/content/Musteranalyse/venv/model_15.pt"
            # ======== pickle dump =========
            print('\ndumping pickle...')
            outfile = open(PATH, 'wb')
            pickle.dump(model, outfile)
            outfile.close()
            print('pickle dumped\n')
        print('Finished Training')
        return model


