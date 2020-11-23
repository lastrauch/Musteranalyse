import torch
from torchtext import data
from torchtext import datasets

TEXT = data.Field(lower=True)

train, valid, test = datasets.WikiText103.splits(TEXT)

print(train)