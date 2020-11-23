from torch.utils.data import Dataset, DataLoader
import torch
import random
import numpy as np

class MappedDataset(Dataset):
    def __init__(self, vocab, images, word_to_image, k):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.vocab = vocab
        self.images = images
        self.word_to_image = word_to_image
        self.k = k

    def __len__(self):
        return len(self.vocab)

    def __getitem__(self, i):
        vocabu = list(self.vocab.keys())
        samples = self.create_negative_vision(self.k, self.images)
        word = vocabu[i]
        img_idx = self.word_to_image[word]
        img = torch.zeros((3, 32, 32))
        if img_idx is not None:
            img = self.images[img_idx]
        samples = torch.stack(samples)

        return word, img, samples

    def create_negative_vision(self, K, images):
        return random.choices(images, k=K)
