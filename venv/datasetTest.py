from __future__ import print_function

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
lem = WordNetLemmatizer()
import random
from imageDataset import img_dataset
from imageModel import Net
import numpy as np
from torch.autograd import Variable
from torchvision import transforms, datasets, models
import torch
from torch.autograd import Variable
TRAINING_TEXT = "corpus.txt"


class word2vec_datasetTest(Dataset):
    def __init__(self, DATA_SOURCE, CONTEXT_SIZE, FRACTION_DATA, SUBSAMPLING, SAMPLING_RATE, k):

        print("Parsing text and loading training data...")
        vocab, word_to_ix, ix_to_word, training_data, images, word_to_image = self.load_data(DATA_SOURCE, CONTEXT_SIZE, FRACTION_DATA, SUBSAMPLING, SAMPLING_RATE, k)

        self.vocab = vocab
        self.word_to_ix = word_to_ix
        self.ix_to_word = ix_to_word
        self.data = training_data
        self.images = images
        self.word_to_image = word_to_image

    def __getitem__(self, index):
        x = self.data[index][0]
        y = self.data[index][1]
        img = self.data[index][2]
        samples = self.data[index][3]
        return x, y, img, samples

    def __len__(self):
        return len(self.data)

    def tokenize_corpus(self, corpus):
        tokens = [x.split() for x in corpus]
        return tokens

    def load_data(self, data_source, context_size, fraction_data, subsampling, sampling_rate, k):
        stop_words = set(stopwords.words('english'))
        if data_source == 'gensim':
            import gensim.downloader as api
            dataset = api.load("text8")

            with open(TRAINING_TEXT, 'w') as outfile:
                for idx, doc in enumerate(dataset):
                    if idx == 100:
                        break
                    outfile.write(" ".join(doc) + "\n")
            data = [d for d in TRAINING_TEXT][:int(fraction_data * len([d_ for d_ in TRAINING_TEXT]))]
            #data = [d for d in dataset][:int(fraction_data * len([d_ for d_ in dataset]))]
            print(f'fraction of data taken: {fraction_data}/1')

            sents = []
            print('forming sentences by joining tokenized words...')
            for d in tqdm(data):
                sents.append(' '.join(d))

        sent_list_tokenized = [word_tokenize(s) for s in sents]
        print('len(sent_list_tokenized): ', len(sent_list_tokenized))

        # remove the stopwords
        sent_list_tokenized_filtered = []
        print('lemmatizing and removing stopwords...')
        for s in tqdm(sent_list_tokenized):
            sent_list_tokenized_filtered.append([lem.lemmatize(w, 'v') for w in s if w not in stop_words])

        sent_list_tokenized_filtered, vocab, word_to_ix, ix_to_word = self.gather_word_freqs(
            sent_list_tokenized_filtered, subsampling, sampling_rate)
        images = img_dataset(vocab)
        word_2_img = images.word_to_img
        #img_model = Net()
        #outputs = img_model(images.inputs)
        training_data = self.gather_training_data(sent_list_tokenized_filtered, word_to_ix, ix_to_word, context_size, images.inputs, word_2_img, k)
        print("return load data")
        return vocab, word_to_ix, ix_to_word, training_data, images, word_2_img

    def gather_training_data(self, tokenized_corpus, word_to_ix, ix_to_word, context_size, images, word_2_img, k):
        idx_pairs = []
        # for each sentence
        for i, sentence in enumerate(tokenized_corpus):
            print("Sentence ", i, " out of: ", len(tokenized_corpus))
            indices = [word_to_ix[word] for word in sentence]
            # for each word, threated as center word
            for center_word_pos in range(len(indices)):
                # for each window position
                for w in range(-context_size, context_size + 1):
                    context_word_pos = center_word_pos + w
                    # make soure not jump out sentence
                    if context_word_pos < 0 or context_word_pos >= len(indices) or center_word_pos == context_word_pos:
                        continue
                    context_word_idx = indices[context_word_pos]
                    #print("---t1")
                    img = torch.zeros(70)
                    samples = self.create_negative_vision(images, k)
                    img_idx = word_2_img[ix_to_word[indices[center_word_pos]]]
                    if img_idx is not None:
                        img = images[img_idx]
                    training_data_center = torch.tensor(indices[center_word_pos], dtype=torch.long)
                    training_data_context = torch.tensor(context_word_idx, dtype=torch.long)
                    #print("t2")
                    idx_pairs.append((Variable(training_data_center), Variable(training_data_context), Variable(img), samples))

        print("return gather data")
        return idx_pairs

    def gather_word_freqs(self, split_text, subsampling, sampling_rate):  # here split_text is sent_list

        vocab = {}
        ix_to_word = {}
        word_to_ix = {}
        total = 0.0

        print('building vocab...')
        for word_tokens in tqdm(split_text):
            for word in word_tokens:  # for every word in the word list(split_text), which might occur multiple times
                if word not in vocab:  # only new words allowed
                    vocab[word] = 0
                    ix_to_word[len(word_to_ix)] = word
                    word_to_ix[word] = len(word_to_ix)
                vocab[word] += 1.0  # count of the word stored in a dict
                total += 1.0  # total number of words in the word_list(split_text)

        print('\nsubsampling: ', subsampling)
        if subsampling:

            print('performing subsampling...')
            for sent in tqdm(split_text):
                word_tokens = sent
                for i, word in enumerate(word_tokens):
                    frac = vocab[word] / total
                    prob = 1 - np.sqrt(sampling_rate / frac)
                    sampling = np.random.sample()
                    if sampling < prob:
                        del word_tokens[i]
                        i -= 1

        print("Done gather frequ")
        return split_text, vocab, word_to_ix, ix_to_word

    def create_negative_vision(self, images, kk):
        test = random.choices(images, k=kk)
        return test
