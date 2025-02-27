from __future__ import print_function
from tqdm import tqdm
import numpy as np
import shutil, pickle
import torch.optim as optim
from model import Word2Vec_neg_sampling
from utils_modified import count_parameters
from dataset import word2vec_dataset
from config import *
from imageDataset import img_dataset
from test import print_nearest_words


# ======================================= make training data ===========================================================
test_corpus = [
    'cat is a female',
    'dog is male',
    'truck is for a man',
    'man is male',
    'horse is for a woman',
    'woman is female',
]

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

train_dataset = word2vec_dataset(DATA_SOURCE, CONTEXT_SIZE, FRACTION_DATA, SUBSAMPLING, SAMPLING_RATE, 1)
# ======================================================================================================================

vocab = train_dataset.vocab
test = []
word_to_ix = train_dataset.word_to_ix
ix_to_word = train_dataset.ix_to_word
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = not True)

print('len(train_dataset): ', len(train_dataset))
print('len(train_loader): ', len(train_loader))
print('len(vocab): ', len(vocab), '\n')
# ======================================================================================================================

# ==================== make noise distribution to sample negative examples from ========================================
word_freqs = np.array(list(vocab.values()))
unigram_dist = word_freqs/sum(word_freqs)
noise_dist = torch.from_numpy(unigram_dist**(0.75)/np.sum(unigram_dist**(0.75)))
losses = []

model = Word2Vec_neg_sampling(EMBEDDING_DIM, len(vocab), DEVICE, noise_dist, NEGATIVE_SAMPLES).to(DEVICE)
# ======================================================================================================================

print('\nWe have {} Million trainable parameters here in the model'.format(count_parameters(model)))

optimizer = optim.Adam(model.parameters(), lr = LR)

for epoch in tqdm(range(NUM_EPOCHS)):  # NUM_EPOCHS
    print('\n===== EPOCH {}/{} ====='.format(epoch + 1, NUM_EPOCHS))

    for batch_idx, (x_batch, y_batch, img, samples) in enumerate(train_loader):
        print('batch# ' + str(batch_idx + 1).zfill(len(str(len(train_loader)))) + '/' + str(len(train_loader)), end='\r')

        model.train()

        x_batch = x_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)
        img = img.to(DEVICE)

        optimizer.zero_grad()
        loss = model(x_batch, y_batch, img, samples)

        loss.backward(retain_graph=True)
        optimizer.step()

        losses.append(loss.item())

        if batch_idx % DISPLAY_EVERY_N_BATCH == 0 and DISPLAY_BATCH_LOSS:
            print(f'Batch: {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item()}')
            # show 5 closest words to some test words
            print_nearest_words(model, TEST_WORDS, word_to_ix, ix_to_word, top=5)

    # ========================== write embeddings every SAVE_EVERY_N_EPOCH epoch =======================================
    if epoch % SAVE_EVERY_N_EPOCH == 0:
        torch.save({'model_state_dict': model.state_dict(),
                    'losses': losses,
                    'word_to_ix': word_to_ix,
                    'ix_to_word': ix_to_word
                    },
                   '{}/model{}.pth'.format(MODEL_DIR, epoch))
