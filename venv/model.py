from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F


class Word2Vec_neg_sampling(nn.Module):

    def __init__(self, embedding_size, vocab_size, device, noise_dist=None, negative_samples=10):
        super(Word2Vec_neg_sampling, self).__init__()

        self.embeddings_input = nn.Embedding(vocab_size, embedding_size)
        self.embeddings_context = nn.Embedding(vocab_size, embedding_size)
        self.vocab_size = vocab_size
        self.negative_samples = negative_samples
        self.device = device
        self.noise_dist = noise_dist

        # Initialize both embedding tables with uniform distribution
        self.embeddings_input.weight.data.uniform_(-1, 1)
        self.embeddings_context.weight.data.uniform_(-1, 1)

    def forward(self, input_word, context_word, img, samples):
        debug = not True
        if debug:
            print('input_word.shape: ', input_word.shape)  # bs
            print('context_word.shape: ', context_word.shape)  # bs

        # computing out loss
        emb_input = self.embeddings_input(input_word)  # bs, emb_dim
        if debug: print('emb_input.shape: ', emb_input.shape)

        emb_context = self.embeddings_context(context_word)  # bs, emb_dim
        if debug: print('emb_context.shape: ', emb_context.shape)

        emb_product = torch.mul(emb_input, emb_context)  # bs, emb_dim
        if debug: print('emb_product.shape: ', emb_product.shape)

        emb_product = torch.sum(emb_product, dim=1)  # bs
        if debug: print('emb_product.shape: ', emb_product.shape)

        out_loss = F.logsigmoid(emb_product)  # bs
        if debug: print('out_loss.shape: ', out_loss.shape)

        loss_visual = self.visual(emb_input, img, samples)
        if self.negative_samples > 0:
            # computing negative loss
            if self.noise_dist is None:
                noise_dist = torch.ones(self.vocab_size)
            else:
                noise_dist = self.noise_dist

            if debug: print('noise_dist.shape: ', noise_dist.shape)

            num_neg_samples_for_this_batch = context_word.shape[0] * self.negative_samples
            negative_example = torch.multinomial(noise_dist, num_neg_samples_for_this_batch,
                                                 replacement=True)  # coz bs*num_neg_samples > vocab_size
            if debug: print('negative_example.shape: ', negative_example.shape)

            negative_example = negative_example.view(context_word.shape[0], self.negative_samples).to(
                self.device)  # bs, num_neg_samples
            if debug: print('negative_example.shape: ', negative_example.shape)

            emb_negative = self.embeddings_context(negative_example)  # bs, neg_samples, emb_dim
            if debug: print('emb_negative.shape: ', emb_negative.shape)

            if debug: print('emb_input.unsqueeze(2).shape: ', emb_input.unsqueeze(2).shape)  # bs, emb_dim, 1
            emb_product_neg_samples = torch.bmm(emb_negative.neg(), emb_input.unsqueeze(2))  # bs, neg_samples, 1
            if debug: print('emb_product_neg_samples.shape: ', emb_product_neg_samples.shape)

            noise_loss = F.logsigmoid(emb_product_neg_samples).squeeze(2).sum(1)  # bs
            if debug: print('noise_loss.shape: ', noise_loss.shape)

            total_loss = -(out_loss + noise_loss).mean()
            if debug: print('total_loss.shape: ', total_loss.shape)

            total_loss = total_loss + loss_visual

            return total_loss

        else:
            if loss_visual is not 0:

                total_loss2 = -out_loss.mean() + loss_visual
                return total_loss2
            else:
                return -out_loss.mean()

    def visual(self, input_word, img, samples):
        cos = nn.CosineSimilarity()
        tmpLoss = 0
        loss = 0
        if img.sum().data != 0:
            cos_pos = cos(input_word, img)
            for v_n in samples:
                cos_neg = cos(input_word, v_n)
                tmpLoss = tmpLoss + cos_neg
            max_loss = cos_pos - cos_neg
            loss = torch.sum(max_loss)
        return -loss

