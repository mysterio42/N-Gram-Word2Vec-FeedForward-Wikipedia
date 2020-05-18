import sys
from datetime import datetime

import numpy as np
from scipy.special import expit as sigmoid

from models.embedding import WikipediaEmbedding
from utils.model import neg_sampling_distribution, word_context, dump_model, drop_words
from utils.plot import plot_costs


class FeedForward:
    def __init__(self, emb: WikipediaEmbedding, hidden_dim=50):
        self.embedding = emb

        self.vocab_size = len(self.embedding.word2idx)
        self.hidden_dim = hidden_dim

        self.W = np.random.randn(self.vocab_size, self.hidden_dim)
        self.V = np.random.randn(self.hidden_dim, self.vocab_size)

    def _forward(self, encoded_word, encoded_context_words):
        return sigmoid(self.W[encoded_word].dot(self.V[:, encoded_context_words]))

    def _backward(self, encoded_word, encoded_context_words, prob, label):
        gV = np.outer(self.W[encoded_word], prob - label)
        gW = np.sum((prob - label) * self.V[:, encoded_context_words], axis=1)
        return gV, gW

    def _step(self, encoded_word, encoded_context_words, grads, learning_rate):
        gV, gW = grads
        self.V[:, encoded_context_words] -= learning_rate * gV
        self.W[encoded_word] -= learning_rate * gW

    def _loss(self, prob, label):
        cost = label * np.log(prob + 1e-10) + (1 - label) * np.log(1 - prob + 1e-10)
        return cost.sum()

    def train(self, window_size=5, threshold=1e-5, learning_rate=0.025, final_learning_rate=0.0001, epochs=20):

        total_words = sum(len(sentence) for sentence in self.embedding.encoded_sentences)
        print(f'total number of words in corpus:{total_words}')

        costs = []

        learning_rate_delta = (learning_rate - final_learning_rate) / epochs

        p_neg = neg_sampling_distribution(self.embedding.encoded_sentences, self.vocab_size)
        p_drop = 1 - np.sqrt(threshold / p_neg)

        for epoch in range(epochs):
            np.random.shuffle(self.embedding.encoded_sentences)
            cost = 0
            t0 = datetime.now()
            for idx, sentence in enumerate(self.embedding.encoded_sentences):
                sentence = drop_words(sentence, p_drop)
                sen_len = len(sentence)
                if sen_len < 2:
                    continue

                for pos in np.random.choice(sen_len, size=sen_len, replace=False):
                    encoded_word, encoded_neg_word = sentence[pos], np.random.choice(self.vocab_size, p=p_neg)
                    encoded_context_words = word_context(pos, sentence, window_size)

                    for word, label in [(encoded_word, 1), (encoded_neg_word, 0)]:
                        prob = self._forward(word, encoded_context_words)
                        grads = self._backward(word, encoded_context_words, prob, label)
                        self._step(word, encoded_context_words, grads, learning_rate)
                        cost += self._loss(prob, label)

                if idx % 100 == 0:
                    sys.stdout.write(f'processed {idx} / {len(self.embedding.encoded_sentences)} epoch: {epoch}/{epochs}  \r')
                    sys.stdout.flush()

            print(f'epoch complete:{epoch} cost:{cost} dt:{datetime.now() - t0}')
            costs.append(cost)
            learning_rate -= learning_rate_delta

        plot_costs(costs)
        dump_model(self.embedding.word2idx, self.W, self.V)
